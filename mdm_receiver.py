import os
import uuid
import requests
import logging
import threading
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MDM SERVER] - %(message)s')

app = Flask(__name__)

# Cloud Database Credentials
POSTGRES_API_URL = os.environ.get("POSTGRES_API_URL")
MYSQL_MASTER_URL = os.environ.get("MYSQL_MASTER_URL")

# The URL to push Real-Time commands to the HES
HES_BULK_API_URL = os.environ.get("HES_BULK_API_URL", "https://hes-2.onrender.com/api/v1/commands/relay-state/bulk")

if not POSTGRES_API_URL or not MYSQL_MASTER_URL:
    logging.critical("Database URLs missing! Ensure environment variables are set.")
    exit(1)

pg_engine = create_engine(POSTGRES_API_URL, pool_size=10, max_overflow=20,max_overflow=20,pool_recycle=280,pool_pre_ping=True)
mysql_engine = create_engine(MYSQL_MASTER_URL, pool_size=10, max_overflow=20)

session = requests.Session()
session.trust_env = False

@app.route('/', methods=['GET'])
def health_check():
    return "MDM Unified Server is awake and listening!", 200

# =====================================================================
# BACKGROUND WORKER (For Bulk Processing)
# =====================================================================
def process_bulk_callback_in_background(records):
    """This runs silently in the background after the MDM has already replied to the HES."""
    logging.info(f"⚙️ BACKGROUND TASK: Processing {len(records)} meter updates...")
    
    pg_updates = []
    dc_meters = []
    rc_meters = []

    # 1. Parse Payload
    for data in records:
        execution_status = data.get('status') or data.get('execution_status')
        if not execution_status:
            continue

        pg_updates.append({
            "status": execution_status,
            "hes_tx": data.get('hes_transaction_id'),
            "cmd_id": data.get('reference_id')
        })

        if execution_status == "SUCCESS":
            command = data.get('command')
            if command == 'DISCONNECT':
                dc_meters.append(data.get('meter_no'))
            elif command == 'RECONNECT':
                rc_meters.append(data.get('meter_no'))

    # 2. Bulk Update PostgreSQL (Tracking Log)
    if pg_updates:
        try:
            with pg_engine.begin() as pg_conn:
                pg_conn.execute(
                    text("""
                        UPDATE dc_rc_log 
                        SET status = :status, 
                            hes_transaction_id = :hes_tx,
                            executed_at = NOW()
                        WHERE command_id = :cmd_id
                    """),
                    pg_updates
                )
        except Exception as e:
            logging.error(f"🚨 Background PostgreSQL Bulk Update Failed: {e}")

    # 3. LIGHTNING FAST Bulk Update MySQL (Master Data State)
    if dc_meters or rc_meters:
        try:
            with mysql_engine.begin() as mysql_conn:
                # Process all Disconnects in ONE query
                if dc_meters:
                    dc_str = "', '".join(dc_meters)
                    mysql_conn.execute(text(f"UPDATE consumer_master SET connection_status = 'D' WHERE meter_no IN ('{dc_str}')"))
                
                # Process all Reconnects in ONE query
                if rc_meters:
                    rc_str = "', '".join(rc_meters)
                    mysql_conn.execute(text(f"UPDATE consumer_master SET connection_status = 'C' WHERE meter_no IN ('{rc_str}')"))
                    
            logging.info(f"✅ BACKGROUND SUCCESS: MySQL Master Updated: {len(dc_meters)} DC, {len(rc_meters)} RC.")
        except Exception as e:
            logging.error(f"🚨 Background MySQL Bulk Update Failed: {e}")


# =====================================================================
# ENDPOINT 1: THE BULK CALLBACK RECEIVER (HES -> MDM)
# =====================================================================
@app.route('/api/v1/callbacks/hes-status/bulk', methods=['POST'])
def hes_status_callback_bulk():
    records = request.json.get('results', [])
    if not records:
        return jsonify({"error": "Empty or missing 'results' array"}), 400
        
    logging.info(f"📥 Received Bulk Callback payload for {len(records)} meters.")

    # 1. FIRE AND FORGET
    # We pass the records to the background worker and let it run independently
    thread = threading.Thread(target=process_bulk_callback_in_background, args=(records,))
    thread.start()

    # 2. INSTANT HTTP REPLY
    # The HES gets this instantly and hangs up the phone, completely avoiding timeouts.
    return jsonify({
        "message": "Payload accepted for background processing.",
        "queued_count": len(records)
    }), 202


# =====================================================================
# ENDPOINT 1.5: THE PRIORITY VIP LANE (RC CALLBACKS ONLY)
# =====================================================================
@app.route('/api/v1/callbacks/hes-status/rc-priority', methods=['POST'])
def hes_status_callback_rc_priority():
    records = request.json.get('results', [])
    if not records:
        return jsonify({"error": "Empty payload"}), 400
        
    logging.info(f"⚡ PRIORITY LANE: Received {len(records)} RECONNECT callbacks.")

    pg_updates = []
    rc_meters = []

    for data in records:
        if data.get('status') == "SUCCESS" and data.get('command') == 'RECONNECT':
            pg_updates.append({
                "status": "SUCCESS",
                "hes_tx": data.get('hes_transaction_id'),
                "cmd_id": data.get('reference_id')
            })
            rc_meters.append(data.get('meter_no'))

    # 1. Update Tracking Log
    if pg_updates:
        with pg_engine.begin() as pg_conn:
            pg_conn.execute(
                text("""
                    UPDATE dc_rc_log 
                    SET status = :status, hes_transaction_id = :hes_tx, executed_at = NOW()
                    WHERE command_id = :cmd_id
                """), pg_updates
            )

    # 2. Lightning Fast Master Update
    if rc_meters:
        rc_str = "', '".join(rc_meters)
        with mysql_engine.begin() as mysql_conn:
            mysql_conn.execute(text(f"UPDATE consumer_master SET connection_status = 'C' WHERE meter_no IN ('{rc_str}')"))
            
    logging.info(f"🎉 VIP Reconnects Complete: {len(rc_meters)} meters restored instantly.")
    return jsonify({"message": f"Priority RC Processed"}), 200


# =====================================================================
# ENDPOINT 2: REAL-TIME RC TRIGGER (PAYMENT GATEWAY -> MDM -> HES)
# =====================================================================
@app.route('/api/v1/trigger-realtime-rc', methods=['POST'])
def trigger_realtime_rc():
    recharged_meters = request.json.get('meters', [])
    if not recharged_meters:
        return jsonify({"error": "No meters provided."}), 400
    if len(recharged_meters) > 1000:
        return jsonify({"error": "Exceeded max batch size of 1000."}), 413

    logging.info(f"⚡ REAL-TIME EVENT: Received {len(recharged_meters)} meters for instant Reconnect.")

    meter_list_str = "', '".join(recharged_meters)
    
    with mysql_engine.connect() as mysql_conn:
        valid_targets = mysql_conn.execute(
            text(f"""
                SELECT consumer_no, meter_no 
                FROM consumer_master 
                WHERE meter_no IN ('{meter_list_str}') 
                AND connection_status = 'D'
                AND CURRENT_BALANCE_INR >= 0
            """)
        ).fetchall()

    if not valid_targets:
        return jsonify({"message": "No valid disconnected meters found for RC in this batch."}), 200

    hes_api_payloads = []
    pg_insert_values = []
    update_cmd_ids = []
    update_meter_nos = []

    for consumer_no, meter_no in valid_targets:
        command_id = f"CMD_RC_{uuid.uuid4().hex[:10].upper()}"
        
        hes_api_payloads.append({"reference_id": command_id, "meter_no": meter_no, "command": "RECONNECT"})
        pg_insert_values.append(f"('{command_id}', {consumer_no}, '{meter_no}', 'RECONNECT', 'PENDING', 'RECHARGE_CLEARED')")
        update_cmd_ids.append(command_id)
        update_meter_nos.append(meter_no)

    insert_query = f"""
        INSERT INTO public.dc_rc_log (command_id, consumer_no, meter_no, command_type, status, reason_code) 
        VALUES {', '.join(pg_insert_values)}
    """
    valid_meter_str = "', '".join(update_meter_nos)
    mysql_update_query = f"UPDATE consumer_master SET connection_status = 'U' WHERE meter_no IN ('{valid_meter_str}')"

    with pg_engine.begin() as pg_trans:
        pg_trans.execute(text(insert_query))
        
    with mysql_engine.begin() as mysql_trans:
        mysql_trans.execute(text(mysql_update_query))

    try:
        response = session.post(HES_BULK_API_URL, json={"commands": hes_api_payloads}, timeout=20)
        
        if response.status_code == 202:
            logging.info(f"✅ HES Accepted Real-Time RC for {len(valid_targets)} meters.")
            cmd_list_str = "', '".join(update_cmd_ids)
            update_query = f"UPDATE public.dc_rc_log SET status = 'SENT_TO_HES' WHERE command_id IN ('{cmd_list_str}')"
            with pg_engine.begin() as pg_trans:
                pg_trans.execute(text(update_query))
                
            return jsonify({"message": f"Successfully triggered RC for {len(valid_targets)} meters.", "queued_count": len(valid_targets)}), 200
        else:
            logging.error(f"❌ HES Rejected Real-Time RC: {response.text}")
            return jsonify({"error": "HES rejected the payload"}), 502
    except Exception as e:
        logging.error(f"🚨 Network error firing HES: {e}")
        return jsonify({"error": "Network failure reaching HES"}), 503


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
