import os
import uuid
import time
import random
import requests
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text

# =====================================================================
# CONFIGURATION & SETUP
# =====================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [HES AMISP-1] - %(message)s')
app = Flask(__name__)

# STRICT ISOLATION CONSTANT
TARGET_AMISP = "AMISP-1"

POSTGRES_API_URL = os.getenv("POSTGRES_API_URL")
MDM_BULK_WEBHOOK_URL = os.getenv("MDM_BULK_WEBHOOK_URL", "https://mdm-ou39.onrender.com/api/v1/callbacks/hes-status/bulk")
MDM_PRIORITY_RC_WEBHOOK_URL = os.getenv("MDM_PRIORITY_RC_WEBHOOK_URL", "https://mdm-ou39.onrender.com/api/v1/callbacks/hes-status/rc-priority")

if not POSTGRES_API_URL:
    logging.critical("🚨 POSTGRES_API_URL is missing!")
    exit(1)

pg_engine = create_engine(POSTGRES_API_URL, pool_size=10, max_overflow=20)

def init_db():
    with pg_engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS amisp1_pending_commands (
                hes_tx_id VARCHAR(50) PRIMARY KEY,
                reference_id VARCHAR(50),
                meter_no VARCHAR(50),
                command VARCHAR(20),
                status VARCHAR(20) DEFAULT 'PENDING',
                is_notified BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS amisp1_consumer_edge_state (
                meter_no VARCHAR(50) PRIMARY KEY,
                consumer_type VARCHAR(20) DEFAULT 'NORMAL',
                connection_status VARCHAR(5) DEFAULT 'C',
                curr_import_total REAL DEFAULT 0.0,
                prev_import_total REAL DEFAULT 0.0,
                curr_import_tz1 REAL DEFAULT 0.0,
                curr_import_tz2 REAL DEFAULT 0.0,
                curr_import_tz3 REAL DEFAULT 0.0,
                curr_export_total REAL DEFAULT 0.0,
                prev_export_total REAL DEFAULT 0.0,
                last_read_date DATE
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS amisp1_monthly_settlement (
                meter_no VARCHAR(50),
                billing_month VARCHAR(10),
                start_import_total REAL DEFAULT 0.0,
                end_import_total REAL DEFAULT 0.0,
                start_export_total REAL DEFAULT 0.0,
                end_export_total REAL DEFAULT 0.0,
                PRIMARY KEY (meter_no, billing_month)
            )
        """))
    logging.info("✅ PostgreSQL Initialized. AMISP-1 Isolation Guaranteed.")

init_db()

@app.route('/', methods=['GET'])
def health_check():
    return f"HES Edge Node [{TARGET_AMISP}] is Live. Database: PostgreSQL.", 200

# =====================================================================
# PART 1: LIVE RC-DC PIPELINE (ASYNC & BULK)
# =====================================================================
@app.route('/api/v1/commands/relay-state/bulk', methods=['POST'])
def receive_commands_bulk():
    commands = request.json.get('commands', [])
    if not commands: return jsonify({"error": "No commands provided."}), 400

    insert_data = []
    for cmd in commands:
        hes_tx_id = f"HES_TX_{uuid.uuid4().hex[:8].upper()}"
        insert_data.append({
            "hes_tx": hes_tx_id, "ref_id": cmd['reference_id'], 
            "m_no": cmd['meter_no'], "cmd": cmd['command']
        })
    
    CHUNK_SIZE = 1000
    with pg_engine.begin() as conn:
        for i in range(0, len(insert_data), CHUNK_SIZE):
            chunk = insert_data[i:i + CHUNK_SIZE]
            conn.execute(text("""
                INSERT INTO amisp1_pending_commands (hes_tx_id, reference_id, meter_no, command) 
                VALUES (:hes_tx, :ref_id, :m_no, :cmd)
            """), chunk)
            
    logging.info(f"📥 Queued {len(insert_data)} Bulk Commands in Postgres.")
    return jsonify({"status": "QUEUED", "count": len(insert_data)}), 202


@app.route('/internal/process-queue', methods=['GET'])
def process_queue():
    """Worker 1: Simulates RF network execution (Lightning Fast Update)"""
    with pg_engine.connect() as conn:
        commands = conn.execute(text("""
            SELECT hes_tx_id, meter_no, command 
            FROM amisp1_pending_commands 
            WHERE status = 'PENDING' LIMIT 1000
        """)).mappings().fetchall()
        
    if not commands:
        return jsonify({"message": "RC-DC Queue empty."}), 200

    # ⚡ OPTIMIZATION: Extract arrays to run exactly 3 queries instead of 2000
    all_tx_ids = [r['hes_tx_id'] for r in commands]
    meters_to_disconnect = [r['meter_no'] for r in commands if r['command'] == 'DISCONNECT']
    meters_to_reconnect = [r['meter_no'] for r in commands if r['command'] == 'RECONNECT']

    with pg_engine.begin() as conn:
        # 1. Update tracking log in one shot
        tx_list_str = "', '".join(all_tx_ids)
        conn.execute(text(f"UPDATE amisp1_pending_commands SET status = 'COMPLETED' WHERE hes_tx_id IN ('{tx_list_str}')"))
        
        # 2. Update physical state for Disconnects in one shot
        if meters_to_disconnect:
            dc_list_str = "', '".join(meters_to_disconnect)
            conn.execute(text(f"UPDATE amisp1_consumer_edge_state SET connection_status = 'D' WHERE meter_no IN ('{dc_list_str}')"))
            
        # 3. Update physical state for Reconnects in one shot
        if meters_to_reconnect:
            rc_list_str = "', '".join(meters_to_reconnect)
            conn.execute(text(f"UPDATE amisp1_consumer_edge_state SET connection_status = 'C' WHERE meter_no IN ('{rc_list_str}')"))
            
    logging.info(f"⚡ Processed {len(commands)} RC-DC physical commands instantly.")
    return jsonify({"message": f"Processed {len(commands)} RC-DC commands over RF."}), 200

@app.route('/internal/push-bulk-callbacks', methods=['GET'])
def push_bulk_callbacks():
    """Worker 2: Pushes async responses back to MDM via Dual-Lanes."""
    rc_response = push_lane_to_mdm("RECONNECT", MDM_PRIORITY_RC_WEBHOOK_URL)
    dc_response = push_lane_to_mdm("DISCONNECT", MDM_BULK_WEBHOOK_URL)
    
    return jsonify({
        "priority_rc_status": rc_response,
        "bulk_dc_status": dc_response
    }), 200

@app.route('/internal/retry-failed-callbacks', methods=['GET'])
def retry_failed_callbacks():
    logging.info("♻️ Initiating Manual Retry for Unsent Callbacks...")
    return push_bulk_callbacks()

def push_lane_to_mdm(command_type, target_url):
    """Generic pusher that filters by command type (RC vs DC)"""
    with pg_engine.connect() as conn:
        unsent = conn.execute(text(f"""
            SELECT hes_tx_id, reference_id, meter_no, command 
            FROM amisp1_pending_commands 
            WHERE status = 'COMPLETED' AND is_notified = FALSE AND command = '{command_type}'
            LIMIT 1000
        """)).mappings().fetchall()

    if not unsent:
        return f"No {command_type}s pending."

    bulk_payload = {
        "results": [{
            "hes_transaction_id": r['hes_tx_id'], "reference_id": r['reference_id'], 
            "meter_no": r['meter_no'], "command": r['command'], "status": "SUCCESS"
        } for r in unsent]
    }

    try:
        logging.info(f"🚀 Pushing {len(unsent)} {command_type}s to {target_url}...")
        resp = requests.post(target_url, json=bulk_payload, timeout=30)
        
        if resp.status_code == 200:
            tx_ids = [r['hes_tx_id'] for r in unsent]
            tx_list_str = "', '".join(tx_ids)
            
            with pg_engine.begin() as conn:
                conn.execute(text(f"UPDATE amisp1_pending_commands SET is_notified = TRUE WHERE hes_tx_id IN ('{tx_list_str}')"))
                
            logging.info(f"✅ {command_type} Lane: Success.")
            return f"Pushed {len(unsent)} records."
        else:
            logging.error(f"❌ MDM rejected {command_type} payload: {resp.text}")
            return f"MDM Rejected (HTTP {resp.status_code})"
            
    except requests.exceptions.RequestException as e:
        logging.error(f"🚨 Network failure on {command_type} Lane: {e}")
        return f"Network Failure: {str(e)}"
# =====================================================================
# PART 2: THE GENERATION WORKER (Midnight Block Load)
# =====================================================================
@app.route('/internal/worker/generate-reads', methods=['GET'])
def generate_daily_reads():
    target_date = datetime.now().date()
    target_date_str = target_date.strftime("%Y-%m-%d")

    with pg_engine.connect() as conn:
        consumers = conn.execute(text("SELECT * FROM amisp1_consumer_edge_state")).mappings().fetchall()
        
    update_payloads = []
    
    for c in consumers:
        if c['connection_status'] == 'D':
            imp_inc, exp_inc = 0.0, 0.0
        else:
            imp_inc = round(random.uniform(5.0, 15.0), 2)
            exp_inc = round(random.uniform(2.0, 8.0), 2) if c['consumer_type'] == 'NET_METER' else 0.0

        update_payloads.append({
            "m_no": c['meter_no'],
            "new_imp": round(c['curr_import_total'] + imp_inc, 2),
            "tz1": round(c['curr_import_tz1'] + (imp_inc * 0.4), 2),
            "tz2": round(c['curr_import_tz2'] + (imp_inc * 0.4), 2),
            "tz3": round(c['curr_import_tz3'] + (imp_inc * 0.2), 2),
            "new_exp": round(c['curr_export_total'] + exp_inc, 2),
            "dt": target_date_str
        })

    CHUNK_SIZE = 1000
    with pg_engine.begin() as conn:
        for i in range(0, len(update_payloads), CHUNK_SIZE):
            chunk = update_payloads[i:i + CHUNK_SIZE]
            conn.execute(text("""
                UPDATE amisp1_consumer_edge_state 
                SET prev_import_total = curr_import_total, 
                    prev_export_total = curr_export_total,
                    curr_import_total = :new_imp, curr_import_tz1 = :tz1, 
                    curr_import_tz2 = :tz2, curr_import_tz3 = :tz3, 
                    curr_export_total = :new_exp, last_read_date = :dt
                WHERE meter_no = :m_no
            """), chunk)

    return jsonify({"message": f"Successfully generated reads for {len(consumers)} meters."}), 200

# =====================================================================
# PART 3: MASTER DATA SYNC & EXPORT APIs
# =====================================================================
@app.route('/api/v1/sync/meters-bulk', methods=['POST'])
def sync_meters_bulk():
    meters = request.json.get('meters', [])
    if not meters: return jsonify({"error": "No meters provided."}), 400

    valid_meters = [m for m in meters if m.get('amisp_name') == TARGET_AMISP]
    if not valid_meters:
        return jsonify({"message": f"Ignored payload. No {TARGET_AMISP} meters found."}), 200

    state_data = []
    for m in valid_meters:
        b_imp = float(m.get('baseline_kwh', 0.0))
        b_exp = float(m.get('baseline_export_kwh', 0.0))
        state_data.append({
            "m_no": m['meter_no'], "c_type": m.get('consumer_type', 'NORMAL'), 
            "stat": m.get('connection_status', 'C'), "imp": b_imp, "exp": b_exp
        })

    CHUNK_SIZE = 1000
    with pg_engine.begin() as conn:
        for i in range(0, len(state_data), CHUNK_SIZE):
            chunk = state_data[i:i + CHUNK_SIZE]
            conn.execute(text("""
                INSERT INTO amisp1_consumer_edge_state (
                    meter_no, consumer_type, connection_status, 
                    curr_import_total, prev_import_total, curr_export_total, prev_export_total
                ) VALUES (
                    :m_no, :c_type, :stat, :imp, :imp, :exp, :exp
                )
                ON CONFLICT (meter_no) DO UPDATE SET 
                    consumer_type = EXCLUDED.consumer_type, 
                    connection_status = EXCLUDED.connection_status,
                    curr_import_total = EXCLUDED.curr_import_total, 
                    prev_import_total = EXCLUDED.prev_import_total,
                    curr_export_total = EXCLUDED.curr_export_total, 
                    prev_export_total = EXCLUDED.prev_export_total
            """), chunk)
            
    return jsonify({"message": f"Successfully synced {len(valid_meters)} {TARGET_AMISP} meters."}), 200

@app.route('/api/v1/export/daily-reads', methods=['GET'])
def export_daily():
    limit = int(request.args.get('limit', 1000))
    offset = int(request.args.get('offset', 0))
    if limit > 1000: return jsonify({"error": "PAYLOAD_TOO_LARGE"}), 413
    
    with pg_engine.connect() as conn:
        data = conn.execute(text("SELECT * FROM amisp1_consumer_edge_state LIMIT :l OFFSET :o"), {"l": limit, "o": offset}).mappings().fetchall()
    
    return jsonify({"count": len(data), "limit": limit, "offset": offset, "data": [dict(r) for r in data]}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
