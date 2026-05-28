import os
import logging
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MDM BULK RECEIVER] - %(message)s')

app = Flask(__name__)

# Cloud Database Credentials
POSTGRES_API_URL = os.environ.get("POSTGRES_API_URL")
MYSQL_MASTER_URL = os.environ.get("MYSQL_MASTER_URL")

if not POSTGRES_API_URL or not MYSQL_MASTER_URL:
    logging.critical("Database URLs missing! Ensure environment variables are set.")
    exit(1)

# Create engines with slightly larger pools for bulk operations
pg_engine = create_engine(POSTGRES_API_URL, pool_size=10, max_overflow=20)
mysql_engine = create_engine(MYSQL_MASTER_URL, pool_size=10, max_overflow=20)

@app.route('/', methods=['GET'])
def health_check():
    return "MDM Bulk Receiver is awake and listening!", 200

# --- HES BULK CALLBACK ROUTE ---
@app.route('/api/v1/callbacks/hes-status/bulk', methods=['POST'])
def hes_status_callback_bulk():
    """Accepts an array of completed commands from the HES and processes them in bulk."""
    payload = request.json
    
    # Expecting the HES to send data inside a "results" array
    records = payload.get('results', [])
    
    if not records:
        return jsonify({"error": "Empty or missing 'results' array"}), 400
        
    logging.info(f"📥 Received Bulk Callback containing {len(records)} meter updates.")

    pg_updates = []
    mysql_updates = []

    # 1. Parse Payload
    for data in records:
        execution_status = data.get('status') or data.get('execution_status')
        command = data.get('command')
        meter_no = data.get('meter_no')
        
        if not execution_status:
            continue

        # Prepare PostgreSQL Log Update
        pg_updates.append({
            "status": execution_status,
            "hes_tx": data.get('hes_transaction_id'),
            "cmd_id": data.get('reference_id')
        })

        # Prepare MySQL Master State Update (Only for SUCCESSful physical actions)
        if execution_status == "SUCCESS":
            if command == 'DISCONNECT':
                mysql_updates.append({"new_status": 'D', "meter_no": meter_no})
            elif command == 'RECONNECT':
                mysql_updates.append({"new_status": 'C', "meter_no": meter_no})

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
            logging.info(f"✅ PostgreSQL Log: Updated {len(pg_updates)} records.")
        except Exception as e:
            logging.error(f"🚨 PostgreSQL Bulk Update Failed: {e}")
            return jsonify({"error": "Failed to update tracking log"}), 500

    # 3. Bulk Update MySQL (Master Data State)
    if mysql_updates:
        try:
            with mysql_engine.begin() as mysql_conn:
                mysql_conn.execute(
                    text("""
                        UPDATE consumer_master 
                        SET connection_status = :new_status 
                        WHERE meter_no = :meter_no
                    """),
                    mysql_updates
                )
            logging.info(f"✅ MySQL Master: Physically updated state for {len(mysql_updates)} meters.")
        except Exception as e:
            logging.error(f"🚨 MySQL Bulk Update Failed: {e}")
            return jsonify({"error": "Failed to update master table"}), 500

    return jsonify({"message": f"Successfully processed {len(records)} callbacks."}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
