import os
import sqlite3
import uuid
import time
import requests
import logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [HES] - %(levelname)s - %(message)s')

app = Flask(__name__)
DB_FILE = "hes_queue.db"

def get_db_connection():
    """Creates a database connection that returns rows as Python dictionaries."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Creates the local SQLite database to store pending HES commands."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_commands (
                hes_tx_id TEXT PRIMARY KEY,
                reference_id TEXT,
                meter_no TEXT,
                command TEXT,
                callback_url TEXT,
                status TEXT DEFAULT 'PENDING',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    logging.info("HES Local SQLite Database Initialized.")

# Run DB init on startup
init_db()

# =====================================================================
# ENDPOINT 1: Receive Command from MDM
# =====================================================================
@app.route('/api/v1/commands/relay-state', methods=['POST'])
def receive_command():
    data = request.json
    hes_tx_id = f"HES_TX_{uuid.uuid4().hex[:8].upper()}"
    
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO pending_commands (hes_tx_id, reference_id, meter_no, command, callback_url)
            VALUES (?, ?, ?, ?, ?)
        """, (hes_tx_id, data['reference_id'], data['meter_no'], data['command'], data['callback_url']))
        conn.commit()
    
    logging.info(f"Command {data['command']} for meter {data['meter_no']} saved to local DB.")
    
    return jsonify({
        "status": "QUEUED",
        "hes_transaction_id": hes_tx_id,
        "message": "Command stored safely in HES local database."
    }), 202

# =====================================================================
# ENDPOINT 2: The Queue Processor (Trigger via Cron/Task)
# =====================================================================
@app.route('/internal/process-queue', methods=['GET', 'POST'])
def process_queue():
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM pending_commands WHERE status = 'PENDING' LIMIT 5")
        commands = cursor.fetchall()
        
    if not commands:
        return jsonify({"message": "Queue is empty. Nothing to process."}), 200

    processed_count = 0
    
    for row in commands:
        logging.info(f"Processing {row['command']} for Meter: {row['meter_no']}...")
        time.sleep(3) # Simulate RF ping
        
        # Build the exact payload the Render MDM expects
        callback_data = {
            "hes_transaction_id": row['hes_tx_id'],
            "reference_id": row['reference_id'],
            "meter_no": row['meter_no'],
            "command": row['command'],
            "status": "SUCCESS",  # FIXED: Matches Render exactly
            "execution_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            response = requests.post(row['callback_url'], json=callback_data, timeout=10)
            
            if response.status_code == 200:
                with get_db_connection() as update_conn:
                    update_conn.execute("UPDATE pending_commands SET status = 'COMPLETED' WHERE hes_tx_id = ?", (row['hes_tx_id'],))
                    update_conn.commit()
                logging.info(f"✅ Successfully completed {row['command']} and notified MDM.")
                processed_count += 1
            else:
                logging.error(f"❌ MDM returned status {response.status_code}. Keeping command as PENDING.")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"⚠️ Failed to reach MDM webhook URL: {e}")
            
    return jsonify({"message": f"Processed {processed_count} commands."}), 200

# =====================================================================
# ENDPOINT 3: OBSERVABILITY - Get ALL records (Limit 100)
# =====================================================================
@app.route('/api/v1/queue/all', methods=['GET'])
def get_all_queue():
    status_filter = request.args.get('status') # Optional: ?status=PENDING
    
    query = "SELECT * FROM pending_commands ORDER BY created_at DESC LIMIT 100"
    params = ()
    
    if status_filter:
        query = "SELECT * FROM pending_commands WHERE status = ? ORDER BY created_at DESC LIMIT 100"
        params = (status_filter.upper(),)
        
    with get_db_connection() as conn:
        cursor = conn.execute(query, params)
        records = [dict(row) for row in cursor.fetchall()]
        
    return jsonify({"count": len(records), "data": records}), 200

# =====================================================================
# ENDPOINT 4: OBSERVABILITY - Get records for a specific meter
# =====================================================================
@app.route('/api/v1/queue/meter/<meter_no>', methods=['GET'])
def get_meter_queue(meter_no):
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM pending_commands WHERE meter_no = ? ORDER BY created_at DESC", (meter_no,))
        records = [dict(row) for row in cursor.fetchall()]
        
    if not records:
        return jsonify({"message": f"No records found for meter {meter_no}"}), 404
        
    return jsonify({"count": len(records), "data": records}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)