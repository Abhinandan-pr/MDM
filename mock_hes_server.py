import os
import sqlite3
import uuid
import time
import random
import requests
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from flask import Flask, request, jsonify

# =====================================================================
# CONFIGURATION & SETUP
# =====================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [HES EDGE] - %(message)s')
app = Flask(__name__)
DB_FILE = "hes_edge_node.db"

# Pointing perfectly to your live Render MDM Bulk Receiver
MDM_BULK_WEBHOOK_URL = os.getenv("MDM_BULK_WEBHOOK_URL", "https://mdm-ou39.onrender.com/api/v1/callbacks/hes-status/bulk")

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        # 1. LIVE RC-DC ZONE
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_commands (
                hes_tx_id TEXT PRIMARY KEY,
                reference_id TEXT,
                meter_no TEXT,
                command TEXT,
                status TEXT DEFAULT 'PENDING',
                is_notified BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. PROVISIONING & STATE ZONE
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hes_provisioned_meters (
                meter_no TEXT PRIMARY KEY,
                amisp_name TEXT,
                status TEXT DEFAULT 'ACTIVE'
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS consumer_edge_state (
                meter_no TEXT PRIMARY KEY,
                consumer_type TEXT DEFAULT 'NORMAL',
                connection_status TEXT DEFAULT 'C',
                
                curr_import_total REAL DEFAULT 0.0,
                prev_import_total REAL DEFAULT 0.0,
                curr_import_tz1 REAL DEFAULT 0.0,
                curr_import_tz2 REAL DEFAULT 0.0,
                curr_import_tz3 REAL DEFAULT 0.0,
                
                curr_export_total REAL DEFAULT 0.0,
                prev_export_total REAL DEFAULT 0.0,
                
                last_read_date DATE
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hes_monthly_settlement (
                meter_no TEXT,
                billing_month TEXT,
                start_import_total REAL DEFAULT 0.0,
                end_import_total REAL DEFAULT 0.0,
                start_export_total REAL DEFAULT 0.0,
                end_export_total REAL DEFAULT 0.0,
                PRIMARY KEY (meter_no, billing_month)
            )
        """)
    logging.info("Edge Database Initialized. RC-DC Pipeline is secure.")

init_db()

@app.route('/', methods=['GET'])
def health_check():
    return "HES Edge Node is Live and Optimized for Bulk Operations.", 200

# =====================================================================
# PART 1: LIVE RC-DC PIPELINE (ASYNC & BULK)
# =====================================================================

# 🆕 NEW: BULK RECEIVER (MDM -> HES)
@app.route('/api/v1/commands/relay-state/bulk', methods=['POST'])
def receive_commands_bulk():
    """Receives an array of up to 5,000 commands from the MDM at once."""
    commands = request.json.get('commands', [])
    
    if not commands:
        return jsonify({"error": "No commands provided."}), 400
    if len(commands) > 5000:
        return jsonify({"error": "Batch size too large. Max 5000."}), 413

    insert_data = []
    for cmd in commands:
        hes_tx_id = f"HES_TX_{uuid.uuid4().hex[:8].upper()}"
        insert_data.append((hes_tx_id, cmd['reference_id'], cmd['meter_no'], cmd['command']))
    
    with get_db() as conn:
        conn.executemany("""
            INSERT INTO pending_commands (hes_tx_id, reference_id, meter_no, command) 
            VALUES (?, ?, ?, ?)
        """, insert_data)
        conn.commit()
        
    logging.info(f"📥 Received and Queued {len(insert_data)} Bulk Commands.")
    return jsonify({"status": "QUEUED", "count": len(insert_data)}), 202

@app.route('/internal/process-queue', methods=['GET'])
def process_queue():
    """Worker 1: Simulates RF network execution (Runs every 2 mins)"""
    with get_db() as conn:
        commands = conn.execute("SELECT * FROM pending_commands WHERE status = 'PENDING' LIMIT 1000").fetchall()
        
    if not commands:
        return jsonify({"message": "RC-DC Queue empty."}), 200

    processed = 0
    with get_db() as update_conn:
        for row in commands:
            # Simulate ultra-fast RF execution for load testing
            update_conn.execute("UPDATE pending_commands SET status = 'COMPLETED' WHERE hes_tx_id = ?", (row['hes_tx_id'],))
            
            # Auto-sync the local edge state
            new_status = 'D' if row['command'] == 'DISCONNECT' else 'C'
            update_conn.execute("UPDATE consumer_edge_state SET connection_status = ? WHERE meter_no = ?", (new_status, row['meter_no']))
            processed += 1
        update_conn.commit()
            
    return jsonify({"message": f"Processed {processed} RC-DC commands over RF."}), 200

# 🆕 UPDATED: BULK CALLBACK SENDER (HES -> MDM)
@app.route('/internal/push-bulk-callbacks', methods=['GET'])
def push_bulk_callbacks():
    """Worker 2: Pushes async responses back to MDM."""
    return execute_bulk_push()

# 🆕 NEW: MANUAL RETRY ENDPOINT
@app.route('/internal/retry-failed-callbacks', methods=['GET'])
def retry_failed_callbacks():
    """Manually triggers a re-push of any callbacks that the MDM previously rejected or missed."""
    logging.info("♻️ Initiating Manual Retry for Unsent Callbacks...")
    return execute_bulk_push()

def execute_bulk_push():
    """Core logic for sending webhooks securely."""
    with get_db() as conn:
        # Grab completed commands that have NOT been successfully acknowledged by the MDM
        unsent = conn.execute("""
            SELECT hes_tx_id, reference_id, meter_no, command 
            FROM pending_commands 
            WHERE status = 'COMPLETED' AND is_notified = 0 
            LIMIT 1000
        """).fetchall()

    if not unsent:
        return jsonify({"message": "No pending or failed callbacks to push."}), 200

    # Formatted specifically for your new MDM Bulk Receiver ('results' array)
    bulk_payload = {
        "results": [{
            "hes_transaction_id": r['hes_tx_id'], 
            "reference_id": r['reference_id'], 
            "meter_no": r['meter_no'], 
            "command": r['command'], 
            "status": "SUCCESS"
        } for r in unsent]
    }

    try:
        logging.info(f"🚀 Pushing payload of {len(unsent)} callbacks to MDM...")
        resp = requests.post(MDM_BULK_WEBHOOK_URL, json=bulk_payload, timeout=20)
        
        if resp.status_code == 200:
            # ONLY mark as notified if the MDM returned a 200 OK
            tx_ids = [r['hes_tx_id'] for r in unsent]
            placeholders = ','.join(['?'] * len(tx_ids))
            with get_db() as conn:
                conn.execute(f"UPDATE pending_commands SET is_notified = 1 WHERE hes_tx_id IN ({placeholders})", tx_ids)
                conn.commit()
            
            logging.info("✅ MDM successfully acknowledged the bulk payload.")
            return jsonify({"message": f"Successfully pushed {len(unsent)} callbacks to MDM."}), 200
        else:
            logging.error(f"🚨 MDM Rejected Payload (Status {resp.status_code}): {resp.text}")
            return jsonify({"error": f"MDM rejected payload (Status {resp.status_code})"}), 502
            
    except requests.exceptions.RequestException as e:
        logging.error(f"🚨 Network Error reaching MDM: {e}")
        return jsonify({"error": "Network failure reaching MDM. Will retry later."}), 503

# =====================================================================
# PART 2: THE GENERATION WORKER (Midnight Block Load)
# =====================================================================
@app.route('/internal/worker/generate-reads', methods=['GET'])
def generate_daily_reads():
    target_date = datetime.now().date()
    target_date_str = target_date.strftime("%Y-%m-%d")
    
    is_month_start = (target_date.day == 1)
    prev_month_str = (target_date - relativedelta(months=1)).strftime("%Y-%m")
    curr_month_str = target_date.strftime("%Y-%m")

    with get_db() as conn:
        consumers = conn.execute("SELECT * FROM consumer_edge_state").fetchall()
        
        for c in consumers:
            m_no = c['meter_no']
            
            if c['connection_status'] == 'D':
                imp_inc, exp_inc = 0.0, 0.0
            else:
                imp_inc = round(random.uniform(5.0, 15.0), 2)
                exp_inc = round(random.uniform(2.0, 8.0), 2) if c['consumer_type'] == 'NET_METER' else 0.0

            new_imp = round(c['curr_import_total'] + imp_inc, 2)
            new_exp = round(c['curr_export_total'] + exp_inc, 2)
            tz1 = round(c['curr_import_tz1'] + (imp_inc * 0.4), 2)
            tz2 = round(c['curr_import_tz2'] + (imp_inc * 0.4), 2)
            tz3 = round(c['curr_import_tz3'] + (imp_inc * 0.2), 2)

            conn.execute("""
                UPDATE consumer_edge_state 
                SET prev_import_total = curr_import_total, 
                    prev_export_total = curr_export_total,
                    curr_import_total=?, curr_import_tz1=?, curr_import_tz2=?, curr_import_tz3=?, 
                    curr_export_total=?, last_read_date=?
                WHERE meter_no=?
            """, (new_imp, tz1, tz2, tz3, new_exp, target_date_str, m_no))

            if is_month_start:
                conn.execute("""
                    UPDATE hes_monthly_settlement 
                    SET end_import_total=?, end_export_total=? 
                    WHERE meter_no=? AND billing_month=?
                """, (new_imp, new_exp, m_no, prev_month_str))
                
                conn.execute("""
                    INSERT INTO hes_monthly_settlement (meter_no, billing_month, start_import_total, start_export_total)
                    VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING
                """, (m_no, curr_month_str, new_imp, new_exp))

        if is_month_start:
            two_months_ago = (target_date - relativedelta(months=2)).strftime("%Y-%m")
            conn.execute("DELETE FROM hes_monthly_settlement WHERE billing_month <= ?", (two_months_ago,))

        conn.commit()
    return jsonify({"message": f"Successfully generated reads for {len(consumers)} meters."}), 200

# =====================================================================
# PART 3: MASTER DATA SYNC & EXPORT APIs
# =====================================================================
@app.route('/api/v1/sync/meters-bulk', methods=['POST'])
def sync_meters_bulk():
    meters = request.json.get('meters', [])
    if not meters: return jsonify({"error": "No meters provided."}), 400
    if len(meters) > 5000: return jsonify({"error": "Batch size too large."}), 413

    provision_data, state_data = [], []
    for m in meters:
        m_no = m['meter_no']
        baseline_imp = float(m.get('baseline_kwh', 0.0))
        baseline_exp = float(m.get('baseline_export_kwh', 0.0))
        
        provision_data.append((m_no, m.get('amisp_name', 'UNKNOWN')))
        state_data.append((
            m_no, m.get('consumer_type', 'NORMAL'), m.get('connection_status', 'C'), 
            baseline_imp, baseline_imp, baseline_exp, baseline_exp
        ))

    with get_db() as conn:
        conn.executemany("INSERT INTO hes_provisioned_meters (meter_no, amisp_name) VALUES (?, ?) ON CONFLICT(meter_no) DO UPDATE SET amisp_name=excluded.amisp_name", provision_data)
        conn.executemany("""
            INSERT INTO consumer_edge_state (meter_no, consumer_type, connection_status, curr_import_total, prev_import_total, curr_export_total, prev_export_total) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(meter_no) DO UPDATE SET 
                consumer_type=excluded.consumer_type, connection_status=excluded.connection_status,
                curr_import_total=excluded.curr_import_total, prev_import_total=excluded.prev_import_total,
                curr_export_total=excluded.curr_export_total, prev_export_total=excluded.prev_export_total
        """, state_data)
        conn.commit()
    return jsonify({"message": f"Successfully synced {len(meters)} meters."}), 200

@app.route('/api/v1/export/daily-reads', methods=['GET'])
def export_daily():
    limit = int(request.args.get('limit', 1000))
    offset = int(request.args.get('offset', 0))
    if limit > 1000: return jsonify({"error": "PAYLOAD_TOO_LARGE", "message": "Max batch size is 1000."}), 413
    
    with get_db() as conn:
        data = [dict(r) for r in conn.execute("SELECT * FROM consumer_edge_state LIMIT ? OFFSET ?", (limit, offset)).fetchall()]
    return jsonify({"count": len(data), "limit": limit, "offset": offset, "data": data}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
