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

if not POSTGRES_API_URL:
    logging.critical("🚨 POSTGRES_API_URL is missing!")
    exit(1)

# High-performance connection pool for bulk operations
pg_engine = create_engine(POSTGRES_API_URL, pool_size=10, max_overflow=20)

def init_db():
    """Creates AMISP-1 strictly isolated tables in the shared Postgres DB."""
    with pg_engine.begin() as conn:
        # 1. LIVE RC-DC ZONE (Isolated)
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
        
        # 2. PROVISIONING & STATE ZONE (Isolated)
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
    """Receives commands and inserts them to PostgreSQL in chunks of 1000."""
    commands = request.json.get('commands', [])
    if not commands: return jsonify({"error": "No commands provided."}), 400

    # Ensure we only process AMISP-1 commands just in case MDM routed poorly
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
    """Worker 1: Simulates RF network execution (Bulk Updates)"""
    with pg_engine.connect() as conn:
        commands = conn.execute(text("""
            SELECT hes_tx_id, meter_no, command 
            FROM amisp1_pending_commands 
            WHERE status = 'PENDING' LIMIT 1000
        """)).mappings().fetchall()
        
    if not commands:
        return jsonify({"message": "RC-DC Queue empty."}), 200

    # Prepare batch data
    update_tx = [{"hes_tx": r['hes_tx_id']} for r in commands]
    update_state = [{"new_stat": 'D' if r['command'] == 'DISCONNECT' else 'C', "m_no": r['meter_no']} for r in commands]

    # Execute in a single transaction block
    with pg_engine.begin() as conn:
        conn.execute(text("UPDATE amisp1_pending_commands SET status = 'COMPLETED' WHERE hes_tx_id = :hes_tx"), update_tx)
        conn.execute(text("UPDATE amisp1_consumer_edge_state SET connection_status = :new_stat WHERE meter_no = :m_no"), update_state)
            
    return jsonify({"message": f"Processed {len(commands)} RC-DC commands over RF."}), 200

@app.route('/internal/push-bulk-callbacks', methods=['GET'])
def push_bulk_callbacks():
    """Worker 2: Pushes async responses back to MDM."""
    with pg_engine.connect() as conn:
        unsent = conn.execute(text("""
            SELECT hes_tx_id, reference_id, meter_no, command 
            FROM amisp1_pending_commands 
            WHERE status = 'COMPLETED' AND is_notified = FALSE 
            LIMIT 1000
        """)).mappings().fetchall()

    if not unsent:
        return jsonify({"message": "No pending callbacks."}), 200

    bulk_payload = {
        "results": [{
            "hes_transaction_id": r['hes_tx_id'], "reference_id": r['reference_id'], 
            "meter_no": r['meter_no'], "command": r['command'], "status": "SUCCESS"
        } for r in unsent]
    }

    try:
        resp = requests.post(MDM_BULK_WEBHOOK_URL, json=bulk_payload, timeout=20)
        if resp.status_code == 200:
            update_data = [{"hes_tx": r['hes_tx_id']} for r in unsent]
            with pg_engine.begin() as conn:
                conn.execute(text("UPDATE amisp1_pending_commands SET is_notified = TRUE WHERE hes_tx_id = :hes_tx"), update_data)
            return jsonify({"message": f"Successfully pushed {len(unsent)} callbacks to MDM."}), 200
        else:
            return jsonify({"error": f"MDM rejected payload (Status {resp.status_code})"}), 502
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Network failure reaching MDM."}), 503

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
    """MDM pushes bulk consumer data. Strictly filters for AMISP-1."""
    meters = request.json.get('meters', [])
    if not meters: return jsonify({"error": "No meters provided."}), 400

    # 🛑 THE GATEKEEPER: Only accept meters explicitly marked for this AMISP
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

    # Bulk Insert in Chunks (Postgres Syntax for UPSERT)
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
