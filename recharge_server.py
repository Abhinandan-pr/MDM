import os
import random
import uuid
import logging
import requests
from sqlalchemy import create_engine, text

# =====================================================================
# SETUP & CONFIGURATION
# =====================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [RECHARGE SIMULATOR] - %(message)s')

POSTGRES_API_URL = os.environ.get("POSTGRES_API_URL")
MYSQL_MASTER_URL = os.environ.get("MYSQL_MASTER_URL")

MDM_REALTIME_RC_URL = os.environ.get("MDM_REALTIME_RC_URL", "https://mdm-ou39.onrender.com/api/v1/trigger-realtime-rc")

TARGET_AMISP = "AMISP-1"
TOTAL_RECHARGES = 100  # "at least 100 recharges"
DC_RATIO = 0.60        # "approx 60% disconnected"

if not POSTGRES_API_URL or not MYSQL_MASTER_URL:
    logging.critical("🚨 Database URLs are missing! Ensure environment variables are set.")
    exit(1)

pg_engine = create_engine(POSTGRES_API_URL, pool_size=5, max_overflow=10)
mysql_engine = create_engine(MYSQL_MASTER_URL, pool_size=5, max_overflow=10)

session = requests.Session()
session.trust_env = False

PAYMENT_MODES = ['BBPS', 'PhonePe', 'GooglePay', 'Paytm', 'AmazonPay', 'CreditCard']

# =====================================================================
# CORE LOGIC
# =====================================================================
def fetch_target_batch():
    """Fetches exactly 100 targets (60 DC'd, 40 Active) in just 2 queries."""
    dc_count = int(TOTAL_RECHARGES * DC_RATIO)
    active_count = TOTAL_RECHARGES - dc_count
    
    targets = []
    
    with mysql_engine.connect() as conn:
        # 1. Fetch 60 Disconnected / Negative Balance Consumers
        logging.info(f"🔍 Fetching {dc_count} Disconnected targets for {TARGET_AMISP}...")
        dc_sql = text(f"""
            SELECT consumer_no, CURRENT_BALANCE_INR, meter_no, connection_status 
            FROM consumer_master 
            WHERE CURRENT_BALANCE_INR < 0 
            AND connection_status = 'D' 
            AND AMISP_PARTNER = '{TARGET_AMISP}'
            ORDER BY RAND() LIMIT {dc_count}
        """)
        targets.extend(conn.execute(dc_sql).fetchall())
        
        # 2. Fetch 40 Active / Positive Balance Consumers (Standard Top-ups)
        logging.info(f"🔍 Fetching {active_count} Active targets for {TARGET_AMISP}...")
        active_sql = text(f"""
            SELECT consumer_no, CURRENT_BALANCE_INR, meter_no, connection_status 
            FROM consumer_master 
            WHERE CURRENT_BALANCE_INR >= 0 
            AND connection_status = 'C' 
            AND AMISP_PARTNER = '{TARGET_AMISP}'
            ORDER BY RAND() LIMIT {active_count}
        """)
        targets.extend(conn.execute(active_sql).fetchall())
        
    # Shuffle them so the logs look like organic, random incoming traffic
    random.shuffle(targets)
    return targets

def process_bulk_recharges():
    logging.info("=" * 60)
    logging.info(f"🚀 STARTING BULK RECHARGE SIMULATION ({TOTAL_RECHARGES} Targets)")
    logging.info("=" * 60)

    targets = fetch_target_batch()
    
    if not targets:
        logging.warning("⚠️ No valid targets found in the database. Aborting.")
        return

    success_count = 0
    meters_to_reconnect = []

    # Bulk Insert Arrays
    pg_transactions = []
    mysql_updates = []

    # 1. CALCULATE FINANCIALS
    for consumer_no, old_balance, meter_no, current_status in targets:
        old_balance = old_balance or 0.00
        
        # Guarantee negative balances are cleared so RC triggers
        if old_balance < 0:
            amount = abs(old_balance) + (random.randint(1, 10) * 100.00)
        else:
            amount = random.randint(2, 50) * 100.00
            
        new_balance = old_balance + amount

        tx_id = f"TXN{uuid.uuid4().hex[:12].upper()}"
        mode = random.choice(PAYMENT_MODES)
        gw_ref = f"GW_{uuid.uuid4().hex[:8].upper()}"
        
        # Add to PostgreSQL ledger array
        pg_transactions.append({
            "tx_id": tx_id, "c_no": consumer_no, 
            "amt": amount, "mode": mode, "ref": gw_ref
        })
        
        # Add to MySQL wallet update array
        mysql_updates.append({
            "new_bal": new_balance, "c_no": consumer_no
        })
        
        # Flag for Reconnection
        if current_status == 'D' and new_balance >= 0:
            meters_to_reconnect.append(meter_no)

    # 2. COMMIT TO DATABASES (Lightning Fast Bulk Mode)
    logging.info(f"🔒 Writing {len(pg_transactions)} receipts to PostgreSQL...")
    with pg_engine.begin() as pg_conn:
        pg_conn.execute(
            text("""
                INSERT INTO recharge_transactions (transaction_id, consumer_no, amount, payment_mode, gateway_ref)
                VALUES (:tx_id, :c_no, :amt, :mode, :ref)
            """),
            pg_transactions
        )

    logging.info(f"🔒 Updating {len(mysql_updates)} wallets in MySQL Master...")
    with mysql_engine.begin() as mysql_trans:
        # Note: In a real prod environment with 10k recharges, we'd use a temp table JOIN here. 
        # For 100 rows, executemany is perfectly fast.
        mysql_trans.execute(
            text("""
                UPDATE consumer_master 
                SET CURRENT_BALANCE_INR = :new_bal
                WHERE consumer_no = :c_no;
            """),
            mysql_updates
        )

    logging.info(f"💰 Wallet Updates Complete. ({len(targets)}/{TOTAL_RECHARGES} successful)")

    # =================================================================
    # 3. FIRE THE REAL-TIME RECONNECT API
    # =================================================================
    if meters_to_reconnect:
        logging.info(f"⚡ Firing Real-Time RC trigger for {len(meters_to_reconnect)} rescued meters...")
        try:
            payload = {"meters": meters_to_reconnect}
            response = session.post(MDM_REALTIME_RC_URL, json=payload, timeout=20)
            
            if response.status_code == 200:
                logging.info(f"🎉 MDM successfully initiated real-time reconnection!")
            else:
                logging.error(f"❌ MDM rejected RC trigger: {response.text}")
        except Exception as e:
            logging.error(f"🚨 Network error firing RC trigger: {e}")
    else:
        logging.info("⏸️ No disconnected meters crossed the positive threshold this batch.")

    logging.info("=" * 60)

if __name__ == "__main__":
    process_bulk_recharges()
