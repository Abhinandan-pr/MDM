import os
import random
import uuid
import logging
import requests
from sqlalchemy import create_engine, text

# =====================================================================
# SETUP & CONFIGURATION
# =====================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

POSTGRES_API_URL = os.environ.get("POSTGRES_API_URL")
MYSQL_MASTER_URL = os.environ.get("MYSQL_MASTER_URL")

# Your new Real-Time RC API on the MDM
MDM_REALTIME_RC_URL = os.environ.get("MDM_REALTIME_RC_URL", "https://mdm-ou39.onrender.com/api/v1/trigger-realtime-rc")

if not POSTGRES_API_URL or not MYSQL_MASTER_URL:
    logging.critical("🚨 Database URLs are missing! Ensure GitHub Secrets are mapped correctly.")
    exit(1)

pg_engine = create_engine(POSTGRES_API_URL, pool_size=5, max_overflow=10)
mysql_engine = create_engine(MYSQL_MASTER_URL, pool_size=5, max_overflow=10)

session = requests.Session()
session.trust_env = False

PAYMENT_MODES = ['BBPS', 'PhonePe', 'GooglePay', 'Paytm', 'AmazonPay', 'CreditCard']

# =====================================================================
# CORE LOGIC
# =====================================================================
def get_weighted_consumer():
    """
    Mimics real-world behavior: 75% chance to target a defaulter/disconnected user,
    25% chance to target a healthy user topping up their wallet.
    """
    is_urgent = random.random() < 0.75 
    
    with mysql_engine.connect() as conn:
        if is_urgent:
            # Look for people in the dark or in debt
            sql = """
                SELECT consumer_no, CURRENT_BALANCE_INR, meter_no, connection_status 
                FROM consumer_master 
                WHERE CURRENT_BALANCE_INR < 0 OR connection_status = 'D' 
                ORDER BY RAND() LIMIT 1
            """
            result = conn.execute(text(sql)).fetchone()
            if result:
                return result
                
        # Fallback (or if the 25% healthy route was chosen)
        sql = """
            SELECT consumer_no, CURRENT_BALANCE_INR, meter_no, connection_status 
            FROM consumer_master 
            WHERE CURRENT_BALANCE_INR >= 0 AND connection_status = 'C' 
            ORDER BY RAND() LIMIT 1
        """
        return conn.execute(text(sql)).fetchone()

def process_batch_recharges():
    batch_size = random.randint(5, 25)
    logging.info(f"🚀 GitHub Action Triggered: Processing {batch_size} synthetic recharges...")
    
    success_count = 0
    meters_to_reconnect = []

    for _ in range(batch_size):
        consumer = get_weighted_consumer()
        if not consumer:
            continue

        consumer_no, old_balance, meter_no, current_status = consumer
        old_balance = old_balance or 0.00
        
        # If they are deeply negative, they need to pay a larger amount to get power back.
        # This mimics them clearing their dues + adding a buffer.
        if old_balance < 0:
            amount = abs(old_balance) + (random.randint(1, 10) * 100.00)
        else:
            amount = random.randint(2, 50) * 100.00
            
        new_balance = old_balance + amount

        tx_id = f"TXN{uuid.uuid4().hex[:12].upper()}"
        mode = random.choice(PAYMENT_MODES)
        gw_ref = f"GW_{uuid.uuid4().hex[:8].upper()}"

        try:
            # 1. Log Transaction to PostgreSQL
            with pg_engine.begin() as pg_conn:
                pg_conn.execute(
                    text("""
                        INSERT INTO recharge_transactions (transaction_id, consumer_no, amount, payment_mode, gateway_ref)
                        VALUES (:tx_id, :c_no, :amt, :mode, :ref)
                    """),
                    {"tx_id": tx_id, "c_no": consumer_no, "amt": amount, "mode": mode, "ref": gw_ref}
                )

            # 2. Update Wallet in MySQL
            with mysql_engine.begin() as mysql_trans:
                mysql_trans.execute(
                    text("""
                        UPDATE consumer_master 
                        SET CURRENT_BALANCE_INR = :new_bal
                        WHERE consumer_no = :c_no;
                    """),
                    {"new_bal": new_balance, "c_no": consumer_no}
                )
            
            logging.info(f"✅ Success: {meter_no} | ₹{amount} added via {mode}. New Bal: ₹{new_balance:.2f}")
            success_count += 1
            
            # 3. IDENTIFY RECONNECT CANDIDATES
            # If they were Disconnected AND their new balance is >= 0, they get power back!
            if current_status == 'D' and new_balance >= 0:
                meters_to_reconnect.append(meter_no)
                
        except Exception as e:
            logging.error(f"🚨 Failed to process recharge for {meter_no}: {e}")

    logging.info(f"💰 Batch Complete. Successfully processed {success_count}/{batch_size} recharges.")
    
    # =================================================================
    # 4. FIRE THE REAL-TIME RECONNECT API
    # =================================================================
    if meters_to_reconnect:
        logging.info(f"⚡ Firing Real-Time RC trigger for {len(meters_to_reconnect)} meters...")
        try:
            payload = {"meters": meters_to_reconnect}
            response = session.post(MDM_REALTIME_RC_URL, json=payload, timeout=15)
            
            if response.status_code == 200:
                logging.info(f"🎉 MDM successfully initiated real-time reconnection!")
            else:
                logging.error(f"❌ MDM rejected RC trigger: {response.text}")
        except Exception as e:
            logging.error(f"🚨 Network error firing RC trigger: {e}")
            # Even if the trigger fails, the continuous Bulk RC engine will catch them later as a backup!
    else:
        logging.info("⏸️ No disconnected meters crossed the positive threshold this batch. No RC triggered.")

if __name__ == "__main__":
    process_batch_recharges()