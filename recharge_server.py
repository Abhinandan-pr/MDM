import os
import random
import uuid
import logging
from sqlalchemy import create_engine, text

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Pull credentials directly from GitHub Secrets environment
POSTGRES_API_URL = os.environ.get("POSTGRES_API_URL")
MYSQL_MASTER_URL = os.environ.get("MYSQL_MASTER_URL")

if not POSTGRES_API_URL or not MYSQL_MASTER_URL:
    logging.critical("Database URLs are missing! Ensure GitHub Secrets are mapped correctly.")
    exit(1)

# Create engines
pg_engine = create_engine(POSTGRES_API_URL, pool_size=5, max_overflow=10)
mysql_engine = create_engine(MYSQL_MASTER_URL, pool_size=5, max_overflow=10)

PAYMENT_MODES = ['BBPS', 'PhonePe', 'GooglePay', 'Paytm', 'AmazonPay', 'CreditCard']

def process_single_recharge():
    """Generates a single recharge and updates the databases."""
    with mysql_engine.connect() as conn:
        result = conn.execute(text("SELECT consumer_no, CURRENT_BALANCE_INR FROM consumer_master ORDER BY RAND() LIMIT 1"))
        consumer = result.fetchone()
        
    if not consumer:
        logging.error("No consumers found in MySQL 'consumer_master'.")
        return False

    consumer_no = consumer[0]
    old_balance = consumer[1] or 0.00
    amount = random.randint(2, 50) * 100.00
    tx_id = f"TXN{uuid.uuid4().hex[:12].upper()}"
    mode = random.choice(PAYMENT_MODES)
    gw_ref = f"GW_{uuid.uuid4().hex[:8].upper()}"

    # Log Transaction to PostgreSQL
    with pg_engine.begin() as pg_conn:
        pg_conn.execute(
            text("""
                INSERT INTO recharge_transactions (transaction_id, consumer_no, amount, payment_mode, gateway_ref)
                VALUES (:tx_id, :c_no, :amt, :mode, :ref)
            """),
            {"tx_id": tx_id, "c_no": consumer_no, "amt": amount, "mode": mode, "ref": gw_ref}
        )

    # Update Wallet in MySQL
    with mysql_engine.begin() as mysql_trans:
        mysql_trans.execute(
            text("""
                UPDATE consumer_master 
                SET CURRENT_BALANCE_INR = CURRENT_BALANCE_INR + :amt,
                    last_recharge_date = NOW()
                WHERE consumer_no = :c_no;
            """),
            {"amt": amount, "c_no": consumer_no}
        )
    
    logging.info(f"Success: {consumer_no} | ₹{amount} added. New Balance: ₹{old_balance + amount}")
    return True

if __name__ == "__main__":
    # Generate a random number of recharges for this 15-minute batch (e.g., between 5 and 25)
    batch_size = random.randint(5, 25)
    logging.info(f"GitHub Action Triggered: Processing {batch_size} synthetic recharges...")
    
    success_count = 0
    for _ in range(batch_size):
        if process_single_recharge():
            success_count += 1
            
    logging.info(f"Batch Complete. Successfully processed {success_count}/{batch_size} recharges.")
