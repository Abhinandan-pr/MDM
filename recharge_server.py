import os
import time
import random
import uuid
import logging
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# 1. Load Environment Variables securely
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

POSTGRES_API_URL = os.getenv("POSTGRES_API_URL")
MYSQL_MASTER_URL = os.getenv("MYSQL_MASTER_URL")

if not POSTGRES_API_URL or not MYSQL_MASTER_URL:
    logging.critical("Database URLs are missing! Check your .env file.")
    exit(1)

# Create engines
pg_engine = create_engine(POSTGRES_API_URL, pool_size=5, max_overflow=10)
mysql_engine = create_engine(MYSQL_MASTER_URL, pool_size=5, max_overflow=10)

PAYMENT_MODES = ['BBPS', 'PhonePe', 'GooglePay', 'Paytm', 'AmazonPay', 'CreditCard']

def get_dynamic_traffic_delay():
    """Calculates the delay between recharges based on the time of day to simulate real traffic."""
    current_hour = datetime.now().hour
    
    # Morning & Evening Peak Hours (High Traffic)
    if (8 <= current_hour <= 11) or (18 <= current_hour <= 21):
        delay = random.uniform(1.0, 4.0)
        traffic_state = "PEAK"
        
    # Night / Graveyard Shift (Very Low Traffic)
    elif current_hour >= 22 or current_hour <= 5:
        delay = random.uniform(45.0, 90.0)
        traffic_state = "OFF-PEAK"
        
    # Standard Daytime Hours (Moderate Traffic)
    else:
        delay = random.uniform(10.0, 20.0)
        traffic_state = "NORMAL"
        
    return delay, traffic_state

def run_end_to_end_recharge():
    """Generates a recharge, logs it to Postgres, and updates the MySQL wallet."""
    
    with mysql_engine.connect() as conn:
        result = conn.execute(text("SELECT consumer_no, CURRENT_BALANCE_INR FROM consumer_master ORDER BY RAND() LIMIT 1"))
        consumer = result.fetchone()
        
    if not consumer:
        logging.error("No consumers found in MySQL 'consumer_master'.")
        return

    consumer_no = consumer[0]
    old_balance = consumer[1] or 0.00
    
    amount = random.randint(2, 50) * 100.00
    tx_id = f"TXN{uuid.uuid4().hex[:12].upper()}"
    mode = random.choice(PAYMENT_MODES)
    gw_ref = f"GW_{uuid.uuid4().hex[:8].upper()}"

    # 1. Log Transaction to PostgreSQL (Retained for 1 Month)
    with pg_engine.begin() as pg_conn:
        pg_conn.execute(
            text("""
                INSERT INTO recharge_transactions (transaction_id, consumer_no, amount, payment_mode, gateway_ref)
                VALUES (:tx_id, :c_no, :amt, :mode, :ref)
            """),
            {"tx_id": tx_id, "c_no": consumer_no, "amt": amount, "mode": mode, "ref": gw_ref}
        )

    # 2. Update CURRENT_BALANCE_INR in MySQL Master
    with mysql_engine.begin() as mysql_trans:
        mysql_trans.execute(
            text("""
                UPDATE consumer_master 
                SET CURRENT_BALANCE_INR = CURRENT_BALANCE_INR + :amt                    
                WHERE consumer_no = :c_no;
            """),
            {"amt": amount, "c_no": consumer_no}
        )
    
    logging.info(f"Recharge Success: {consumer_no} | ₹{amount} added. New Balance: ₹{old_balance + amount}")

if __name__ == "__main__":
    logging.info("Starting Enterprise Traffic Simulator...")
    while True:
        try:
            run_end_to_end_recharge()
        except Exception as e:
            logging.error(f"Pipeline error: {str(e)}")
            
        # Apply the dynamic time delay
        delay_seconds, current_state = get_dynamic_traffic_delay()
        logging.info(f"[{current_state} TRAFFIC] Waiting {round(delay_seconds, 1)} seconds for next transaction...\n" + "-"*40)
        time.sleep(delay_seconds)
