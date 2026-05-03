import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sqlalchemy import create_engine, text

# ---------------------------------------------------------
# DATABASE CONFIGURATION
# ---------------------------------------------------------
DB_USER = os.getenv('DB_USER', 'your_username')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'your_password')
DB_HOST = os.getenv('DB_HOST', 'your_host_url.aivencloud.com')
DB_PORT = os.getenv('DB_PORT', 'your_port')
DB_NAME = os.getenv('DB_NAME', 'defaultdb')

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ---------------------------------------------------------
# PHASE 1: NUMPY-ACCELERATED AMISP SIMULATOR
# ---------------------------------------------------------
def simulate_amisp_reads(target_date):
    print(f"\n=== PHASE 1: INGESTING AMISP READS FOR {target_date} ===")
    
    fetch_sql = """
        SELECT METER_NO, MAX(TARIFF_CODE) as TARIFF_CODE 
        FROM consumer_master 
        GROUP BY METER_NO
    """
    meter_chunks = pd.read_sql(fetch_sql, engine, chunksize=50000)
    
    total_processed = 0
    total_failed = 0

    try:
        with engine.begin() as conn:
            conn.execute(text(f"DELETE FROM daily_meter_reads WHERE READING_DATE = '{target_date}'"))

        for chunk in meter_chunks:
            # Drop 5% for network failure
            successful_reads = chunk.sample(frac=0.95) 
            num_reads = len(successful_reads)
            
            total_failed += (len(chunk) - num_reads)
            total_processed += num_reads
            
            # NUMPY VECTORIZATION: Replaces slow iterrows() loop
            is_ht = successful_reads['TARIFF_CODE'].isin(['HTS-I', 'HTS-II']).values
            
            # Generate base load arrays instantly
            base_kwh = np.where(is_ht, 
                                np.random.uniform(500.0, 2000.0, num_reads), 
                                np.random.uniform(5.0, 25.0, num_reads))
            
            # Calculate TOD arrays instantly
            kwh_off_peak = np.round(base_kwh * np.random.uniform(0.15, 0.25, num_reads), 2)
            kwh_normal = np.round(base_kwh * np.random.uniform(0.45, 0.55, num_reads), 2)
            kwh_peak = np.round(base_kwh * np.random.uniform(0.25, 0.35, num_reads), 2)
            
            # Construct DataFrame
            df_reads = pd.DataFrame({
                "METER_NO": successful_reads['METER_NO'].values,
                "READING_DATE": target_date,
                "KWH_OFF_PEAK": kwh_off_peak,
                "KWH_NORMAL": kwh_normal,
                "KWH_PEAK": kwh_peak
            }).drop_duplicates(subset=['METER_NO'])
            
            # Push to DB
            df_reads.to_sql('daily_meter_reads', con=engine, if_exists='append', index=False, chunksize=10000, method='multi')

        print(f"✓ AMISP Ingestion Complete. Success: {total_processed} | Offline/Failed: {total_failed}")
        return True 
        
    except Exception as e:
        print(f"❌ AMISP Ingestion Failed: {e}")
        return False

# ---------------------------------------------------------
# PHASE 2: IN-MEMORY NUMPY BILLING ENGINE
# ---------------------------------------------------------
def run_billing_engine(target_date):
    print(f"\n=== PHASE 2: EXECUTING NUMPY BILLING ENGINE FOR {target_date} ===")
    
    # 1. Extract raw data into Python RAM
    extract_sql = text("""
        SELECT 
            cm.CONSUMER_NO, cm.METER_NO, cm.CURRENT_BALANCE_INR, cm.LAST_BILLED_DATE,
            tc.DAILY_FIXED_CHARGE, tc.DAILY_METER_RENT, tc.BASE_RATE_PER_UNIT, 
            tc.TOD_OFF_PEAK_MULT, tc.TOD_NORMAL_MULT, tc.TOD_PEAK_MULT, tc.SUBSIDY_PER_UNIT_SLAB1,
            dmr.KWH_OFF_PEAK, dmr.KWH_NORMAL, dmr.KWH_PEAK
        FROM consumer_master cm
        JOIN tariff_config tc ON cm.TARIFF_CODE = tc.TARIFF_CODE
        LEFT JOIN daily_meter_reads dmr ON cm.METER_NO = dmr.METER_NO AND dmr.READING_DATE = :target_date
        WHERE cm.TARIFF_CATEGORY = 'Prepaid' AND cm.LAST_BILLED_DATE < :target_date
    """)
    
    print("  -> Fetching master data and reads into Memory...")
    df = pd.read_sql(extract_sql, engine, params={"target_date": target_date})
    
    if df.empty:
        print("✓ No consumers to bill today.")
        return

    print("  -> Crunching TOD and Financials using NumPy...")
    
    # 2. Prepare Data (Handle missing reads safely)
    df['KWH_OFF_PEAK'] = df['KWH_OFF_PEAK'].fillna(0.0).astype(np.float32)
    df['KWH_NORMAL'] = df['KWH_NORMAL'].fillna(0.0).astype(np.float32)
    df['KWH_PEAK'] = df['KWH_PEAK'].fillna(0.0).astype(np.float32)
    
    # 3. Vectorized Math (Lightning Fast)
    target_date_dt = pd.to_datetime(target_date)
    last_billed_dt = pd.to_datetime(df['LAST_BILLED_DATE'])
    
    # Days capped at 60
    df['UNBILLED_DAYS'] = np.clip((target_date_dt - last_billed_dt).dt.days, a_min=0, a_max=60).astype(np.int32)
    
    # Fixed Deductions
    df['FC_DED'] = df['UNBILLED_DAYS'] * df['DAILY_FIXED_CHARGE']
    df['MR_DED'] = df['UNBILLED_DAYS'] * df['DAILY_METER_RENT']
    
    # TOD Deductions
    base_rate = df['BASE_RATE_PER_UNIT']
    df['EC_OFF_PEAK'] = df['KWH_OFF_PEAK'] * base_rate * df['TOD_OFF_PEAK_MULT']
    df['EC_NORMAL'] = df['KWH_NORMAL'] * base_rate * df['TOD_NORMAL_MULT']
    df['EC_PEAK'] = df['KWH_PEAK'] * base_rate * df['TOD_PEAK_MULT']
    
    df['TOTAL_EC'] = df['EC_OFF_PEAK'] + df['EC_NORMAL'] + df['EC_PEAK']
    
    # Subsidy
    total_kwh = df['KWH_OFF_PEAK'] + df['KWH_NORMAL'] + df['KWH_PEAK']
    df['SUBSIDY_CREDIT'] = total_kwh * df['SUBSIDY_PER_UNIT_SLAB1']
    
    # Final Math
    df['TOTAL_DED'] = df['FC_DED'] + df['MR_DED'] + df['TOTAL_EC'] - df['SUBSIDY_CREDIT']
    df['CLOSING_BAL'] = df['CURRENT_BALANCE_INR'] - df['TOTAL_DED']
    
    # Ledger IDs and Static Columns
    df['LEDGER_ID'] = df['METER_NO'].astype(str) + '-' + target_date.strftime('%Y-%m-%d')
    df['BILLING_DATE'] = target_date
    df['V_CREDIT'] = 0.0
    df['V_DEBIT'] = 0.0

    print("  -> Preparing Ledger and Master Updates...")
    
    # Prepare Ledger DataFrame
    ledger_cols = [
        'LEDGER_ID', 'CONSUMER_NO', 'METER_NO', 'BILLING_DATE', 'UNBILLED_DAYS',
        'FC_DED', 'MR_DED', 'KWH_OFF_PEAK', 'EC_OFF_PEAK', 'KWH_NORMAL', 'EC_NORMAL',
        'KWH_PEAK', 'EC_PEAK', 'TOTAL_EC', 'SUBSIDY_CREDIT', 'V_CREDIT', 'V_DEBIT',
        'TOTAL_DED', 'CLOSING_BAL'
    ]
    
    # Rename DataFrame columns to match Ledger SQL columns exactly
    rename_map = {
        'FC_DED': 'FIXED_CHARGE_DEDUCTED', 'MR_DED': 'METER_RENT_DEDUCTED',
        'KWH_OFF_PEAK': 'KWH_OFF_PEAK_CONSUMED', 'EC_OFF_PEAK': 'EC_OFF_PEAK_DEDUCTED',
        'KWH_NORMAL': 'KWH_NORMAL_CONSUMED', 'EC_NORMAL': 'EC_NORMAL_DEDUCTED',
        'KWH_PEAK': 'KWH_PEAK_CONSUMED', 'EC_PEAK': 'EC_PEAK_DEDUCTED',
        'TOTAL_EC': 'TOTAL_ENERGY_CHARGE_DEDUCTED', 'SUBSIDY_CREDIT': 'STATE_SUBSIDY_CREDITED',
        'V_CREDIT': 'VIRTUAL_CREDIT_AMT', 'V_DEBIT': 'VIRTUAL_DEBIT_AMT',
        'TOTAL_DED': 'TOTAL_DAILY_DEDUCTION', 'CLOSING_BAL': 'CLOSING_BALANCE'
    }
    df_ledger = df[ledger_cols].rename(columns=rename_map)
    
    # 4. Push updates to Database
    try:
        print("  -> Writing detailed receipts to the Daily Ledger...")
        df_ledger.to_sql('daily_billing_ledger', con=engine, if_exists='append', index=False, chunksize=10000, method='multi')

        print("  -> Updating Master Wallets via Staging Table...")
        # Fastest way to update 500k rows: Load to a temp table, then SQL UPDATE JOIN
        df_master_updates = df[['METER_NO', 'CLOSING_BALANCE', 'BILLING_DATE']].copy()
        
        with engine.begin() as conn:
            # Create staging table
            conn.execute(text("DROP TABLE IF EXISTS temp_wallet_updates;"))
            
            # Push Python data to temp table
            df_master_updates.to_sql('temp_wallet_updates', con=conn, if_exists='replace', index=False, chunksize=10000, method='multi')
            
            # Bulk update the actual master table in one swift motion
            conn.execute(text("""
                UPDATE consumer_master cm
                JOIN temp_wallet_updates tmp ON cm.METER_NO = tmp.METER_NO
                SET cm.CURRENT_BALANCE_INR = tmp.CLOSING_BALANCE,
                    cm.LAST_BILLED_DATE = tmp.BILLING_DATE
            """))
            
            # Generate DC List
            print("  -> Generating Daily Disconnection (DC) List...")
            conn.execute(text("DROP TABLE IF EXISTS daily_dc_list;"))
            conn.execute(text("""
                CREATE TABLE daily_dc_list AS
                SELECT CONSUMER_NO, METER_NO, NAME, PHONE, AMISP_PARTNER, CURRENT_BALANCE_INR, DISCOM, CIRCLE, SECTION
                FROM consumer_master 
                WHERE TARIFF_CATEGORY = 'Prepaid' AND CURRENT_BALANCE_INR < 0;
            """))
            
            # Cleanup
            conn.execute(text("DROP TABLE temp_wallet_updates;"))

        print("✓ Financial Processing & DC List Generation Complete.")
        
    except Exception as e:
        print(f"❌ Billing Engine Failed: {e}")

# ---------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------
if __name__ == "__main__":
    processing_date = date.today() - timedelta(days=1)
    print("=====================================================")
    print(f"  STARTING NUMPY-ACCELERATED BATCH FOR: {processing_date}")
    print("=====================================================")
    
    ingestion_success = simulate_amisp_reads(processing_date)
    
    if ingestion_success:
        run_billing_engine(processing_date)
        print("\n🏆 ALL BATCH OPERATIONS COMPLETED SUCCESSFULLY 🏆")
    else:
        print("\n⚠️ PIPELINE HALTED: Billing Engine aborted due to Ingestion Failure.")