import pandas as pd
import os
import sys
import logging

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, os.path.join(root_dir, 'src'))

from cleaning.cleaners import clean_transactions, clean_outlets

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_bronze_to_silver():
    bronze_path = os.path.join(root_dir, 'data', 'bronze')
    silver_path = os.path.join(root_dir, 'data', 'silver')
    rejected_path = os.path.join(root_dir, 'data', 'rejected')

    # Ensure directories exist
    os.makedirs(silver_path, exist_ok=True)
    os.makedirs(rejected_path, exist_ok=True)

    logger.info("Starting Bronze to Silver pipeline...")

    # 1. Load Data
    logger.info("Loading Bronze datasets...")
    df_trans = pd.read_csv(os.path.join(bronze_path, 'transactions_history_final.csv'))
    df_master = pd.read_csv(os.path.join(bronze_path, 'outlet_master.csv'))
    df_coords = pd.read_csv(os.path.join(bronze_path, 'outlet_coordinates.csv'))
    df_holiday = pd.read_csv(os.path.join(bronze_path, 'holiday_list.csv'))
    df_season = pd.read_csv(os.path.join(bronze_path, 'distributor_seasonality_details.csv'))

    # 2. Clean Outlets
    logger.info("Cleaning Outlet data...")
    clean_outlet_df, rejected_outlet_df = clean_outlets(df_master, df_coords)
    
    clean_outlet_df.to_parquet(os.path.join(silver_path, 'dim_outlets.parquet'), index=False)
    if not rejected_outlet_df.empty:
        rejected_outlet_df.to_csv(os.path.join(rejected_path, 'rejected_outlets.csv'), index=False)
    logger.info(f"Outlets: {len(clean_outlet_df)} cleaned, {len(rejected_outlet_df)} rejected.")

    # 3. Clean Transactions
    logger.info("Cleaning Transaction data...")
    # Use cleaned outlets for referential integrity check
    clean_trans_df, rejected_trans_df = clean_transactions(df_trans, clean_outlet_df)
    
    clean_trans_df.to_parquet(os.path.join(silver_path, 'fact_transactions.parquet'), index=False)
    if not rejected_trans_df.empty:
        rejected_trans_df.to_csv(os.path.join(rejected_path, 'rejected_transactions.csv'), index=False)
    logger.info(f"Transactions: {len(clean_trans_df)} cleaned, {len(rejected_trans_df)} rejected.")

    # 4. Pass-through (or minor cleaning) for other files
    logger.info("Processing remaining datasets...")
    df_holiday.to_parquet(os.path.join(silver_path, 'dim_holidays.parquet'), index=False)
    df_season.to_parquet(os.path.join(silver_path, 'dim_distributor_seasonality.parquet'), index=False)

    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    run_bronze_to_silver()
