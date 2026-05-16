import pandas as pd
import os
import sys
import logging

# Add project root to path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
src_path = os.path.join(root_dir, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

from validation.quality_checks import (
    check_nulls, 
    check_duplicates, 
    check_ranges, 
    check_referential_integrity
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def validate_silver_layer():
    silver_path = os.path.join(root_dir, 'data', 'silver')
    
    logger.info("--- Validating Silver Layer Data ---")
    
    # 1. Load Data
    df_trans = pd.read_parquet(os.path.join(silver_path, 'fact_transactions.parquet'))
    df_outlets = pd.read_parquet(os.path.join(silver_path, 'dim_outlets.parquet'))
    
    # 2. Null Checks
    logger.info("Checking for Nulls...")
    trans_nulls = check_nulls(df_trans, df_trans.columns.tolist())
    outlet_nulls = check_nulls(df_outlets, df_outlets.columns.tolist())
    
    if sum(trans_nulls.values()) == 0:
        logger.info("OK: No nulls in transactions.")
    else:
        logger.error(f"FAIL: Nulls found in transactions: {trans_nulls}")
        
    if sum(outlet_nulls.values()) == 0:
        logger.info("OK: No nulls in outlets.")
    else:
        logger.error(f"FAIL: Nulls found in outlets: {outlet_nulls}")
        
    # 3. Duplicate Checks
    logger.info("Checking for Duplicates...")
    trans_keys = ['Outlet_ID', 'Year', 'Month', 'Distributor_ID', 'SKU_ID', 'Volume_Liters', 'Total_Bill_Value']
    trans_dups = check_duplicates(df_trans, trans_keys)
    outlet_dups = check_duplicates(df_outlets, ['Outlet_ID'])
    
    if trans_dups == 0:
        logger.info("OK: No duplicate transactions.")
    else:
        logger.error(f"FAIL: Found {trans_dups} duplicate transactions.")
        
    if outlet_dups == 0:
        logger.info("OK: No duplicate outlets.")
    else:
        logger.error(f"FAIL: Found {outlet_dups} duplicate outlets.")
        
    # 4. Referential Integrity
    logger.info("Checking Referential Integrity...")
    missing_outlets = check_referential_integrity(df_trans, df_outlets, 'Outlet_ID', 'Outlet_ID')
    if len(missing_outlets) == 0:
        logger.info("OK: All transaction outlets exist in outlet master.")
    else:
        logger.error(f"FAIL: Found {len(missing_outlets)} transaction outlets missing from master.")
        
    # 5. Range Checks (GPS)
    logger.info("Checking GPS ranges...")
    LAT_RANGE = (5.9, 9.9)
    LON_RANGE = (79.6, 81.9)
    invalid_lat = check_ranges(df_outlets, 'Latitude', *LAT_RANGE)
    invalid_lon = check_ranges(df_outlets, 'Longitude', *LON_RANGE)
    
    if len(invalid_lat) == 0 and len(invalid_lon) == 0:
        logger.info("OK: All outlet coordinates are within Sri Lanka bounds.")
    else:
        logger.error(f"FAIL: Found invalid coordinates. Lat errors: {len(invalid_lat)}, Lon errors: {len(invalid_lon)}")

if __name__ == "__main__":
    validate_silver_layer()
