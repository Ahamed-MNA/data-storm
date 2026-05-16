import pandas as pd
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

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

def validate_transactions():
    data_path = os.path.join(root_dir, 'data', 'bronze')
    trans_file = os.path.join(data_path, 'transactions_history_final.csv')
    outlet_file = os.path.join(data_path, 'outlet_master.csv')

    print(f"Loading {trans_file}...")
    transactions = pd.read_csv(trans_file)
    outlet_master = pd.read_csv(outlet_file)

    print("\n--- Transaction Data Quality Report ---")
    
    # Null Checks
    cols_to_check = ['Outlet_ID', 'SKU_ID', 'Volume_Liters', 'Total_Bill_Value']
    nulls = check_nulls(transactions, cols_to_check)
    print("Null Counts:", nulls)

    # Range Checks
    print("Checking Volume_Liters (>=0)...")
    invalid_vol = check_ranges(transactions, 'Volume_Liters', 0, 1e9)
    print(f"Invalid Volume rows: {len(invalid_vol)}")

    # Referential Integrity
    print("Checking Referential Integrity (Outlet_ID)...")
    missing = check_referential_integrity(transactions, outlet_master, 'Outlet_ID', 'Outlet_ID')
    print(f"Unique missing Outlet_IDs: {len(missing)}")
    if len(missing) > 0:
        print("Sample missing IDs:", missing[:5])

if __name__ == "__main__":
    validate_transactions()
