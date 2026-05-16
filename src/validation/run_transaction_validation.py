"""
Quick Pre-Silver Diagnostic
============================
Runs DQ checks directly on the raw Bronze CSV files (before cleaning).
Useful for an initial sanity check when the Silver layer hasn't been built yet.

Usage
-----
    uv run python src/validation/run_transaction_validation.py
"""
import pandas as pd
import os
import sys
import logging

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir   = os.path.abspath(os.path.join(script_dir, '..', '..'))
src_path   = os.path.join(root_dir, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

from validation.quality_checks import (
    check_nulls,
    check_duplicates,
    check_ranges,
    check_referential_integrity,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_bronze_transactions() -> None:
    """Run DQ checks on raw bronze transaction and outlet files."""
    bronze_path = os.path.join(root_dir, 'data', 'bronze')
    trans_file  = os.path.join(bronze_path, 'transactions_history_final.csv')
    outlet_file = os.path.join(bronze_path, 'outlet_master.csv')

    logger.info(f"Loading: {trans_file}")
    transactions  = pd.read_csv(trans_file)
    outlet_master = pd.read_csv(outlet_file)

    logger.info("\n--- Bronze Transaction Data Quality Report ---")

    # Null Checks
    cols_to_check = ['Outlet_ID', 'SKU_ID', 'Volume_Liters', 'Total_Bill_Value']
    nulls = check_nulls(transactions, cols_to_check)
    logger.info(f"Null Counts: {nulls}")

    # Range Checks
    logger.info("Checking Volume_Liters (>= 0)...")
    invalid_vol = check_ranges(transactions, 'Volume_Liters', 0, 1e9)
    logger.info(f"Invalid Volume rows: {len(invalid_vol)}")

    # Duplicate Check
    dup_keys = ['Outlet_ID', 'Year', 'Month', 'SKU_ID', 'Volume_Liters', 'Total_Bill_Value']
    dup_count = check_duplicates(transactions, dup_keys)
    logger.info(f"Duplicate transaction rows: {dup_count}")

    # Referential Integrity
    logger.info("Checking Referential Integrity (Outlet_ID)...")
    missing = check_referential_integrity(transactions, outlet_master, 'Outlet_ID', 'Outlet_ID')
    logger.info(f"Unique missing Outlet_IDs: {len(missing)}")
    if len(missing) > 0:
        logger.warning(f"Sample missing IDs: {missing[:5]}")


if __name__ == '__main__':
    validate_bronze_transactions()
