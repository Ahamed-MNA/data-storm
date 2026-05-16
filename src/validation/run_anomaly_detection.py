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

from validation.anomaly_detection import *

def run_anomaly_report():
    data_path = os.path.join(root_dir, 'data', 'bronze')
    trans_file = os.path.join(data_path, 'transactions_history_final.csv')
    outlet_file = os.path.join(data_path, 'outlet_master.csv')
    coord_file = os.path.join(data_path, 'outlet_coordinates.csv')

    print("Loading data for anomaly detection...")
    df_trans = pd.read_csv(trans_file)
    df_outlet = pd.read_csv(outlet_file)
    df_coords = pd.read_csv(coord_file)

    print("\n=== TRANSACTION ANOMALIES ===")
    neg_q = detect_negative_quantities(df_trans)
    print(f"- Negative Quantities: {len(neg_q)}")
    
    spikes = detect_impossible_spikes(df_trans, 'Total_Bill_Value')
    print(f"- Sales Spikes (>3 STD): {len(spikes)}")
    
    dups = detect_duplicate_invoices(df_trans, ['Outlet_ID', 'Year', 'Month', 'SKU_ID', 'Volume_Liters'])
    print(f"- Potential Duplicate Invoices: {len(dups)}")

    print("\n=== OUTLET ANOMALIES ===")
    dup_gps = detect_duplicate_gps(df_coords)
    print(f"- Duplicate GPS locations: {len(dup_gps)}")
    
    inv_coords = detect_invalid_coordinates(df_coords)
    print(f"- Invalid Coordinates: {len(inv_coords)}")
    
    ocean_outlets = detect_outlets_in_ocean(df_coords)
    print(f"- Outlets outside Sri Lanka bounds: {len(ocean_outlets)}")
    
    inactive = detect_inactive_outlets(df_outlet, df_trans)
    print(f"- Inactive Outlets (no transactions): {len(inactive)}")

    print("\n=== DISTRIBUTOR ANOMALIES ===")
    caps = detect_delivery_caps(df_trans)
    print(f"- Distributors with suspicious volume caps: {len(caps['Distributor_ID'].unique())}")
    
    round_nums = detect_suspicious_round_numbers(df_trans, 'Total_Bill_Value', 1000)
    print(f"- Transactions with perfectly round numbers (mod 1000): {len(round_nums)}")

if __name__ == "__main__":
    run_anomaly_report()
