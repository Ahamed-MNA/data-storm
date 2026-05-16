"""
Verification Script
===================
Compares the final SFA predictions against raw/silver historical transactions
to verify that predictions act as a mathematically sound ceiling (bordering above
historical demand).

Usage
-----
    uv run python src/validation/verify_predictions.py
"""
import os
import sys
import pandas as pd
import numpy as np

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

def run_verification():
    outputs_dir = os.path.join(root_dir, 'outputs')
    silver_dir = os.path.join(root_dir, 'data', 'silver')
    
    # 1. Locate files
    pred_path = os.path.join(outputs_dir, 'data_storm_predictions.csv')
    trans_path = os.path.join(silver_dir, 'fact_transactions.parquet')
    
    if not os.path.exists(pred_path):
        print(f"Error: Predictions file not found at {pred_path}")
        return
    if not os.path.exists(trans_path):
        print(f"Error: Silver transactions file not found at {trans_path}")
        return
        
    print("Loading datasets...")
    df_pred = pd.read_csv(pred_path)
    df_trans = pd.read_parquet(trans_path)
    
    print(f"Loaded {len(df_pred):,} predictions.")
    print(f"Loaded {len(df_trans):,} transaction records.")
    
    # 2. Aggregate historical transaction volumes at the outlet-month level
    # In SFA features, we group by ['Outlet_ID', 'Year', 'Month']
    monthly_trans = df_trans.groupby(['Outlet_ID', 'Year', 'Month'])['Volume_Liters'].sum().reset_index()
    
    # Now get the Max and Mean historical monthly volume for each outlet
    outlet_hist = monthly_trans.groupby('Outlet_ID')['Volume_Liters'].agg(
        Max_Historical_Monthly_Volume='max',
        Mean_Historical_Monthly_Volume='mean',
        Min_Historical_Monthly_Volume='min'
    ).reset_index()
    
    # 3. Merge predictions with historical summary
    df_merged = pd.merge(df_pred, outlet_hist, on='Outlet_ID', how='inner')
    
    missing_predictions = len(df_pred) - len(df_merged)
    if missing_predictions > 0:
        print(f"Warning: {missing_predictions} outlets in predictions do not have historical transactions.")
        
    # 4. Perform ceiling verification checks
    df_merged['Potential_vs_Max_Diff'] = df_merged['Maximum_Monthly_Liters'] - df_merged['Max_Historical_Monthly_Volume']
    df_merged['Ratio_to_Max'] = df_merged['Maximum_Monthly_Liters'] / df_merged['Max_Historical_Monthly_Volume']
    df_merged['Ratio_to_Mean'] = df_merged['Maximum_Monthly_Liters'] / df_merged['Mean_Historical_Monthly_Volume']
    
    is_above_max = df_merged['Maximum_Monthly_Liters'] >= df_merged['Max_Historical_Monthly_Volume']
    is_above_mean = df_merged['Maximum_Monthly_Liters'] >= df_merged['Mean_Historical_Monthly_Volume']
    
    pct_above_max = is_above_max.mean() * 100
    pct_above_mean = is_above_mean.mean() * 100
    
    print("\n==============================================")
    print("      CEILING VALIDATION METRICS              ")
    print("==============================================")
    print(f"Outlets evaluated: {len(df_merged):,}")
    print(f"Percentage of outlets where predicted potential is:")
    print(f"  - Above or equal to Historical MEAN: {pct_above_mean:.2f}%")
    print(f"  - Above or equal to Historical MAX : {pct_above_max:.2f}%")
    
    print("\nDistribution of Ratio (Potential / Max Historical):")
    print(df_merged['Ratio_to_Max'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    
    print("\nDistribution of Ratio (Potential / Mean Historical):")
    print(df_merged['Ratio_to_Mean'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    
    # Check for outlets where the potential is extremely high or low
    under_max = df_merged[~is_above_max]
    print(f"\nNumber of outlets where potential < historical max: {len(under_max):,} ({100 - pct_above_max:.2f}%)")
    if len(under_max) > 0:
        print("Summary of Ratio for these under-performing outlets:")
        print(under_max['Ratio_to_Max'].describe())
        
    print("\nSanity checks:")
    print(f"  - Nulls in predictions: {df_pred['Maximum_Monthly_Liters'].isnull().sum()}")
    print(f"  - Predictions <= 0: {(df_pred['Maximum_Monthly_Liters'] <= 0).sum()}")
    print(f"  - Min Prediction value: {df_pred['Maximum_Monthly_Liters'].min():.2f}")
    print(f"  - Max Prediction value: {df_pred['Maximum_Monthly_Liters'].max():.2f}")

if __name__ == '__main__':
    run_verification()
