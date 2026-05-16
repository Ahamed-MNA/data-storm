import pandas as pd
import numpy as np
import os
import sys
import logging

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, os.path.join(root_dir, 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def build_refined_gold_layer():
    silver_path = os.path.join(root_dir, 'data', 'silver')
    gold_path = os.path.join(root_dir, 'data', 'gold')
    os.makedirs(gold_path, exist_ok=True)

    logger.info("Loading Silver datasets...")
    df_trans = pd.read_parquet(os.path.join(silver_path, 'fact_transactions.parquet'))
    df_outlets = pd.read_parquet(os.path.join(silver_path, 'dim_outlets.parquet'))
    df_season = pd.read_parquet(os.path.join(silver_path, 'dim_distributor_seasonality.parquet'))
    df_holiday = pd.read_parquet(os.path.join(silver_path, 'dim_holidays.parquet'))
    df_poi = pd.read_csv(os.path.join(silver_path, 'outlets_with_poi_counts.csv'))

    # --- 1. Aggregation & Basic Join ---
    logger.info("Aggregating transactions...")
    df_agg = df_trans.groupby(['Outlet_ID', 'Year', 'Month', 'Distributor_ID']).agg({
        'Volume_Liters': 'sum',
        'Total_Bill_Value': 'sum'
    }).reset_index()

    # Merge Dimensions
    df_gold = df_agg.merge(df_outlets, on='Outlet_ID', how='inner')
    df_gold = df_gold.merge(df_poi[['Outlet_ID', 'POI_Count_500m', 'POI_Count_1km']], on='Outlet_ID', how='left')
    df_gold = df_gold.merge(df_season, on=['Distributor_ID', 'Year', 'Month'], how='left')

    # --- 2. Holiday Adjustment (Demand Driver) ---
    logger.info("Calculating holiday features...")
    df_holiday['Date'] = pd.to_datetime(df_holiday['Date'])
    df_holiday['Year'] = df_holiday['Date'].dt.year
    df_holiday['Month'] = df_holiday['Date'].dt.month
    holiday_counts = df_holiday.groupby(['Year', 'Month']).size().reset_index(name='Holiday_Count')
    df_gold = df_gold.merge(holiday_counts, on=['Year', 'Month'], how='left')
    df_gold['Holiday_Count'] = df_gold['Holiday_Count'].fillna(0)

    # --- 3. Constraint Proxies (Calculated per Outlet) ---
    logger.info("Computing constraint proxies...")
    
    # a. Flatline Score & CV of Volume
    outlet_stats = df_agg.groupby('Outlet_ID')['Volume_Liters'].agg(['mean', 'std', 'count']).reset_index()
    outlet_stats['CV_Volume'] = outlet_stats['std'] / outlet_stats['mean']
    
    # Flatline: % of identical consecutive or absolute volume matches
    # Simplified: count of most frequent volume / total months
    def get_flatline(x):
        if len(x) < 2: return 0.0
        return x.value_counts().max() / len(x)
    
    flatline_scores = df_agg.groupby('Outlet_ID')['Volume_Liters'].apply(get_flatline).reset_index(name='Flatline_Score')
    
    # b. Round-Number Bias
    # % of volumes divisible by 100 or 50
    def get_round_bias(x):
        is_round = (x % 50 == 0) | (x % 100 == 0)
        return is_round.mean()
    
    round_bias = df_agg.groupby('Outlet_ID')['Volume_Liters'].apply(get_round_bias).reset_index(name='Round_Number_Bias')
    
    # c. Bill Value Rigidity
    # Variance of (Bill / Volume)
    df_agg['Price_Per_Liter'] = df_agg['Total_Bill_Value'] / df_agg['Volume_Liters'].replace(0, np.nan)
    price_var = df_agg.groupby('Outlet_ID')['Price_Per_Liter'].var().reset_index(name='Price_Rigidity')
    
    # Combine proxies
    proxies = outlet_stats[['Outlet_ID', 'CV_Volume']].merge(flatline_scores, on='Outlet_ID')
    proxies = proxies.merge(round_bias, on='Outlet_ID')
    proxies = proxies.merge(price_var, on='Outlet_ID')

    # --- 4. Final Data Preparation ---
    logger.info("Finalizing features...")
    
    # Province extraction
    province_map = {
        'W': 'Western', 'C': 'Central', 'S': 'Southern', 'NW': 'North Western',
        'E': 'Eastern', 'NC': 'North Central', 'U': 'Uva', 'SG': 'Sabaragamuwa', 'N': 'Northern'
    }
    df_gold['Province'] = df_gold['Distributor_ID'].apply(lambda x: province_map.get(x.split('_')[1], 'Other') if len(x.split('_')) > 1 else 'Unknown')

    # Filter for positive volumes for SFA target
    df_gold = df_gold[df_gold['Volume_Liters'] > 0].copy()
    df_gold['ln_volume'] = np.log(df_gold['Volume_Liters'])

    # Merge Proxies into Gold (for downstream validation/reporting)
    df_gold = df_gold.merge(proxies, on='Outlet_ID', how='left')

    # One-hot encoding for Demand Drivers
    categorical_cols = ['Outlet_Type', 'Outlet_Size', 'Province', 'Seasonality_Index']
    df_gold = pd.get_dummies(df_gold, columns=categorical_cols, drop_first=True)
    
    # Convert bool to int
    bool_cols = df_gold.select_dtypes(include='bool').columns
    df_gold[bool_cols] = df_gold[bool_cols].astype(int)

    # Save
    output_file = os.path.join(gold_path, 'sfa_refined.parquet')
    df_gold.to_parquet(output_file, index=False)
    
    logger.info(f"Refined Gold layer built! Shape: {df_gold.shape}")
    logger.info(f"Saved to: {output_file}")

if __name__ == "__main__":
    build_refined_gold_layer()
