import os
import sys
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def build_refined_gold_layer():
    """
    Orchestrates the Silver to Gold feature engineering pipeline.
    Loads all Silver datasets, performs feature engineering for competitors and POIs,
    computes constraint proxies, applies one-hot encoding, and saves the final 
    model-ready dataset to the Gold layer directory.
    """
    # Define directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    silver_path = os.path.join(root_dir, 'data', 'silver')
    gold_path = os.path.join(root_dir, 'data', 'gold')
    os.makedirs(gold_path, exist_ok=True)

    logger.info("Loading Silver datasets...")
    
    # Check that required files exist
    required_files = [
        'fact_transactions.parquet', 
        'dim_outlets.parquet', 
        'dim_distributor_seasonality.parquet', 
        'dim_holidays.parquet',
        'silver_layer_poi_edges.parquet',
        'silver_layer_competitor_graph.parquet'
    ]
    
    for filename in required_files:
        filepath = os.path.join(silver_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required silver dataset not found: {filepath}")

    # Load datasets
    df_trans = pd.read_parquet(os.path.join(silver_path, 'fact_transactions.parquet'))
    df_outlets = pd.read_parquet(os.path.join(silver_path, 'dim_outlets.parquet'))
    df_season = pd.read_parquet(os.path.join(silver_path, 'dim_distributor_seasonality.parquet'))
    df_holiday = pd.read_parquet(os.path.join(silver_path, 'dim_holidays.parquet'))
    df_poi = pd.read_parquet(os.path.join(silver_path, 'silver_layer_poi_edges.parquet'))
    df_comp = pd.read_parquet(os.path.join(silver_path, 'silver_layer_competitor_graph.parquet'))

    # --- 1. Aggregate Transactions ---
    logger.info("Aggregating monthly transactions per outlet...")
    df_agg = df_trans.groupby(['Outlet_ID', 'Year', 'Month', 'Distributor_ID']).agg({
        'Volume_Liters': 'sum',
        'Total_Bill_Value': 'sum'
    }).reset_index()

    # --- 2. Feature Engineering: Competitor Catchment Density ---
    logger.info("Engineering competitor graph features...")
    # Reconstruct bidirectional edges since the graph is undirected and stored alphabetically
    edges_fwd = df_comp[['Source_Outlet_ID', 'Target_Competitor_ID', 'Distance_Meters', 'Competitive_Friction']].rename(
        columns={'Source_Outlet_ID': 'Outlet_ID', 'Target_Competitor_ID': 'Competitor_ID'}
    )
    edges_bwd = df_comp[['Target_Competitor_ID', 'Source_Outlet_ID', 'Distance_Meters', 'Competitive_Friction']].rename(
        columns={'Target_Competitor_ID': 'Outlet_ID', 'Source_Outlet_ID': 'Competitor_ID'}
    )
    edges_all = pd.concat([edges_fwd, edges_bwd], ignore_index=True)

    # Group by outlet to calculate structural features
    comp_features = edges_all.groupby('Outlet_ID').agg(
        Competitor_Count_5km=('Competitor_ID', 'count'),
        Min_Competitor_Distance_Meters=('Distance_Meters', 'min'),
        Total_Competitive_Friction=('Competitive_Friction', 'sum'),
        Average_Competitive_Friction=('Competitive_Friction', 'mean')
    ).reset_index()

    # --- 3. Feature Engineering: Spatial Distance-Decay POI Impact ---
    logger.info("Engineering POI distance-decay features...")
    # Map POI Types to 7 broader logical categories
    category_map = {
        'bus_stop': 'Transport', 'bus_station': 'Transport', 'station': 'Transport', 'halt': 'Transport',
        'school': 'Education', 'university': 'Education', 'college': 'Education',
        'hospital': 'Health', 'clinic': 'Health',
        'industrial': 'Commercial', 'marketplace': 'Commercial', 'restaurant': 'Commercial', 'cafe': 'Commercial', 'kitchen': 'Commercial',
        'apartments': 'Residential',
        'place_of_worship': 'Social', 'community_centre': 'Social', 'theatre': 'Social', 'shelter': 'Social',
        'hotel': 'Tourism', 'guest_house': 'Tourism', 'attraction': 'Tourism', 'museum': 'Tourism'
    }
    
    df_poi['POI_Category'] = df_poi['POI_Type'].map(category_map).fillna('Other')

    # Sum footfall impact scores per category for each outlet
    poi_cat_impact = df_poi.groupby(['Outlet_ID', 'POI_Category'])['Footfall_Impact_Score'].sum().unstack(fill_value=0.0).reset_index()
    # Apply clean prefix names
    poi_cat_cols = {col: f'POI_Impact_{col}' for col in poi_cat_impact.columns if col != 'Outlet_ID'}
    poi_cat_impact.rename(columns=poi_cat_cols, inplace=True)

    # Aggregate overall POI features (omitting old simple POI counts)
    poi_agg = df_poi.groupby('Outlet_ID').agg(
        POI_Total_Impact_Score=('Footfall_Impact_Score', 'sum'),
        POI_Avg_Distance_Meters=('Distance_Meters', 'mean')
    ).reset_index()

    # Combine POI features
    poi_features = pd.merge(poi_agg, poi_cat_impact, on='Outlet_ID', how='outer')

    # --- 4. Merge Dimensions and Features ---
    logger.info("Merging outlets, competitors, and POI features...")
    df_gold = df_agg.merge(df_outlets, on='Outlet_ID', how='inner')
    
    # Merge Competitor Features
    df_gold = df_gold.merge(comp_features, on='Outlet_ID', how='left')
    # Fill outlets with no competitors in 5km
    df_gold['Competitor_Count_5km'] = df_gold['Competitor_Count_5km'].fillna(0.0)
    df_gold['Min_Competitor_Distance_Meters'] = df_gold['Min_Competitor_Distance_Meters'].fillna(5000.0)
    df_gold['Total_Competitive_Friction'] = df_gold['Total_Competitive_Friction'].fillna(0.0)
    df_gold['Average_Competitive_Friction'] = df_gold['Average_Competitive_Friction'].fillna(0.0)

    # Merge POI Features
    df_gold = df_gold.merge(poi_features, on='Outlet_ID', how='left')
    # Fill outlets with no POIs in 1km
    df_gold['POI_Total_Impact_Score'] = df_gold['POI_Total_Impact_Score'].fillna(0.0)
    df_gold['POI_Avg_Distance_Meters'] = df_gold['POI_Avg_Distance_Meters'].fillna(1000.0)
    
    impact_cols = [c for c in df_gold.columns if c.startswith('POI_Impact_')]
    for col in impact_cols:
        df_gold[col] = df_gold[col].fillna(0.0)

    # Merge Seasonality
    df_gold = df_gold.merge(df_season, on=['Distributor_ID', 'Year', 'Month'], how='left')

    # --- 5. Holiday Adjustment ---
    logger.info("Integrating holiday demand driver...")
    df_holiday['Date'] = pd.to_datetime(df_holiday['Date'])
    df_holiday['Year'] = df_holiday['Date'].dt.year
    df_holiday['Month'] = df_holiday['Date'].dt.month
    holiday_counts = df_holiday.groupby(['Year', 'Month']).size().reset_index(name='Holiday_Count')
    df_gold = df_gold.merge(holiday_counts, on=['Year', 'Month'], how='left')
    df_gold['Holiday_Count'] = df_gold['Holiday_Count'].fillna(0.0)

    # --- 6. Constraint Proxies (Calculated per Outlet) ---
    logger.info("Computing constraint proxies...")
    # a. Coefficient of Variation (CV) of Volume
    outlet_stats = df_agg.groupby('Outlet_ID')['Volume_Liters'].agg(['mean', 'std', 'count']).reset_index()
    outlet_stats['CV_Volume'] = outlet_stats['std'] / outlet_stats['mean']
    # If standard deviation is NaN (1 month of data), set CV to 0
    outlet_stats['CV_Volume'] = outlet_stats['CV_Volume'].fillna(0.0)

    # b. Flatline Score: Proportion of the most frequent volume sold
    def get_flatline(x):
        if len(x) < 2: 
            return 0.0
        return x.value_counts().max() / len(x)
    
    flatline_scores = df_agg.groupby('Outlet_ID')['Volume_Liters'].apply(get_flatline).reset_index(name='Flatline_Score')

    # c. Round-Number Bias: Ratio of volumes sold divisible by 50 or 100
    def get_round_bias(x):
        is_round = (x % 50 == 0) | (x % 100 == 0)
        return is_round.mean()

    round_bias = df_agg.groupby('Outlet_ID')['Volume_Liters'].apply(get_round_bias).reset_index(name='Round_Number_Bias')

    # d. Price Rigidity: Variance of Price Per Liter
    df_agg['Price_Per_Liter'] = df_agg['Total_Bill_Value'] / df_agg['Volume_Liters'].replace(0, np.nan)
    price_var = df_agg.groupby('Outlet_ID')['Price_Per_Liter'].var().reset_index(name='Price_Rigidity')
    price_var['Price_Rigidity'] = price_var['Price_Rigidity'].fillna(0.0)

    # Combine all proxies
    proxies = outlet_stats[['Outlet_ID', 'CV_Volume']].merge(flatline_scores, on='Outlet_ID')
    proxies = proxies.merge(round_bias, on='Outlet_ID')
    proxies = proxies.merge(price_var, on='Outlet_ID')
    
    # Merge proxies into main table
    df_gold = df_gold.merge(proxies, on='Outlet_ID', how='left')

    # --- 7. Final Formatting & Categorical Encoding ---
    logger.info("Formatting columns and applying one-hot encoding...")
    
    # Province extraction from Distributor ID
    province_map = {
        'W': 'Western', 'C': 'Central', 'S': 'Southern', 'NW': 'North Western',
        'E': 'Eastern', 'NC': 'North Central', 'U': 'Uva', 'SG': 'Sabaragamuwa', 'N': 'Northern'
    }
    df_gold['Province'] = df_gold['Distributor_ID'].apply(
        lambda x: province_map.get(x.split('_')[1], 'Other') if len(x.split('_')) > 1 else 'Unknown'
    )

    # Target variable generation (logarithm of volume for positive demand)
    df_gold = df_gold[df_gold['Volume_Liters'] > 0].copy()
    df_gold['ln_volume'] = np.log(df_gold['Volume_Liters'])

    # One-hot encode categorical features (these are demand drivers)
    categorical_cols = ['Outlet_Type', 'Outlet_Size', 'Province', 'Seasonality_Index']
    df_gold = pd.get_dummies(df_gold, columns=categorical_cols, drop_first=True)
    
    # Convert booleans from get_dummies to integers (0 or 1)
    bool_cols = df_gold.select_dtypes(include='bool').columns
    df_gold[bool_cols] = df_gold[bool_cols].astype(int)

    # Save to Gold layer
    output_file = os.path.join(gold_path, 'sfa_refined.parquet')
    df_gold.to_parquet(output_file, index=False)
    
    logger.info(f"Refined Gold layer successfully built! Shape: {df_gold.shape}")
    logger.info(f"Saved gold file to: {output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    build_refined_gold_layer()
