"""
Generate Final Predictions
==========================
Reads SFA model parameters, historical transactions, and outlet dimensions to
compute individual-specific January 2026 latent potential predictions.

The analytical framework implements a mathematically sound uncapping logic:
1. Constructs January 2026 feature vectors for all outlets.
2. Predicts January-specific SFA group frontier benchmarks.
3. Retrieves historical January peaks (with overall peaks as fallback).
4. Scales the baseline up using the SFA-derived Systemic Inefficiency Multiplier
   exp(sigma_u * sqrt(2/pi)) to uncap the latent demand.

Schema required by the judges:
  Outlet_ID | Maximum_Monthly_Liters
"""
import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir   = os.path.abspath(os.path.join(script_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, os.path.join(root_dir, 'src'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TEAM_NAME       = 'data_mavericks'
TARGET_YEAR     = 2026
TARGET_MONTH    = 1                     # January


def generate_predictions(
    gold_path:   str | None = None,
    silver_path: str | None = None,
    model_path:  str | None = None,
    output_path: str | None = None,
    team_name:   str        = TEAM_NAME,
) -> pd.DataFrame:
    """
    Reads SFA results and writes the final predictions CSV with January-specific
    latent potential estimations bordering bordering above historical maximums.
    """
    output_dir = os.path.join(root_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Resolve paths
    if gold_path is None:
        gold_path = os.path.join(root_dir, 'data', 'gold', 'sfa_refined.parquet')
    if silver_path is None:
        silver_path = os.path.join(root_dir, 'data', 'silver')
    if model_path is None:
        model_path = os.path.join(output_dir, 'sfa_model.pkl')
    if output_path is None:
        output_path = os.path.join(output_dir, f'{team_name}_predictions.csv')

    # ── 1. Load Data & SFA Model ──────────────────────────────────────────
    logger.info("Loading SFA model and datasets...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run SFA pipeline first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    df_gold = pd.read_parquet(gold_path)
    
    trans_path = os.path.join(silver_path, 'fact_transactions.parquet')
    season_path = os.path.join(silver_path, 'dim_distributor_seasonality.parquet')
    df_trans = pd.read_parquet(trans_path)
    df_season = pd.read_parquet(season_path)

    # ── 2. Create January 2026 Feature Vectors ────────────────────────────
    logger.info("Constructing January 2026 demand feature matrix...")
    # Map each Outlet_ID to its primary Distributor_ID
    outlet_dist = df_trans.groupby('Outlet_ID')['Distributor_ID'].first().reset_index()

    # Get distributor's January seasonality index
    jan_season = df_season[df_season['Month'] == TARGET_MONTH].groupby('Distributor_ID')['Seasonality_Index'].first().reset_index()
    outlet_jan_season = outlet_dist.merge(jan_season, on='Distributor_ID', how='left')

    # Exclude dynamic transaction-level columns to isolate static outlet features
    exclude_cols = ['Year', 'Month', 'Distributor_ID', 'Volume_Liters', 'Total_Bill_Value', 
                    'ln_volume', 'CV_Volume', 'Flatline_Score', 'Round_Number_Bias', 
                    'Price_Rigidity', 'Potential_Volume', 'Efficiency_Score']
    
    outlet_features = df_gold.drop(columns=[col for col in exclude_cols if col in df_gold.columns]).groupby('Outlet_ID').first().reset_index()

    # Merge January distributor seasonality
    outlet_features = outlet_features.merge(outlet_jan_season[['Outlet_ID', 'Seasonality_Index']], on='Outlet_ID', how='left')

    # Re-create one-hot encoded Seasonality_Index columns for Jan 2026
    outlet_features['Seasonality_Index_Moderate'] = (outlet_features['Seasonality_Index'] == 'Moderate').astype(int)
    outlet_features['Seasonality_Index_Un-Favorable'] = (outlet_features['Seasonality_Index'] == 'Un-Favorable').astype(int)

    # Re-create Holiday_Count for Jan (average January holidays count = 11.3)
    outlet_features['Holiday_Count'] = 11.3
    outlet_features['Intercept'] = 1.0

    # Extract X matrix matching model features
    X_jan = outlet_features[model.feature_names]

    # Predict Group SFA Frontier Potential for January 2026
    outlet_features['Group_Frontier_Jan_2026'] = model.predict_potential(X_jan)

    # ── 3. Retrieve Historical Maximums (Uncapping Baseline) ──────────────
    logger.info("Calculating historical performance benchmarks...")
    positive_trans = df_trans[df_trans['Volume_Liters'] > 0]
    monthly_trans = positive_trans.groupby(['Outlet_ID', 'Year', 'Month'])['Volume_Liters'].sum().reset_index()

    overall_max = monthly_trans.groupby('Outlet_ID')['Volume_Liters'].max().reset_index(name='Overall_Max')
    jan_max = monthly_trans[monthly_trans['Month'] == TARGET_MONTH].groupby('Outlet_ID')['Volume_Liters'].max().reset_index(name='Jan_Max')

    # Join benchmarks
    df_compare = outlet_features[['Outlet_ID', 'Group_Frontier_Jan_2026']].merge(overall_max, on='Outlet_ID', how='left')
    df_compare = df_compare.merge(jan_max, on='Outlet_ID', how='left')
    
    # Fallback to overall max if no transactions exist historically in January
    df_compare['Jan_Max'] = df_compare['Jan_Max'].fillna(df_compare['Overall_Max'])

    # ── 4. Apply Systemic Inefficiency Multiplier ────────────────────────
    # Expected value of half-normal inefficiency: E[u] = sigma_u * sqrt(2/pi)
    expected_u = model.sigma_u * np.sqrt(2 / np.pi)
    ineff_multiplier = np.exp(expected_u)
    
    logger.info(f"Applying SFA Systemic Inefficiency Multiplier: {ineff_multiplier:.4f} (+{(ineff_multiplier-1)*100:.1f}% uncapping boost)")

    # Latent January potential is the maximum of the group frontier and outlet's
    # January historical peak, scaled up by the Systemic Inefficiency Multiplier
    df_compare['Maximum_Monthly_Liters'] = (
        np.maximum(df_compare['Group_Frontier_Jan_2026'], df_compare['Jan_Max']) * ineff_multiplier
    ).round(2)

    # Sanity check: ensure positive volumes
    df_compare['Maximum_Monthly_Liters'] = df_compare['Maximum_Monthly_Liters'].clip(lower=0.01)

    # ── 5. Save Predictions & Print Metrics ──────────────────────────────
    submission = df_compare[['Outlet_ID', 'Maximum_Monthly_Liters']]
    submission.to_csv(output_path, index=False)
    
    logger.info(f"\nPredictions saved successfully to: {output_path}")
    logger.info(f"  Total outlets predicted : {len(submission):,}")
    logger.info(f"  Potential range (L)     : {submission['Maximum_Monthly_Liters'].min():.2f}  –  {submission['Maximum_Monthly_Liters'].max():.2f}")
    logger.info(f"  Median potential (L)    : {submission['Maximum_Monthly_Liters'].median():.2f}")
    
    # Verify ceiling boundary logic
    pct_above_jan = (df_compare['Maximum_Monthly_Liters'] >= df_compare['Jan_Max']).mean() * 100
    pct_above_overall = (df_compare['Maximum_Monthly_Liters'] >= df_compare['Overall_Max']).mean() * 100
    logger.info(f"  Ceiling verification:")
    logger.info(f"    - Sits bordering ABOVE January historical peaks : {pct_above_jan:.2f}% of outlets")
    logger.info(f"    - Sits bordering ABOVE overall historical peaks : {pct_above_overall:.2f}% of outlets")

    return submission


if __name__ == '__main__':
    generate_predictions()
