"""
Generate Final Predictions
==========================
Reads SFA model parameters, historical transactions, and outlet dimensions to
compute individual-specific January 2026 latent potential predictions.

Analytical Framework
--------------------
Implements a three-step uncapping logic grounded in SFA economic theory:

1. Constructs January 2026 feature vectors per outlet using the outlet's
   primary distributor's historically-consistent January seasonality index
   (Moderate / Favorable) and the average historical January holiday count.

2. Predicts the SFA group-level frontier benchmark for January 2026 via
   the fitted model:  Y*_group = exp(X * beta) * exp(0.5 * sigma_v^2)

3. Derives the Systemic Inefficiency Multiplier from the fitted half-normal
   inefficiency parameter:  multiplier = exp(sigma_u * sqrt(2/pi))
   This multiplier represents the expected fraction of demand suppressed by
   systemic constraints (credit limits, stockouts, delivery caps) and is used
   to uncap the latent ceiling above the observed historical peak.

Final potential:
  Maximum_Monthly_Liters_i =
      max(Y*_group_jan2026_i,  Historical_Jan_Max_i)  *  multiplier

This guarantees the predicted potential sits strictly above the outlet's own
historical January peak for 100% of outlets, while the group frontier provides
a principled minimum floor for historically inactive or low-volume outlets.

Schema required by judges:
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
TEAM_NAME    = 'data_mavericks'
TARGET_MONTH = 1  # January 2026

# Average number of public holidays observed in January across 2023-2025
JAN_AVG_HOLIDAY_COUNT = 11.3


def generate_predictions(
    gold_path:   str | None = None,
    silver_path: str | None = None,
    model_path:  str | None = None,
    output_path: str | None = None,
    team_name:   str        = TEAM_NAME,
) -> pd.DataFrame:
    """
    Compute January 2026 latent potential for every outlet and write the
    competition submission CSV.

    Parameters
    ----------
    gold_path   : path to sfa_refined.parquet  (Gold layer)
    silver_path : directory containing Silver layer parquets
    model_path  : path to sfa_model.pkl
    output_path : destination CSV  (defaults to outputs/<team_name>_predictions.csv)
    team_name   : used to name the output file

    Returns
    -------
    pd.DataFrame with columns  Outlet_ID, Maximum_Monthly_Liters
    """
    output_dir = os.path.join(root_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Resolve default paths
    if gold_path is None:
        gold_path = os.path.join(root_dir, 'data', 'gold', 'sfa_refined.parquet')
    if silver_path is None:
        silver_path = os.path.join(root_dir, 'data', 'silver')
    if model_path is None:
        model_path = os.path.join(output_dir, 'sfa_model.pkl')
    if output_path is None:
        output_path = os.path.join(output_dir, f'{team_name}_predictions.csv')

    # ── 1. Load SFA model and datasets ───────────────────────────────────
    logger.info("Loading SFA model and datasets...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            "Run the full pipeline first:  python main.py"
        )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    df_gold   = pd.read_parquet(gold_path)
    df_trans  = pd.read_parquet(os.path.join(silver_path, 'fact_transactions.parquet'))
    df_season = pd.read_parquet(os.path.join(silver_path, 'dim_distributor_seasonality.parquet'))

    logger.info(f"  Model features : {len(model.feature_names)}")
    logger.info(f"  sigma_u        : {model.sigma_u:.6f}  (systemic inefficiency)")
    logger.info(f"  sigma_v        : {model.sigma_v:.6f}  (random noise)")

    # ── 2. Construct January 2026 feature matrix ──────────────────────────
    logger.info("Constructing January 2026 feature matrix per outlet...")

    # Each outlet belongs to exactly one distributor (verified: 100% of 19,960 outlets)
    outlet_dist = df_trans.groupby('Outlet_ID')['Distributor_ID'].first().reset_index()

    # January seasonality is consistent across all years per distributor:
    #   DIST_S_01, DIST_S_02  → Favorable  (reference category, both dummies = 0)
    #   All other 8 dists     → Moderate
    jan_season = (df_season[df_season['Month'] == TARGET_MONTH]
                  .groupby('Distributor_ID')['Seasonality_Index']
                  .first()
                  .reset_index())
    outlet_jan_season = outlet_dist.merge(jan_season, on='Distributor_ID', how='left')

    # Take one static feature row per outlet from the Gold layer
    # (all dynamic columns — Year, Month, volumes — are excluded)
    dynamic_cols = [
        'Year', 'Month', 'Distributor_ID',
        'Volume_Liters', 'Total_Bill_Value', 'ln_volume',
        'CV_Volume', 'Flatline_Score', 'Round_Number_Bias', 'Price_Rigidity',
        'Potential_Volume', 'Efficiency_Score',
    ]
    outlet_features = (
        df_gold
        .drop(columns=[c for c in dynamic_cols if c in df_gold.columns])
        .groupby('Outlet_ID')
        .first()
        .reset_index()
    )

    # Merge January distributor seasonality and reconstruct OHE columns
    outlet_features = outlet_features.merge(
        outlet_jan_season[['Outlet_ID', 'Seasonality_Index']],
        on='Outlet_ID', how='left'
    )
    outlet_features['Seasonality_Index_Moderate']     = (outlet_features['Seasonality_Index'] == 'Moderate').astype(int)
    outlet_features['Seasonality_Index_Un-Favorable'] = (outlet_features['Seasonality_Index'] == 'Un-Favorable').astype(int)

    # Set January-specific holiday count and intercept
    outlet_features['Holiday_Count'] = JAN_AVG_HOLIDAY_COUNT
    outlet_features['Intercept']     = 1.0

    # Align to the exact feature order the SFA model was trained on
    X_jan = outlet_features[model.feature_names]

    # Group-level frontier for January 2026:  exp(X*beta) * exp(0.5 * sigma_v^2)
    outlet_features['Group_Frontier_Jan_2026'] = model.predict_potential(X_jan)

    # ── 3. Historical January peak volumes (uncapping baseline) ───────────
    logger.info("Retrieving historical January performance benchmarks...")

    positive_trans = df_trans[df_trans['Volume_Liters'] > 0]
    monthly_vol    = positive_trans.groupby(['Outlet_ID', 'Year', 'Month'])['Volume_Liters'].sum().reset_index()

    overall_max = monthly_vol.groupby('Outlet_ID')['Volume_Liters'].max().reset_index(name='Overall_Max')
    jan_max     = (monthly_vol[monthly_vol['Month'] == TARGET_MONTH]
                   .groupby('Outlet_ID')['Volume_Liters']
                   .max()
                   .reset_index(name='Jan_Max'))

    df_result = (
        outlet_features[['Outlet_ID', 'Group_Frontier_Jan_2026']]
        .merge(overall_max, on='Outlet_ID', how='left')
        .merge(jan_max,     on='Outlet_ID', how='left')
    )
    # Fallback: outlets with no historical January transactions use overall max
    df_result['Jan_Max'] = df_result['Jan_Max'].fillna(df_result['Overall_Max'])

    # ── 4. Apply Systemic Inefficiency Multiplier ─────────────────────────
    # E[u]  =  sigma_u * sqrt(2/pi)  for a half-normal distribution
    # The multiplier exp(E[u]) is the SFA model's own estimate of the average
    # fraction of latent demand that is suppressed by systemic constraints.
    expected_u   = model.sigma_u * np.sqrt(2.0 / np.pi)
    ineff_mult   = np.exp(expected_u)

    logger.info(
        f"Systemic Inefficiency Multiplier = exp({expected_u:.4f}) = {ineff_mult:.4f}  "
        f"(+{(ineff_mult - 1) * 100:.1f}% uncapping of latent demand)"
    )

    # Ceiling = max(group frontier, outlet's own Jan peak)  ×  uncapping multiplier
    df_result['Maximum_Monthly_Liters'] = (
        np.maximum(df_result['Group_Frontier_Jan_2026'], df_result['Jan_Max'])
        * ineff_mult
    ).round(2).clip(lower=0.01)

    # ── 5. Output & ceiling verification ─────────────────────────────────
    submission = df_result[['Outlet_ID', 'Maximum_Monthly_Liters']]
    submission.to_csv(output_path, index=False)

    pct_above_jan     = (df_result['Maximum_Monthly_Liters'] >= df_result['Jan_Max']).mean()     * 100
    pct_above_overall = (df_result['Maximum_Monthly_Liters'] >= df_result['Overall_Max']).mean() * 100

    logger.info(f"\nPredictions saved to: {output_path}")
    logger.info(f"  Total outlets predicted : {len(submission):,}")
    logger.info(f"  Potential range (L)     : {submission['Maximum_Monthly_Liters'].min():.2f} – {submission['Maximum_Monthly_Liters'].max():.2f}")
    logger.info(f"  Median potential (L)    : {submission['Maximum_Monthly_Liters'].median():.2f}")
    logger.info(f"  Ceiling verification:")
    logger.info(f"    Above January historical peak  : {pct_above_jan:.2f}% of outlets")
    logger.info(f"    Above overall historical peak  : {pct_above_overall:.2f}% of outlets")

    return submission


if __name__ == '__main__':
    generate_predictions()
