"""
Generate Final Predictions
==========================
Reads the outlet-level SFA potential report and produces the final
competition submission file:  outputs/teamname_predictions.csv

Schema required by the judges:
  Outlet_ID | Maximum_Monthly_Liters
"""
import os
import sys
import logging
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
TEAM_NAME       = 'data_storm'          # Change to your actual team name
TARGET_YEAR     = 2026
TARGET_MONTH    = 1                     # January 2026


def generate_predictions(
    report_path: str | None = None,
    output_path: str | None = None,
    team_name:   str        = TEAM_NAME,
) -> pd.DataFrame:
    """
    Reads the SFA outlet potential report and writes the final predictions CSV.

    The January 2026 potential is estimated by applying the distributor
    seasonality index for January to the model's base potential estimate.

    Parameters
    ----------
    report_path : str, optional
        Path to the SFA outlet report parquet (defaults to
        ``outputs/outlet_potential_report_refined.parquet``).
    output_path : str, optional
        Destination CSV path (defaults to
        ``outputs/<team_name>_predictions.csv``).
    team_name : str
        Used to name the output file.

    Returns
    -------
    pd.DataFrame  with columns  Outlet_ID, Maximum_Monthly_Liters
    """
    output_dir = os.path.join(root_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    if report_path is None:
        report_path = os.path.join(output_dir, 'outlet_potential_report_refined.parquet')
    if output_path is None:
        output_path = os.path.join(output_dir, f'{team_name}_predictions.csv')

    # ── 1. Load SFA outlet report ──────────────────────────────────────────
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            f"SFA report not found at: {report_path}\n"
            "Run the full pipeline first:  python main.py"
        )

    logger.info(f"Loading SFA report from: {report_path}")
    report = pd.read_parquet(report_path)
    logger.info(f"  Outlets in report: {len(report):,}")

    # ── 2. Apply January seasonality adjustment ────────────────────────────
    silver_path = os.path.join(root_dir, 'data', 'silver')
    season_file = os.path.join(silver_path, 'dim_distributor_seasonality.parquet')

    jan_multiplier = 1.0   # fallback: no adjustment
    if os.path.exists(season_file):
        df_season = pd.read_parquet(season_file)

        # Filter for January of target year (or any year if target year missing)
        jan_rows = df_season[df_season['Month'] == TARGET_MONTH]
        if TARGET_YEAR in jan_rows['Year'].values:
            jan_rows = jan_rows[jan_rows['Year'] == TARGET_YEAR]

        if not jan_rows.empty and 'Seasonality_Index' in jan_rows.columns:
            # Use the mean seasonality index across all distributors for January
            jan_multiplier = pd.to_numeric(jan_rows['Seasonality_Index'], errors='coerce').mean()
            if pd.isna(jan_multiplier):
                jan_multiplier = 1.0
            logger.info(f"  January seasonality multiplier (mean): {jan_multiplier:.4f}")
        else:
            logger.warning("  No January seasonality data found — using multiplier = 1.0")
    else:
        logger.warning("  Seasonality file not found — using multiplier = 1.0")

    # ── 3. Compute Maximum_Monthly_Liters for January 2026 ────────────────
    predictions = report[['Outlet_ID', 'Potential_Volume']].copy()
    predictions['Maximum_Monthly_Liters'] = (
        predictions['Potential_Volume'] * jan_multiplier
    ).round(2)

    # Sanity clamp: potential cannot be negative or zero
    predictions['Maximum_Monthly_Liters'] = predictions['Maximum_Monthly_Liters'].clip(lower=0.01)

    # ── 4. Final output ───────────────────────────────────────────────────
    submission = predictions[['Outlet_ID', 'Maximum_Monthly_Liters']]

    submission.to_csv(output_path, index=False)
    logger.info(f"\nPredictions saved to: {output_path}")
    logger.info(f"  Total outlets predicted : {len(submission):,}")
    logger.info(f"  Potential range (L)     : {submission['Maximum_Monthly_Liters'].min():.1f}  –  {submission['Maximum_Monthly_Liters'].max():.1f}")
    logger.info(f"  Median potential (L)    : {submission['Maximum_Monthly_Liters'].median():.1f}")

    return submission


if __name__ == '__main__':
    generate_predictions()
