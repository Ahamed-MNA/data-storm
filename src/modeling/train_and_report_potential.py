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

from modeling.sfa_model import SFAModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_refined_sfa_pipeline():
    gold_path = os.path.join(root_dir, 'data', 'gold')
    input_file = os.path.join(gold_path, 'sfa_refined.parquet')
    
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found. Please build refined Gold layer first.")
        return

    logger.info("Loading Refined Gold layer data...")
    df = pd.read_parquet(input_file)

    # Setup output directory
    output_dir = os.path.join(root_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Split Features ---
    # Constraint Proxies (NOT used in frontier, used for validation/reporting)
    proxy_cols = ['CV_Volume', 'Flatline_Score', 'Round_Number_Bias', 'Price_Rigidity']
    
    # Target
    y = df['ln_volume']
    
    # Identify Demand Drivers for the frontier function f(x)
    # Exclude metadata, target, and proxies
    exclude_cols = ['Outlet_ID', 'Year', 'Month', 'Distributor_ID', 'Volume_Liters', 'Total_Bill_Value', 
                    'Latitude', 'Longitude', 'ln_volume'] + proxy_cols
    
    X = df.drop(columns=exclude_cols)
    X['Intercept'] = 1.0

    logger.info(f"Training SFA model with {len(X)} records and {X.shape[1]} demand drivers...")
    
    # --- 2. Fit Model ---
    model = SFAModel()
    model.fit(X, y)

    # Save fitted model
    model_file = os.path.join(output_dir, 'sfa_model.pkl')
    model.save(model_file)

    # --- 3. Predict Potential and Efficiency ---
    logger.info("Calculating Potential and Efficiency...")
    df['Potential_Volume'] = model.predict_potential(X)
    df['Efficiency_Score'] = df['Volume_Liters'] / df['Potential_Volume']
    
    # --- 4. Generate Outlet-Level Report ---
    # Aggregate back to Outlet level, including proxies for interpretation
    agg_dict = {
        'Volume_Liters': 'mean',
        'Potential_Volume': 'mean',
        'Efficiency_Score': 'mean'
    }
    # Add proxies to aggregation (they are already outlet-level, but take mean to be safe)
    for col in proxy_cols:
        agg_dict[col] = 'mean'

    outlet_report = df.groupby('Outlet_ID').agg(agg_dict).reset_index()
    outlet_report.rename(columns={'Volume_Liters': 'Avg_Monthly_Volume'}, inplace=True)

    # --- 5. Save Report ---
    output_file = os.path.join(output_dir, 'outlet_potential_report_refined.parquet')
    outlet_report.to_parquet(output_file, index=False)
    
    logger.info(f"Refined Report saved to: {output_file}")
    
    # --- 6. Quick Analysis ---
    # Correlation between Efficiency and Proxies
    logger.info("\n--- CORRELATION: EFFICIENCY vs CONSTRAINT PROXIES ---")
    corrs = outlet_report[['Efficiency_Score'] + proxy_cols].corr()['Efficiency_Score'].drop('Efficiency_Score')
    print(corrs)
    
    # --- 7. Print SFA Model Parameters & Weights ---
    logger.info("\n--- ESTIMATED SFA MODEL PARAMETERS & WEIGHTS ---")
    weights_df = pd.DataFrame({
        'Feature': model.feature_names,
        'Beta (Weight)': model.beta
    })
    # Sort by absolute weight value to see most important demand drivers
    weights_df['Abs_Beta'] = weights_df['Beta (Weight)'].abs()
    weights_df = weights_df.sort_values(by='Abs_Beta', ascending=False).drop(columns=['Abs_Beta']).reset_index(drop=True)
    print(weights_df.to_string())

    print(f"\nSigma_u (Systemic Inefficiency standard deviation): {model.sigma_u:.6f}")
    print(f"Sigma_v (Random Noise standard deviation): {model.sigma_v:.6f}")
    print(f"Total variance (Sigma^2 = Sigma_u^2 + Sigma_v^2): {(model.sigma_u**2 + model.sigma_v**2):.6f}")
    print(f"Inefficiency ratio (Lambda = Sigma_u / Sigma_v): {(model.sigma_u / model.sigma_v):.6f}")

    return model

if __name__ == "__main__":
    run_refined_sfa_pipeline()
