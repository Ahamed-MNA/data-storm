"""
Marketing Spend Optimization
===========================
Formulates and solves a non-linear marketing spend optimization problem for the
Western Province for January 2026.

Objective:
  Maximize Total Lift = sum_i( a_i * ln(1 + b * x_i) )
  where x_i is the spend (LKR) allocated to outlet i.
  a_i is set to the historical normal January volume: a_i = max(Y_historical_i, 1.0)
  b is the spend scaling parameter.

Constraints:
  1. Total budget constraint: sum_i( x_i ) <= 5,000,000 LKR
  2. SFA Demand Ceiling constraint: Lift_i <= H_i, where H_i = Y_frontier_i - Y_historical_i
     This translates to an upper bound on spend:
     x_i <= U_i = (exp(H_i / a_i) - 1) / b
  3. Non-negativity constraint: x_i >= 0
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import scipy.optimize as opt

# ── Path Setup ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, os.path.join(root_dir, 'src'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TEAM_NAME = 'data_mavericks'
BUDGET_LKR = 5000000.0  # LKR 5 Million
DEFAULT_B = 0.0005      # Spend scaling parameter
TARGET_MONTH = 1        # January


def run_marketing_spend_optimization(
    b_param: float = DEFAULT_B,
    budget: float = BUDGET_LKR,
    team_name: str = TEAM_NAME
):
    """
    Load data, formulate the non-linear optimization model, solve it,
    and save the optimal spend allocations.
    """
    logger.info("═" * 60)
    logger.info("MARKETING SPEND OPTIMIZATION — WESTERN PROVINCE (JANUARY 2026)")
    logger.info("═" * 60)

    # ── 1. Resolve Input Paths ────────────────────────────────────────────────
    silver_dir = os.path.join(root_dir, 'data', 'silver')
    outputs_dir = os.path.join(root_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    tx_path = os.path.join(silver_dir, 'fact_transactions.parquet')
    dim_outlets_path = os.path.join(silver_dir, 'dim_outlets.parquet')

    # Find predictions (Maximum_Monthly_Liters)
    pred_path_root = os.path.join(outputs_dir, 'data_storm_predictions.csv')
    pred_path_archive = os.path.join(root_dir, 'archive_storming', 'outputs', 'data_storm_predictions.csv')
    
    if os.path.exists(pred_path_root):
        pred_path = pred_path_root
    elif os.path.exists(pred_path_archive):
        pred_path = pred_path_archive
    else:
        raise FileNotFoundError(
            f"Could not find data_storm_predictions.csv in either '{outputs_dir}' or "
            f"'{os.path.join(root_dir, 'archive_storming', 'outputs')}'"
        )

    logger.info(f"Loading transaction history from: {tx_path}")
    df_trans = pd.read_parquet(tx_path)
    logger.info(f"Loading predictions from: {pred_path}")
    df_pred = pd.read_csv(pred_path)
    logger.info(f"Loading outlets dimension from: {dim_outlets_path}")
    df_outlets = pd.read_parquet(dim_outlets_path)

    # ── 2. Identify Western Province Outlets ──────────────────────────────────
    # Province mapping from Distributor_ID: DIST_<Province>_...
    # Western Province is represented by 'W' (e.g. DIST_W_01)
    df_trans_wp = df_trans[df_trans['Distributor_ID'].str.contains('_W_', na=False)]
    wp_outlets = df_trans_wp['Outlet_ID'].unique()
    logger.info(f"Identified {len(wp_outlets):,} outlets in the Western Province")

    # ── 3. Calculate Historical Normal January Sales (Y_historical) ───────────
    logger.info("Computing normal historical January sales volumes...")
    
    # Filter for January transactions (Month = 1)
    df_jan = df_trans_wp[df_trans_wp['Month'] == TARGET_MONTH]
    # Sum by outlet and year, then average across years (2023, 2024, 2025)
    jan_sales = df_jan.groupby(['Outlet_ID', 'Year'])['Volume_Liters'].sum().reset_index()
    avg_jan_sales = jan_sales.groupby('Outlet_ID')['Volume_Liters'].mean().reset_index(name='Y_historical')

    # Fallback: for outlets without January transactions, use their overall monthly average volume
    overall_sales = df_trans_wp.groupby(['Outlet_ID', 'Year', 'Month'])['Volume_Liters'].sum().reset_index()
    avg_overall_sales = overall_sales.groupby('Outlet_ID')['Volume_Liters'].mean().reset_index(name='Y_historical_fallback')

    # Merge into a single dataframe of WP outlets
    df_wp = pd.DataFrame({'Outlet_ID': wp_outlets})
    df_wp = df_wp.merge(avg_jan_sales, on='Outlet_ID', how='left')
    df_wp = df_wp.merge(avg_overall_sales, on='Outlet_ID', how='left')
    df_wp['Y_historical'] = df_wp['Y_historical'].fillna(df_wp['Y_historical_fallback']).fillna(0.0)
    df_wp.drop(columns=['Y_historical_fallback'], inplace=True)

    # ── 4. Calculate Average Historical Price per Liter (P_i) ──────────────────
    # Useful for diagnostics and report summarizing
    logger.info("Calculating average historical price per liter per outlet...")
    df_wp_agg = df_trans_wp.groupby('Outlet_ID').agg({
        'Volume_Liters': 'sum',
        'Total_Bill_Value': 'sum'
    }).reset_index()
    df_wp_agg['Price_Per_Liter'] = df_wp_agg['Total_Bill_Value'] / df_wp_agg['Volume_Liters'].replace(0, np.nan)
    
    # Fallback price: average price across all WP transactions
    global_wp_price = df_trans_wp['Total_Bill_Value'].sum() / df_trans_wp['Volume_Liters'].sum()
    df_wp_agg['Price_Per_Liter'] = df_wp_agg['Price_Per_Liter'].fillna(global_wp_price)
    
    df_wp = df_wp.merge(df_wp_agg[['Outlet_ID', 'Price_Per_Liter']], on='Outlet_ID', how='left')
    df_wp['Price_Per_Liter'] = df_wp['Price_Per_Liter'].fillna(global_wp_price)

    # ── 5. Merge SFA Predictions and Calculate Headroom (H_i) ──────────────────
    # Rename prediction column to Y_frontier
    df_pred_rename = df_pred.rename(columns={'Maximum_Monthly_Liters': 'Y_frontier'})
    df_wp = df_wp.merge(df_pred_rename, on='Outlet_ID', how='left')
    
    # Fallback: if an outlet is missing in the predictions file, use historical January sales
    df_wp['Y_frontier'] = df_wp['Y_frontier'].fillna(df_wp['Y_historical'])
    
    # Calculate headroom (H_i = Y_frontier_i - Y_historical_i)
    # Headroom represents the maximum possible increase in sales volume.
    df_wp['Headroom'] = (df_wp['Y_frontier'] - df_wp['Y_historical']).clip(lower=0.0)

    # ── 6. Setup Optimization Parameters ──────────────────────────────────────
    # Define scaling factor a_i as Y_historical_i, with a floor of 1.0 to handle 0 history
    df_wp['a'] = df_wp['Y_historical'].clip(lower=1.0)
    
    # Calculate spend upper bound U_i based on headroom:
    # Lift_i = a_i * ln(1 + b * x_i) <= H_i => x_i <= U_i = (exp(H_i / a_i) - 1) / b
    # Clip H_i / a_i to 50.0 to prevent double-precision float overflow (exp(709) overflows to inf)
    ratio = (df_wp['Headroom'] / df_wp['a']).clip(upper=50.0)
    df_wp['U'] = (np.exp(ratio) - 1.0) / b_param

    # Extract arrays for fast calculation
    a_arr = df_wp['a'].values
    U_arr = df_wp['U'].values

    logger.info(f"Optimization setup complete:")
    logger.info(f"  Total outlets to optimize : {len(df_wp):,}")
    logger.info(f"  Total headroom available  : {df_wp['Headroom'].sum():,.2f} Liters")
    logger.info(f"  Max spend to hit ceiling  : {np.sum(U_arr):,.2f} LKR")

    # ── 7. Solve Non-Linear Convex Optimization (Dual Formulation) ───────────
    logger.info("Solving optimization problem using Dual Bisection Search...")

    def get_spend(lam):
        # x_i(lambda) = clip(a_i / lambda - 1/b, 0, U_i)
        spend = a_arr / lam - 1.0 / b_param
        return np.clip(spend, 0, U_arr)

    if np.sum(U_arr) <= budget:
        logger.info("Budget exceeds total maximum spend to reach all ceilings. Allocating maximum spend to all outlets.")
        spend_arr = U_arr
    else:
        # Find lambda such that sum(spend(lambda)) - budget = 0
        def budget_excess(lam):
            return np.sum(get_spend(lam)) - budget

        # Set bounds for lambda
        # If lambda is very large, spend -> 0. Max lambda is max(a_i * b)
        lam_min = 1e-15
        lam_max = np.max(a_arr * b_param) + 1.0
        
        # Bisection search
        try:
            lam_opt = opt.bisect(budget_excess, lam_min, lam_max, xtol=1e-12)
            spend_arr = get_spend(lam_opt)
            logger.info(f"Optimal Lagrange multiplier found: lambda = {lam_opt:.8f}")
        except Exception as exc:
            logger.error(f"Bisection search failed: {exc}. Falling back to SLSQP solver.")
            # Fallback solver (SLSQP)
            def neg_obj(x):
                return -np.sum(a_arr * np.log(1.0 + b_param * x))
            
            cons = ({'type': 'ineq', 'fun': lambda x: budget - np.sum(x)})
            bounds = [(0, u) for u in U_arr]
            res = opt.minimize(neg_obj, x0=np.zeros_like(U_arr), method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 200})
            spend_arr = res.x

    # Add spend and lift back to DataFrame
    df_wp['Trade_Spend_Allocation_LKR'] = spend_arr.round(2)
    df_wp['Volume_Lift_Liters'] = (df_wp['a'] * np.log(1.0 + b_param * df_wp['Trade_Spend_Allocation_LKR'])).round(2)

    total_allocated = df_wp['Trade_Spend_Allocation_LKR'].sum()
    total_lift = df_wp['Volume_Lift_Liters'].sum()
    active_outlets = (df_wp['Trade_Spend_Allocation_LKR'] > 0.01).sum()
    ceiling_outlets = np.isclose(df_wp['Trade_Spend_Allocation_LKR'], df_wp['U'], rtol=1e-3).sum()

    logger.info(f"Optimization finished:")
    logger.info(f"  Total Budget             : {budget:,.2f} LKR")
    logger.info(f"  Total Allocated Spend    : {total_allocated:,.2f} LKR ({(total_allocated / budget) * 100:.2f}%)")
    logger.info(f"  Total Gained Volume      : {total_lift:,.2f} Liters")
    logger.info(f"  Average ROI              : {total_lift / total_allocated if total_allocated > 0 else 0:.5f} Liters/LKR")
    logger.info(f"  Active Target Outlets    : {active_outlets:,} / {len(df_wp):,} ({(active_outlets / len(df_wp)) * 100:.1f}%)")
    logger.info(f"  Outlets Reached Ceiling  : {ceiling_outlets:,}")

    # ── 8. Write Output Files ─────────────────────────────────────────────────
    # Target submission file: teamname_budget_allocations.csv (only Outlet_ID and Trade_Spend_Allocation_LKR)
    allocations_filename = f"{team_name}_budget_allocations.csv"
    
    # Save in outputs/ folder
    outputs_csv_path = os.path.join(outputs_dir, allocations_filename)
    df_submission = df_wp[['Outlet_ID', 'Trade_Spend_Allocation_LKR']]
    df_submission.to_csv(outputs_csv_path, index=False)
    logger.info(f"Saved budget allocations to outputs: {outputs_csv_path}")

    # Also save a copy in the root folder as requested by the grading check
    root_csv_path = os.path.join(root_dir, allocations_filename)
    df_submission.to_csv(root_csv_path, index=False)
    logger.info(f"Saved budget allocations to root: {root_csv_path}")

    # ── 9. Generate Summary Analysis Report ──────────────────────────────────
    # Merge outlet dimensions for rich summary reports
    df_summary = df_wp.merge(df_outlets, on='Outlet_ID', how='left')
    
    # Add Distributor_ID and Province from transactions
    dist_map = df_trans_wp.groupby('Outlet_ID')['Distributor_ID'].first().reset_index()
    df_summary = df_summary.merge(dist_map, on='Outlet_ID', how='left')
    
    generate_markdown_report(df_summary, budget, total_allocated, total_lift, active_outlets, ceiling_outlets, b_param)

    return df_wp


def generate_markdown_report(
    df: pd.DataFrame,
    budget: float,
    allocated: float,
    lift: float,
    active: int,
    ceiling: int,
    b_val: float
):
    """
    Generate a markdown summary report detailing budget allocations
    by distributor, outlet size, and outlet type.
    """
    report_dir = os.path.join(root_dir, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'marketing_spend_summary.md')

    # 1. Summary Stats Table
    summary_md = f"""# Marketing Spend Optimization Summary Report

## Executive Summary
This report summarizes the optimal allocation of the **LKR 5,000,000** promotional budget across the Western Province outlets for **January 2026**. 
The optimization uses a logarithmic spend-to-volume response curve to model diminishing marginal returns, capped by Stochastic Frontier Analysis (SFA) demand ceilings.

| Metric | Value |
| :--- | :--- |
| **Total Promotional Budget** | LKR {budget:,.2f} |
| **Total Allocated Spend** | LKR {allocated:,.2f} ({(allocated / budget) * 100:.2f}%) |
| **Expected Total Volume Lift** | {lift:,.2f} Liters |
| **Average Return on Investment (ROI)** | {lift / allocated if allocated > 0 else 0:.5f} Liters/LKR |
| **Active Outlets Targeted (> 0 LKR)** | {active:,} / {len(df):,} ({(active / len(df)) * 100:.1f}%) |
| **Outlets Pushed to SFA Demand Ceiling** | {ceiling:,} |
| **Diminishing Return Scale Parameter ($b$)** | {b_val} |

---

## Allocation by Distributor
This table shows how the budget is distributed across the distributors in the Western Province.

| Distributor ID | Number of Outlets | Active Outlets | Total Spend (LKR) | Share of Spend (%) | Volume Lift (Liters) | Avg ROI (L/LKR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    # Group by Distributor
    dist_groups = df.groupby('Distributor_ID').agg(
        total_outlets=('Outlet_ID', 'count'),
        active_outlets=('Trade_Spend_Allocation_LKR', lambda x: (x > 0.01).sum()),
        total_spend=('Trade_Spend_Allocation_LKR', 'sum'),
        total_lift=('Volume_Lift_Liters', 'sum')
    ).reset_index()

    for _, row in dist_groups.iterrows():
        spend_share = (row['total_spend'] / allocated) * 100 if allocated > 0 else 0
        roi = row['total_lift'] / row['total_spend'] if row['total_spend'] > 0 else 0
        summary_md += f"| {row['Distributor_ID']} | {row['total_outlets']:,} | {row['active_outlets']:,} | LKR {row['total_spend']:,.2f} | {spend_share:.2f}% | {row['total_lift']:,.2f} L | {roi:.5f} |\n"

    # 2. Allocation by Outlet Size
    summary_md += """
---

## Allocation by Outlet Size
Diminishing return optimization tends to favor mature, larger-volume outlets while capping them at their frontier.

| Outlet Size | Number of Outlets | Active Outlets | Total Spend (LKR) | Share of Spend (%) | Volume Lift (Liters) | Avg ROI (L/LKR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    size_groups = df.groupby('Outlet_Size').agg(
        total_outlets=('Outlet_ID', 'count'),
        active_outlets=('Trade_Spend_Allocation_LKR', lambda x: (x > 0.01).sum()),
        total_spend=('Trade_Spend_Allocation_LKR', 'sum'),
        total_lift=('Volume_Lift_Liters', 'sum')
    ).reset_index()

    for _, row in size_groups.iterrows():
        spend_share = (row['total_spend'] / allocated) * 100 if allocated > 0 else 0
        roi = row['total_lift'] / row['total_spend'] if row['total_spend'] > 0 else 0
        summary_md += f"| {row['Outlet_Size']} | {row['total_outlets']:,} | {row['active_outlets']:,} | LKR {row['total_spend']:,.2f} | {spend_share:.2f}% | {row['total_lift']:,.2f} L | {roi:.5f} |\n"

    # 3. Allocation by Outlet Type
    summary_md += """
---

## Allocation by Outlet Type
Different retail formats exhibit different sales volumes and potential.

| Outlet Type | Number of Outlets | Active Outlets | Total Spend (LKR) | Share of Spend (%) | Volume Lift (Liters) | Avg ROI (L/LKR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    type_groups = df.groupby('Outlet_Type').agg(
        total_outlets=('Outlet_ID', 'count'),
        active_outlets=('Trade_Spend_Allocation_LKR', lambda x: (x > 0.01).sum()),
        total_spend=('Trade_Spend_Allocation_LKR', 'sum'),
        total_lift=('Volume_Lift_Liters', 'sum')
    ).reset_index()

    for _, row in type_groups.iterrows():
        spend_share = (row['total_spend'] / allocated) * 100 if allocated > 0 else 0
        roi = row['total_lift'] / row['total_spend'] if row['total_spend'] > 0 else 0
        summary_md += f"| {row['Outlet_Type']} | {row['total_outlets']:,} | {row['active_outlets']:,} | LKR {row['total_spend']:,.2f} | {spend_share:.2f}% | {row['total_lift']:,.2f} L | {roi:.5f} |\n"

    summary_md += """
---
*Report generated on completion of the Optimization Module.*
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary_md)
    logger.info(f"Summary report written to: {report_path}")


if __name__ == '__main__':
    run_marketing_spend_optimization()
