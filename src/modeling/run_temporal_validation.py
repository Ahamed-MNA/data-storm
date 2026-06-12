#!/usr/bin/env python
"""
Temporal Validation: Train on 70%, Test on 15%
Run this script to execute the full pipeline
"""
import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.special import log_ndtr
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

TARGET_MONTH = 1  # January
JAN_AVG_HOLIDAY_COUNT = 11.3

# Path setup
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
silver_path = os.path.join(root_dir, 'data', 'silver')
output_dir = os.path.join(root_dir, 'outputs')
splits_dir = os.path.join(root_dir, 'data', 'splits')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(splits_dir, exist_ok=True)

print(f"Root directory: {root_dir}")
print(f"Output directory: {output_dir}\n")

# ── SFA Model Class ────────────────────────────────────────────────────────────
def sfa_log_likelihood(params, X, y):
    """Log-likelihood for SFA with half-normal inefficiency."""
    k = X.shape[1]
    beta = params[:k]
    sigma_u = np.exp(params[k])
    sigma_v = np.exp(params[k+1])

    sigma_sq = sigma_u**2 + sigma_v**2
    sigma = np.sqrt(sigma_sq)
    lam = sigma_u / sigma_v

    epsilon = y - np.dot(X, beta)

    term1 = np.log(2/sigma)
    z = epsilon / sigma
    term2 = -0.9189385332046727 - 0.5 * (z**2)
    term3 = log_ndtr(-z * lam)

    ll = np.sum(term1 + term2 + term3)
    return -ll

class SFAModel:
    def __init__(self):
        self.beta = None
        self.sigma_u = None
        self.sigma_v = None
        self.feature_names = None

    def fit(self, X, y, verbose=True):
        self.feature_names = X.columns.tolist()
        X_val = X.values
        y_val = y.values

        k = X_val.shape[1]
        initial_beta = np.linalg.lstsq(X_val, y_val, rcond=None)[0]
        initial_params = np.concatenate([initial_beta, [0.1, 0.1]])

        logger.info(f"Starting SFA optimization for {k} features on {len(X_val):,} records...")

        iteration = 0
        def callback(xk):
            nonlocal iteration
            iteration += 1
            if verbose and iteration % 10 == 0:
                ll_val = sfa_log_likelihood(xk, X_val, y_val)
                logger.info(f"Iteration {iteration:03d}: Neg Log-Likelihood = {ll_val:.4f}")

        res = minimize(sfa_log_likelihood, initial_params, args=(X_val, y_val), method='L-BFGS-B', callback=callback)

        if res.success:
            self.beta = res.x[:k]
            self.sigma_u = np.exp(res.x[k])
            self.sigma_v = np.exp(res.x[k+1])
            logger.info("SFA Model fitted successfully.")
            logger.info(f"  sigma_u (inefficiency): {self.sigma_u:.6f}")
            logger.info(f"  sigma_v (noise):        {self.sigma_v:.6f}")
        else:
            logger.error(f"SFA Optimization failed: {res.message}")
            raise Exception("Model fitting failed.")

    def predict_potential(self, X):
        """Predicts potential (frontier) volume."""
        X_val = X.values
        ln_y_potential = np.dot(X_val, self.beta)
        y_potential = np.exp(ln_y_potential) * np.exp(0.5 * self.sigma_v**2)
        return y_potential

# ── Step 1: Load data ──────────────────────────────────────────────────────────
logger.info("Loading transaction data...")
df_trans = pd.read_parquet(os.path.join(silver_path, 'fact_transactions.parquet'))
df_outlets = pd.read_parquet(os.path.join(silver_path, 'dim_outlets.parquet'))
df_season = pd.read_parquet(os.path.join(silver_path, 'dim_distributor_seasonality.parquet'))

# Create date column
df_trans['Date'] = pd.to_datetime(
    df_trans['Year'].astype(str) + '-' +
    df_trans['Month'].astype(str).str.zfill(2) + '-01'
)

print(f"Transactions shape: {df_trans.shape}")
print(f"Date range: {df_trans['Date'].min().date()} to {df_trans['Date'].max().date()}\n")

# ── Step 2: Temporal split ─────────────────────────────────────────────────────
df_trans = df_trans.sort_values('Date').reset_index(drop=True)

n_records = len(df_trans)
n_train = int(n_records * TRAIN_RATIO)
n_val = int(n_records * VAL_RATIO)

df_train = df_trans.iloc[:n_train].copy()
df_val = df_trans.iloc[n_train:n_train + n_val].copy()
df_test = df_trans.iloc[n_train + n_val:].copy()

logger.info(f"Total records: {n_records:,}")
logger.info(f"Train: {len(df_train):,} ({100*len(df_train)/n_records:.1f}%) | {df_train['Date'].min().date()} to {df_train['Date'].max().date()}")
logger.info(f"Val:   {len(df_val):,} ({100*len(df_val)/n_records:.1f}%) | {df_val['Date'].min().date()} to {df_val['Date'].max().date()}")
logger.info(f"Test:  {len(df_test):,} ({100*len(df_test)/n_records:.1f}%) | {df_test['Date'].min().date()} to {df_test['Date'].max().date()}\n")

# ── Step 3: Prepare training data ──────────────────────────────────────────────
logger.info("Preparing training data for SFA model...")
positive_trans = df_train[df_train['Volume_Liters'] > 0]
monthly_vol = positive_trans.groupby(['Outlet_ID', 'Year', 'Month'])['Volume_Liters'].sum().reset_index()
monthly_vol['ln_volume'] = np.log(monthly_vol['Volume_Liters'])

# Get outlet dimensions from silver layer
outlet_features = df_outlets[['Outlet_ID']].drop_duplicates()

train_data = monthly_vol.merge(outlet_features, on='Outlet_ID', how='left')
logger.info(f"Training data: {len(train_data):,} monthly observations\n")

# ── Step 4: Add features to training data ──────────────────────────────────────
outlet_dist = df_train.groupby('Outlet_ID')['Distributor_ID'].first().reset_index()
jan_season = (df_season[df_season['Month'] == TARGET_MONTH]
              .groupby('Distributor_ID')['Seasonality_Index']
              .first()
              .reset_index())

outlet_jan_season = outlet_dist.merge(jan_season, on='Distributor_ID', how='left')

train_data = train_data.merge(
    outlet_jan_season[['Outlet_ID', 'Seasonality_Index']],
    on='Outlet_ID', how='left'
)

train_data['Seasonality_Index_Moderate'] = (train_data['Seasonality_Index'] == 'Moderate').astype(int)
train_data['Seasonality_Index_Un-Favorable'] = (train_data['Seasonality_Index'] == 'Un-Favorable').astype(int)
train_data['Holiday_Count'] = JAN_AVG_HOLIDAY_COUNT
train_data['Intercept'] = 1.0

feature_cols = [col for col in train_data.columns if col not in [
    'Outlet_ID', 'Year', 'Month', 'Volume_Liters', 'ln_volume',
    'Distributor_ID', 'Seasonality_Index', 'Date'
]]

selected_features = [col for col in feature_cols if col in train_data.columns]
print(f"Selected features for SFA: {selected_features}\n")

X_train = train_data[selected_features].fillna(0)
y_train = train_data['ln_volume']

logger.info(f"SFA training: X shape {X_train.shape}, y shape {y_train.shape}\n")

# ── Step 5: Train SFA model ────────────────────────────────────────────────────
sfa_model = SFAModel()
sfa_model.fit(X_train, y_train, verbose=True)

print(f"\nModel parameters:")
print(f"  Beta coefficients: {len(sfa_model.beta)}")
print(f"  Sigma_u (inefficiency): {sfa_model.sigma_u:.6f}")
print(f"  Sigma_v (noise): {sfa_model.sigma_v:.6f}\n")

# ── Step 6: Generate predictions for test set ──────────────────────────────────
logger.info("Generating predictions for test outlets...\n")

test_outlet_dist = df_test.groupby('Outlet_ID')['Distributor_ID'].first().reset_index()

test_outlet_features = df_outlets[['Outlet_ID']].drop_duplicates()

test_outlet_features = test_outlet_features.merge(
    test_outlet_dist[['Outlet_ID', 'Distributor_ID']],
    on='Outlet_ID', how='left'
)

test_outlet_features = test_outlet_features.merge(
    outlet_jan_season[['Outlet_ID', 'Seasonality_Index']],
    on='Outlet_ID', how='left'
)

test_outlet_features['Seasonality_Index_Moderate'] = (test_outlet_features['Seasonality_Index'] == 'Moderate').astype(int)
test_outlet_features['Seasonality_Index_Un-Favorable'] = (test_outlet_features['Seasonality_Index'] == 'Un-Favorable').astype(int)
test_outlet_features['Holiday_Count'] = JAN_AVG_HOLIDAY_COUNT
test_outlet_features['Intercept'] = 1.0

X_test = test_outlet_features[sfa_model.feature_names].fillna(0)
test_outlet_features['Group_Frontier_Jan'] = sfa_model.predict_potential(X_test)

logger.info(f"Generated predictions for {len(test_outlet_features):,} test outlets\n")

# ── Step 7: Apply inefficiency multiplier ──────────────────────────────────────
expected_u = sfa_model.sigma_u * np.sqrt(2.0 / np.pi)
ineff_mult = np.exp(expected_u)

logger.info(
    f"Inefficiency Multiplier = exp({expected_u:.4f}) = {ineff_mult:.4f} "
    f"(+{(ineff_mult - 1) * 100:.1f}% uncapping)\n"
)

# Get historical maximum from test data
positive_trans_test = df_test[df_test['Volume_Liters'] > 0]
monthly_vol_test = positive_trans_test.groupby(['Outlet_ID', 'Year', 'Month'])['Volume_Liters'].sum().reset_index()
overall_max_test = monthly_vol_test.groupby('Outlet_ID')['Volume_Liters'].max().reset_index(name='Overall_Max')
jan_max_test = (
    monthly_vol_test[monthly_vol_test['Month'] == TARGET_MONTH]
    .groupby('Outlet_ID')['Volume_Liters']
    .max()
    .reset_index(name='Jan_Max')
)

df_result = (
    test_outlet_features[['Outlet_ID', 'Group_Frontier_Jan']]
    .merge(overall_max_test, on='Outlet_ID', how='left')
    .merge(jan_max_test, on='Outlet_ID', how='left')
)

df_result['Jan_Max'] = df_result['Jan_Max'].fillna(df_result['Overall_Max'])

df_result['Maximum_Monthly_Liters'] = (
    np.maximum(df_result['Group_Frontier_Jan'], df_result['Jan_Max'])
    * ineff_mult
).round(2).clip(lower=0.01)

submission = df_result[['Outlet_ID', 'Maximum_Monthly_Liters']]

print(f"Test Predictions Summary:")
print(f"  Total outlets: {len(submission):,}")
print(f"  Min: {submission['Maximum_Monthly_Liters'].min():.2f} L")
print(f"  Max: {submission['Maximum_Monthly_Liters'].max():.2f} L")
print(f"  Mean: {submission['Maximum_Monthly_Liters'].mean():.2f} L")
print(f"  Median: {submission['Maximum_Monthly_Liters'].median():.2f} L\n")

# ── Step 8: Get actual test data ───────────────────────────────────────────────
test_max = (
    monthly_vol_test
    .groupby('Outlet_ID')['Volume_Liters'].max()
    .reset_index(name='Max_Volume_Actual')
)

comparison = test_max.merge(submission, on='Outlet_ID', how='inner')
comparison = comparison.rename(columns={'Maximum_Monthly_Liters': 'Predicted_Volume'})

logger.info(f"Comparison data:")
logger.info(f"  Outlets with actual test data: {len(test_max):,}")
logger.info(f"  Outlets with predictions: {len(submission):,}")
logger.info(f"  Outlets in comparison: {len(comparison):,}\n")

# ── Step 9: Calculate statistics ───────────────────────────────────────────────
stats = {
    'n_outlets': len(comparison),
    'actual': {
        'min': comparison['Max_Volume_Actual'].min(),
        'max': comparison['Max_Volume_Actual'].max(),
        'mean': comparison['Max_Volume_Actual'].mean(),
        'median': comparison['Max_Volume_Actual'].median(),
        'std': comparison['Max_Volume_Actual'].std(),
        'q25': comparison['Max_Volume_Actual'].quantile(0.25),
        'q75': comparison['Max_Volume_Actual'].quantile(0.75),
    },
    'predicted': {
        'min': comparison['Predicted_Volume'].min(),
        'max': comparison['Predicted_Volume'].max(),
        'mean': comparison['Predicted_Volume'].mean(),
        'median': comparison['Predicted_Volume'].median(),
        'std': comparison['Predicted_Volume'].std(),
        'q25': comparison['Predicted_Volume'].quantile(0.25),
        'q75': comparison['Predicted_Volume'].quantile(0.75),
    }
}

comparison['Pred_to_Actual_Ratio'] = comparison['Predicted_Volume'] / comparison['Max_Volume_Actual']
stats['pred_actual_ratio_mean'] = comparison['Pred_to_Actual_Ratio'].mean()
stats['pred_actual_ratio_median'] = comparison['Pred_to_Actual_Ratio'].median()
stats['pred_above_actual'] = (comparison['Predicted_Volume'] >= comparison['Max_Volume_Actual']).mean() * 100

print("="*70)
print("DISTRIBUTION COMPARISON: Actual Test vs Predictions (Trained on 70%)")
print("="*70)
print(f"Number of outlets: {stats['n_outlets']:,}\n")

print("Actual Test Data (Jul-Dec 2025):")
for key, val in stats['actual'].items():
    print(f"  {key.upper():8} : {val:12,.2f}")

print("\nPredicted Data (SFA trained on Jan 2023 - Feb 2025):")
for key, val in stats['predicted'].items():
    print(f"  {key.upper():8} : {val:12,.2f}")

print(f"\nPrediction vs Actual:")
print(f"  Mean ratio (Pred/Actual)   : {stats['pred_actual_ratio_mean']:.4f}x")
print(f"  Median ratio (Pred/Actual) : {stats['pred_actual_ratio_median']:.4f}x")
print(f"  % Predictions >= Actual    : {stats['pred_above_actual']:.2f}%")
print("="*70 + "\n")

# ── Step 10: Create visualizations ─────────────────────────────────────────────
logger.info("Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Temporal Validation: Predictions (70% train) vs Actual Test Data (15% test)', fontsize=14, fontweight='bold')

# 1. Histogram with KDE curve
ax = axes[0, 0]

n_actual, bins_actual, patches_actual = ax.hist(
    comparison['Max_Volume_Actual'],
    bins=60,
    alpha=0.5,
    label='Actual Test',
    color='#2E86AB',
    edgecolor='#1B4965',
    density=True
)

n_pred, bins_pred, patches_pred = ax.hist(
    comparison['Predicted_Volume'],
    bins=60,
    alpha=0.5,
    label='Predicted',
    color='#A23B72',
    edgecolor='#6C1E5A',
    density=True
)

kde_actual = gaussian_kde(comparison['Max_Volume_Actual'].dropna())
kde_pred = gaussian_kde(comparison['Predicted_Volume'].dropna())

x_range = np.linspace(
    min(comparison['Max_Volume_Actual'].min(), comparison['Predicted_Volume'].min()),
    max(comparison['Max_Volume_Actual'].max(), comparison['Predicted_Volume'].max()),
    200
)

ax.fill_between(x_range, kde_actual(x_range), alpha=0.3, color='#2E86AB', label='Actual KDE')
ax.fill_between(x_range, kde_pred(x_range), alpha=0.3, color='#A23B72', label='Predicted KDE')

ax.plot(x_range, kde_actual(x_range), color='#1B4965', linewidth=2)
ax.plot(x_range, kde_pred(x_range), color='#6C1E5A', linewidth=2)

ax.set_xlabel('Volume (Liters)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Distribution with Kernel Density Estimate', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_axisbelow(True)

# 2. Box plot
ax = axes[0, 1]
box_data = [comparison['Max_Volume_Actual'], comparison['Predicted_Volume']]
bp = ax.boxplot(box_data, tick_labels=['Actual Test', 'Predicted'], patch_artist=True)
colors = ['#2E86AB', '#A23B72']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for whisker in bp['whiskers']:
    whisker.set(linewidth=1.5)
for median in bp['medians']:
    median.set(color='#1B4965', linewidth=2)
ax.set_ylabel('Volume (Liters)', fontsize=11, fontweight='bold')
ax.set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2, axis='y', linestyle='--')
ax.set_axisbelow(True)

# 3. Scatter plot
ax = axes[1, 0]
ax.scatter(
    comparison['Max_Volume_Actual'],
    comparison['Predicted_Volume'],
    alpha=0.4,
    s=15,
    color='#2E86AB',
    edgecolors='#1B4965',
    linewidth=0.5
)

min_val = min(comparison['Max_Volume_Actual'].min(), comparison['Predicted_Volume'].min())
max_val = max(comparison['Max_Volume_Actual'].max(), comparison['Predicted_Volume'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction', alpha=0.8)

ax.set_xlabel('Actual Test Volume (Liters)', fontsize=11, fontweight='bold')
ax.set_ylabel('Predicted Volume (Liters)', fontsize=11, fontweight='bold')
ax.set_title('Actual vs Predicted (Scatter)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_axisbelow(True)

# 4. Ratio distribution
ax = axes[1, 1]
ratio = comparison['Predicted_Volume'] / comparison['Max_Volume_Actual']

n_ratio, bins_ratio, patches_ratio = ax.hist(
    ratio,
    bins=60,
    alpha=0.6,
    color='#F18F01',
    edgecolor='#C7601C',
    density=True,
    label='Ratio Distribution'
)

kde_ratio = gaussian_kde(ratio.dropna())
x_ratio = np.linspace(ratio.min(), ratio.max(), 200)
ax.fill_between(x_ratio, kde_ratio(x_ratio), alpha=0.4, color='#F18F01', label='KDE')
ax.plot(x_ratio, kde_ratio(x_ratio), color='#C7601C', linewidth=2)

ax.axvline(ratio.mean(), color='#E63946', linestyle='--', linewidth=2.5, label=f'Mean: {ratio.mean():.3f}x', alpha=0.8)
ax.axvline(ratio.median(), color='#457B9D', linestyle='--', linewidth=2.5, label=f'Median: {ratio.median():.3f}x', alpha=0.8)

ax.set_xlabel('Predicted / Actual Ratio', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Prediction Ratio Distribution with KDE', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plot_path = os.path.join(output_dir, 'temporal_validation_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
logger.info(f"Plot saved to: {plot_path}")
plt.close()

# ── Step 11: Save results ──────────────────────────────────────────────────────
logger.info("Saving results...")

comparison.to_csv(os.path.join(output_dir, 'temporal_validation_comparison.csv'), index=False)
submission.to_csv(os.path.join(output_dir, 'test_predictions_from_train70.csv'), index=False)

stats_json = {k: v for k, v in stats.items() if not isinstance(v, dict)}
stats_json['actual'] = {k: float(v) for k, v in stats['actual'].items()}
stats_json['predicted'] = {k: float(v) for k, v in stats['predicted'].items()}

with open(os.path.join(output_dir, 'temporal_validation_stats.json'), 'w') as f:
    json.dump(stats_json, f, indent=2)

with open(os.path.join(output_dir, 'sfa_model_train70.pkl'), 'wb') as f:
    pickle.dump(sfa_model, f)

logger.info(f"Results saved to {output_dir}")
print(f"\n✓ Comparison CSV: temporal_validation_comparison.csv")
print(f"✓ Predictions CSV: test_predictions_from_train70.csv")
print(f"✓ Statistics JSON: temporal_validation_stats.json")
print(f"✓ Trained Model: sfa_model_train70.pkl")
print(f"✓ Visualization: temporal_validation_comparison.png")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
