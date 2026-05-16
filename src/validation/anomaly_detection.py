import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Transaction Anomalies ---

def detect_negative_quantities(df: pd.DataFrame, col: str = 'Volume_Liters') -> pd.DataFrame:
    """Returns rows with negative quantities."""
    return df[df[col] < 0]

def detect_impossible_spikes(df: pd.DataFrame, col: str = 'Total_Bill_Value', threshold_z: float = 3.0) -> pd.DataFrame:
    """Returns rows where the value is an outlier (default: > 3 standard deviations)."""
    mean = df[col].mean()
    std = df[col].std()
    return df[df[col] > (mean + threshold_z * std)]

def detect_duplicate_invoices(df: pd.DataFrame, keys: list = None) -> pd.DataFrame:
    """Returns duplicate rows based on provided keys."""
    if keys is None:
        keys = df.columns.tolist()
    return df[df.duplicated(subset=keys, keep=False)]

def detect_repeated_timestamps(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Returns rows where multiple transactions occur at the exact same timestamp (for the same outlet)."""
    if time_col not in df.columns:
        logger.warning(f"Time column '{time_col}' not found. Skipping repeated timestamp check.")
        return pd.DataFrame()
    return df[df.duplicated(subset=['Outlet_ID', time_col], keep=False)]

def detect_midnight_batches(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Returns rows where the timestamp is exactly midnight (often indicative of batch uploads)."""
    if time_col not in df.columns:
        return pd.DataFrame()
    # Convert to datetime if not already
    times = pd.to_datetime(df[time_col])
    return df[(times.dt.hour == 0) & (times.dt.minute == 0) & (times.dt.second == 0)]

# --- Outlet Anomalies ---

def detect_duplicate_gps(df_coords: pd.DataFrame) -> pd.DataFrame:
    """Returns outlets with identical Latitude and Longitude."""
    return df_coords[df_coords.duplicated(subset=['Latitude', 'Longitude'], keep=False)]

def detect_invalid_coordinates(df_coords: pd.DataFrame) -> pd.DataFrame:
    """Returns rows with coordinates outside global bounds."""
    return df_coords[(df_coords['Latitude'] < -90) | (df_coords['Latitude'] > 90) |
                     (df_coords['Longitude'] < -180) | (df_coords['Longitude'] > 180)]

def detect_outlets_in_ocean(df_coords: pd.DataFrame) -> pd.DataFrame:
    """Returns outlets outside the bounding box of Sri Lanka."""
    # Approximate bounding box for Sri Lanka
    LAT_RANGE = (5.9, 9.9)
    LON_RANGE = (79.6, 81.9)
    return df_coords[~((df_coords['Latitude'].between(*LAT_RANGE)) & 
                       (df_coords['Longitude'].between(*LON_RANGE)))]

def detect_inactive_outlets(df_master: pd.DataFrame, df_trans: pd.DataFrame) -> pd.DataFrame:
    """Returns outlets present in master but missing from transaction history."""
    active_outlets = df_trans['Outlet_ID'].unique()
    return df_master[~df_master['Outlet_ID'].isin(active_outlets)]

def detect_extreme_volatility(df_trans: pd.DataFrame, col: str = 'Total_Bill_Value') -> pd.DataFrame:
    """Returns outlets where the sales coefficient of variation (std/mean) is extreme (> 2.0)."""
    stats = df_trans.groupby('Outlet_ID')[col].agg(['mean', 'std'])
    stats['cv'] = stats['std'] / stats['mean']
    volatile_ids = stats[stats['cv'] > 2.0].index
    return df_trans[df_trans['Outlet_ID'].isin(volatile_ids)]

# --- Distributor Anomalies ---

def detect_delivery_caps(df_trans: pd.DataFrame, col: str = 'Volume_Liters') -> pd.DataFrame:
    """Returns distributors who hit the exact same total volume/value in multiple months (indicative of caps)."""
    monthly_dist = df_trans.groupby(['Distributor_ID', 'Year', 'Month'])[col].sum().reset_index()
    # Find cases where the same Distributor has the same sum in different months
    caps = monthly_dist[monthly_dist.duplicated(subset=['Distributor_ID', col], keep=False)]
    return caps

def detect_suspicious_round_numbers(df_trans: pd.DataFrame, col: str = 'Total_Bill_Value', divisor: int = 1000) -> pd.DataFrame:
    """Returns transactions with perfectly round numbers (e.g., multiples of 1000)."""
    return df_trans[df_trans[col] % divisor == 0]

def detect_monthly_ceiling_patterns(df_trans: pd.DataFrame, col: str = 'Total_Bill_Value') -> pd.DataFrame:
    """Returns distributors whose monthly sales never exceed a certain fixed value (potential ceiling)."""
    # This is a variation of delivery caps. We can check if max monthly sales is repeated.
    monthly_dist = df_trans.groupby(['Distributor_ID', 'Year', 'Month'])[col].sum().reset_index()
    dist_max = monthly_dist.groupby('Distributor_ID')[col].max().reset_index()
    # Check if this max is hit exactly multiple times
    ceiling_hits = monthly_dist.merge(dist_max, on=['Distributor_ID', col])
    counts = ceiling_hits['Distributor_ID'].value_counts()
    suspicious_ids = counts[counts > 1].index
    return monthly_dist[monthly_dist['Distributor_ID'].isin(suspicious_ids)]
