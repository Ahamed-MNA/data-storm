import pandas as pd
import numpy as np
import os

def clean_transactions(df: pd.DataFrame, outlet_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans transaction data.
    Returns (cleaned_df, rejected_df).
    """
    rejected_records = []
    
    # 1. Duplicate Invoices
    # Define keys for duplicates (assuming SKU-level transactions per month)
    keys = ['Outlet_ID', 'Year', 'Month', 'Distributor_ID', 'SKU_ID', 'Volume_Liters', 'Total_Bill_Value']
    dup_mask = df.duplicated(subset=keys, keep='first')
    duplicates = df[dup_mask].copy()
    duplicates['failure_reason'] = 'Duplicate Transaction'
    rejected_records.append(duplicates)
    df = df[~dup_mask]
    
    # 3. Referential Integrity (Outlet ID)
    valid_outlets = outlet_master['Outlet_ID'].unique()
    invalid_outlet_mask = ~df['Outlet_ID'].isin(valid_outlets)
    invalid_outlets = df[invalid_outlet_mask].copy()
    invalid_outlets['failure_reason'] = 'Invalid Outlet_ID'
    rejected_records.append(invalid_outlets)
    df = df[~invalid_outlet_mask]
    
    # Combine rejected
    if rejected_records:
        rejected_df = pd.concat(rejected_records, ignore_index=True)
    else:
        rejected_df = pd.DataFrame(columns=df.columns.tolist() + ['failure_reason'])
        
    return df, rejected_df

def clean_outlets(df_master: pd.DataFrame, df_coords: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans outlet master and coordinate data (merged).
    Returns (cleaned_df, rejected_df).
    """
    rejected_records = []
    
    # Merge master and coords to validate together
    merged = df_master.merge(df_coords, on='Outlet_ID', how='left')
    
    # 1. Missing Coordinates
    missing_coord_mask = merged['Latitude'].isnull() | merged['Longitude'].isnull()
    missing_coords = merged[missing_coord_mask].copy()
    missing_coords['failure_reason'] = 'Missing GPS Coordinates'
    rejected_records.append(missing_coords)
    merged = merged[~missing_coord_mask]
    
    # 2. Outlets in Ocean / Swapped Coordinates
    LAT_RANGE = (5.9, 9.9)
    LON_RANGE = (79.6, 81.9)
    
    def fix_swapped_coords(row):
        lat, lon = row['Latitude'], row['Longitude']
        
        # Check if current is valid
        current_valid = (LAT_RANGE[0] <= lat <= LAT_RANGE[1]) and (LON_RANGE[0] <= lon <= LON_RANGE[1])
        if current_valid:
            return lat, lon, False, None
            
        # Check if swapped is valid
        swapped_valid = (LAT_RANGE[0] <= lon <= LAT_RANGE[1]) and (LON_RANGE[0] <= lat <= LON_RANGE[1])
        if swapped_valid:
            return lon, lat, False, None # Swap them
            
        # Both invalid
        return lat, lon, True, 'Coordinates Outside Sri Lanka Bounds'

    res = merged.apply(fix_swapped_coords, axis=1, result_type='expand')
    merged['Latitude'] = res[0]
    merged['Longitude'] = res[1]
    ocean_mask = res[2]
    
    ocean_outlets = merged[ocean_mask].copy()
    ocean_outlets['failure_reason'] = res[3]
    rejected_records.append(ocean_outlets)
    merged = merged[~ocean_mask]
    
    # 3. Duplicate GPS
    dup_gps_mask = merged.duplicated(subset=['Latitude', 'Longitude'], keep='first')
    dup_gps = merged[dup_gps_mask].copy()
    dup_gps['failure_reason'] = 'Duplicate GPS Location'
    rejected_records.append(dup_gps)
    merged = merged[~dup_gps_mask]
    
    # 4. Fill Missing Values
    merged['Outlet_Size'] = merged['Outlet_Size'].fillna('Unknown')
    
    if rejected_records:
        rejected_df = pd.concat(rejected_records, ignore_index=True)
    else:
        rejected_df = pd.DataFrame(columns=merged.columns.tolist() + ['failure_reason'])
        
    return merged, rejected_df
