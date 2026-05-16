import pandas as pd
import logging

logger = logging.getLogger(__name__)

def check_nulls(df: pd.DataFrame, columns: list) -> dict:
    """
    Checks for null values in specified columns.
    Returns a dictionary with column names as keys and the count of nulls as values.
    """
    null_counts = df[columns].isnull().sum().to_dict()
    for col, count in null_counts.items():
        if count > 0:
            logger.warning(f"Column '{col}' has {count} null values.")
    return null_counts

def check_duplicates(df: pd.DataFrame, keys: list) -> int:
    """
    Checks for duplicate rows based on specified keys.
    Returns the number of duplicate rows found.
    """
    duplicate_count = df.duplicated(subset=keys).sum()
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate rows based on keys {keys}.")
    return duplicate_count

def check_ranges(df: pd.DataFrame, column: str, min_val: float, max_val: float) -> pd.DataFrame:
    """
    Checks if values in a column are within the specified [min_val, max_val] range.
    Returns a DataFrame containing rows that fall outside the range.
    """
    out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
    if not out_of_range.empty:
        logger.warning(f"Column '{column}' has {len(out_of_range)} values outside the range [{min_val}, {max_val}].")
    return out_of_range

def check_referential_integrity(df_child: pd.DataFrame, df_parent: pd.DataFrame, child_key: str, parent_key: str) -> pd.Series:
    """
    Checks if all values in df_child[child_key] exist in df_parent[parent_key].
    Returns the set of values in the child table that are missing from the parent table.
    """
    missing_keys = df_child[~df_child[child_key].isin(df_parent[parent_key])][child_key].unique()
    if len(missing_keys) > 0:
        logger.warning(f"Referential integrity violation: {len(missing_keys)} unique keys in '{child_key}' not found in parent.")
    return missing_keys

def check_datatypes(df: pd.DataFrame, schema: dict) -> dict:
    """
    Checks if columns in the DataFrame match the expected data types in the schema.
    Schema format: {'column_name': 'dtype'} (e.g., {'age': 'int64', 'name': 'object'})
    Returns a dictionary of columns with mismatches: {column: (expected, actual)}
    """
    mismatches = {}
    for col, expected_dtype in schema.items():
        if col not in df.columns:
            mismatches[col] = (expected_dtype, "Missing Column")
            continue
        
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            mismatches[col] = (expected_dtype, actual_dtype)
            logger.warning(f"Data type mismatch in '{col}': Expected {expected_dtype}, got {actual_dtype}.")
            
    return mismatches
