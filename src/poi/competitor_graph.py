import os
import gc
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

def generate_competitor_graph(
    outlets_silver_path: str,
    coordinates_bronze_path: str,
    output_parquet_path: str,
    use_legacy_coords: bool = True,
    radius: float = 5000.0,
    sigma: float = 1000.0,
    chunk_size: int = 500
):
    """
    Generates the competitive catchment density graph (undirected competitor network).
    
    Parameters:
    -----------
    outlets_silver_path : str
        Path to the cleaned outlets silver layer parquet file.
    coordinates_bronze_path : str
        Path to the raw bronze coordinates CSV file (needed for legacy coords fallback).
    output_parquet_path : str
        Path to save the generated competitor graph parquet file.
    use_legacy_coords : bool, default True
        If True, uses the uncorrected coordinates from bronze (matches legacy Colab run).
        If False, uses the corrected coordinates from dim_outlets.parquet.
    radius : float, default 5000.0
        Search radius in meters for competitors.
    sigma : float, default 1000.0
        Sigma parameter for the exponential distance decay friction weight.
    chunk_size : int, default 500
        Chunk size for KDTree querying to save memory.
    """
    logger.info("Starting competitor graph generation...")
    
    # 1. Load silver outlets to get the target set of 19,960 valid outlets
    if not os.path.exists(outlets_silver_path):
        raise FileNotFoundError(f"Cleaned outlets parquet not found at {outlets_silver_path}")
        
    outlets_df = pd.read_parquet(outlets_silver_path)
    logger.info(f"Loaded {len(outlets_df)} outlets from silver layer.")
    
    # 2. Select Coordinate Set
    if use_legacy_coords:
        logger.warning("Using LEGACY (uncorrected) coordinates from bronze to match teammate's Colab run.")
        if not os.path.exists(coordinates_bronze_path):
            raise FileNotFoundError(f"Bronze coordinates CSV not found at {coordinates_bronze_path}")
            
        bronze_coords_df = pd.read_csv(coordinates_bronze_path)
        # Drop columns other than coordinates to merge cleanly
        bronze_coords_df = bronze_coords_df[['Outlet_ID', 'Latitude', 'Longitude']]
        
        # Merge to keep only the 19,960 valid outlets but with legacy coordinates
        outlets_df = outlets_df[['Outlet_ID']].merge(bronze_coords_df, on='Outlet_ID', how='inner')
        logger.info(f"Merged coordinates from bronze. Remaining outlets: {len(outlets_df)}")
    else:
        logger.info("Using CORRECTED coordinates from silver layer (dim_outlets.parquet).")
        outlets_df = outlets_df[['Outlet_ID', 'Latitude', 'Longitude']]

    # 3. Project to UTM Zone 44N (EPSG:32644) for metric distance calculations
    outlets_gdf = gpd.GeoDataFrame(
        outlets_df, 
        geometry=gpd.points_from_xy(outlets_df['Longitude'], outlets_df['Latitude']), 
        crs="EPSG:4326"
    ).to_crs("EPSG:32644")

    # Extract coordinates into numpy array
    coords = np.array(list(zip(outlets_gdf.geometry.x, outlets_gdf.geometry.y)))
    outlet_ids = outlets_df['Outlet_ID'].astype(str).values

    del outlets_df, outlets_gdf
    gc.collect()

    # 4. Build Self-Referencing KD-Tree
    logger.info("Building cKDTree for outlet coordinates...")
    tree = cKDTree(coords)

    edges = []
    total_edges = 0

    # 5. Query KD-Tree in chunks to manage memory footprint
    logger.info(f"Mapping competitor network with radius={radius}m, sigma={sigma}...")
    for start_idx in range(0, len(outlet_ids), chunk_size):
        end_idx = min(start_idx + chunk_size, len(outlet_ids))
        chunk_coords = coords[start_idx:end_idx]
        chunk_ids = outlet_ids[start_idx:end_idx]

        # Query KD-Tree for all outlets within 5km radius
        competitor_indices_list = tree.query_ball_point(chunk_coords, r=radius)

        for i, comp_indices in enumerate(competitor_indices_list):
            source_id = chunk_ids[i]
            source_pt = chunk_coords[i]

            comp_pts = coords[comp_indices]
            target_ids = outlet_ids[comp_indices]

            # Vectorized distance and friction weights calculation
            distances = np.linalg.norm(comp_pts - source_pt, axis=1)
            friction_weights = np.exp(-(distances**2) / (2 * sigma**2))

            for j in range(len(comp_indices)):
                target_id = target_ids[j]

                # CRITICAL ARCHITECTURAL OPTIMIZATION:
                # To prevent bidirectional duplicates (A -> B and B -> A) and self-matches (A -> A),
                # only record edges where the Source ID is alphabetically smaller than Target ID.
                if source_id < target_id:
                    edges.append({
                        'Source_Outlet_ID': source_id,
                        'Target_Competitor_ID': target_id,
                        'Distance_Meters': round(distances[j], 2),
                        'Competitive_Friction': round(friction_weights[j], 4)
                    })

        total_edges += len(edges)
        logger.debug(f"Processed chunk {start_idx}-{end_idx}. Total edges collected so far: {len(edges)}")

    # 6. Convert to DataFrame and Save as Parquet
    if len(edges) == 0:
        logger.warning("No competitive edges found! Writing an empty DataFrame.")
        result_df = pd.DataFrame(columns=['Source_Outlet_ID', 'Target_Competitor_ID', 'Distance_Meters', 'Competitive_Friction'])
    else:
        result_df = pd.DataFrame(edges)

    # Ensure target folder exists
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    
    # Save as parquet format using pyarrow engine
    result_df.to_parquet(output_parquet_path, engine='pyarrow', index=False)
    logger.info(f"Saved competitor graph with {len(result_df)} edges to {output_parquet_path}")
    
    del edges, result_df
    gc.collect()
