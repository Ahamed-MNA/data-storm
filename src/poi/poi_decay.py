import os
import gc
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

def generate_poi_decay_edges(
    pbf_file_path: str,
    outlets_silver_path: str,
    coordinates_bronze_path: str,
    output_parquet_path: str,
    use_legacy_coords: bool = True,
    radius_macro: float = 3000.0,
    radius_micro: float = 1000.0,
    sigma: float = 300.0,
    chunk_size: int = 100
):
    """
    Extracts POIs from OSM PBF and computes spatial distance-decay edges for each outlet.
    
    Parameters:
    -----------
    pbf_file_path : str
        Path to the Sri Lanka osm.pbf file.
    outlets_silver_path : str
        Path to the cleaned outlets silver layer parquet file.
    coordinates_bronze_path : str
        Path to the raw bronze coordinates CSV file (needed for legacy coords fallback).
    output_parquet_path : str
        Path to save the generated POI edges parquet file.
    use_legacy_coords : bool, default True
        If True, uses the uncorrected coordinates from bronze (matches legacy Colab run).
        If False, uses the corrected coordinates from dim_outlets.parquet.
    radius_macro : float, default 3000.0
        Search radius in meters for regional density multipliers.
    radius_micro : float, default 1000.0
        Search radius in meters for actual POI connections.
    sigma : float, default 300.0
        Sigma parameter for Gaussian distance decay.
    chunk_size : int, default 100
        Chunk size for query processing to conserve RAM.
    """
    logger.info("Starting POI distance-decay mapping pipeline...")

    # 1. Gracefully handle environment where pyrosm is missing
    try:
        from pyrosm import OSM
    except ModuleNotFoundError:
        logger.warning("Module 'pyrosm' is not installed (compilation is often unsupported on Windows).")
        if os.path.exists(output_parquet_path):
            logger.warning(f"Preserving existing POI edges output at: {output_parquet_path}")
            return
        else:
            raise ImportError(
                "Module 'pyrosm' is required to generate POI edges from scratch, but it is not installed. "
                f"Additionally, no existing output was found at {output_parquet_path}."
            )

    # 2. Extract POIs from OSM File
    if not os.path.exists(pbf_file_path):
        raise FileNotFoundError(f"OSM PBF file not found at: {pbf_file_path}")

    logger.info(f"Loading OSM Data from PBF file: {pbf_file_path}...")
    osm = OSM(pbf_file_path)

    # Define custom tags filter matching our demand drivers
    custom_filter = {
        'amenity': ['school', 'university', 'college', 'hospital', 'clinic', 'bus_station', 'marketplace', 'place_of_worship'],
        'highway': ['bus_stop'], 
        'railway': ['station', 'halt'],
        'tourism': ['attraction', 'museum', 'hotel', 'guest_house'],
        'building': ['apartments'], 
        'landuse': ['industrial']
    }
    
    logger.info("Extracting POIs based on custom criteria...")
    pois = osm.get_data_by_custom_criteria(
        custom_filter=custom_filter, 
        keep_nodes=True, 
        keep_ways=True, 
        keep_relations=False
    )

    if pois is None or len(pois) == 0:
        raise ValueError("No POIs extracted from OSM file. Check filter tags or PBF contents.")

    # Helper function to categorize POIs
    def extract_poi_type(row):
        for col in ['amenity', 'highway', 'railway', 'tourism', 'building', 'landuse']:
            if col in pois.columns and pd.notna(row[col]): 
                return f"{row[col]}"
        return "unknown"

    logger.info("Processing POI attributes and geometries...")
    pois['POI_Type'] = pois.apply(extract_poi_type, axis=1)
    pois['POI_ID'] = [f"POI_{str(i).zfill(6)}" for i in range(len(pois))]
    
    # Drop rows without geometry and take centroids
    pois = pois.dropna(subset=['geometry']).copy()
    pois['geometry'] = pois['geometry'].centroid
    
    # Project to metric CRS
    pois_gdf = pois.to_crs("EPSG:32644")

    # Define Base Footfall weights
    base_footfall_mapping = {
        'station': 25000, 'bus_station': 10000, 'hospital': 5000, 'university': 4000,
        'marketplace': 3000, 'college': 2000, 'school': 1000, 'apartments': 500,
        'place_of_worship': 300, 'clinic': 200, 'bus_stop': 150, 'hotel': 100,
        'guest_house': 50, 'industrial': 1000
    }
    pois_gdf['Base_Footfall'] = pois_gdf['POI_Type'].map(base_footfall_mapping).fillna(100.0)

    poi_coords = np.array(list(zip(pois_gdf.geometry.x, pois_gdf.geometry.y)))
    poi_ids = pois_gdf['POI_ID'].values
    poi_types = pois_gdf['POI_Type'].values
    poi_footfalls = pois_gdf['Base_Footfall'].values

    # Clean up large geodataframe objects to prevent memory exhaustion
    del osm, pois, pois_gdf
    gc.collect()

    # 3. Build KD-Tree for POIs
    logger.info("Building cKDTree for POIs...")
    tree = cKDTree(poi_coords)

    # 4. Load outlets and select coordinates
    if not os.path.exists(outlets_silver_path):
        raise FileNotFoundError(f"Cleaned outlets parquet not found at {outlets_silver_path}")
        
    outlets_df = pd.read_parquet(outlets_silver_path)
    logger.info(f"Loaded {len(outlets_df)} outlets from silver layer.")
    
    if use_legacy_coords:
        logger.warning("Using LEGACY (uncorrected) coordinates from bronze to match teammate's Colab run.")
        if not os.path.exists(coordinates_bronze_path):
            raise FileNotFoundError(f"Bronze coordinates CSV not found at {coordinates_bronze_path}")
            
        bronze_coords_df = pd.read_csv(coordinates_bronze_path)
        bronze_coords_df = bronze_coords_df[['Outlet_ID', 'Latitude', 'Longitude']]
        
        # Keep only the valid Outlet_IDs but merge in the legacy coordinates
        outlets_df = outlets_df[['Outlet_ID']].merge(bronze_coords_df, on='Outlet_ID', how='inner')
    else:
        logger.info("Using CORRECTED coordinates from silver layer (dim_outlets.parquet).")
        outlets_df = outlets_df[['Outlet_ID', 'Latitude', 'Longitude']]

    outlets_gdf = gpd.GeoDataFrame(
        outlets_df, 
        geometry=gpd.points_from_xy(outlets_df['Longitude'], outlets_df['Latitude']), 
        crs="EPSG:4326"
    ).to_crs("EPSG:32644")

    outlet_coords = np.array(list(zip(outlets_gdf.geometry.x, outlets_gdf.geometry.y)))
    outlet_ids = outlets_df['Outlet_ID'].values
    
    del outlets_df, outlets_gdf
    gc.collect()

    edges = []
    total_edges = 0

    # 5. Process outlets in chunks and calculate multipliers + decay weights
    logger.info("Calculating POI distances and dynamic footfall impact multipliers...")
    for start_idx in range(0, len(outlet_ids), chunk_size):
        end_idx = min(start_idx + chunk_size, len(outlet_ids))
        chunk_outlets_coords = outlet_coords[start_idx:end_idx]
        chunk_outlet_ids = outlet_ids[start_idx:end_idx]

        # 1. Macro query (3km) to capture regional density multiplier
        macro_indices = tree.query_ball_point(chunk_outlets_coords, r=radius_macro)
        macro_counts = np.array([len(idx) for idx in macro_indices])
        
        # Multiply scale: 0.2x to 2.0x based on macro density counts
        density_multipliers = np.clip(macro_counts / 100.0, 0.2, 2.0)

        # 2. Micro query (1km) for actual physical decay connections
        micro_indices = tree.query_ball_point(chunk_outlets_coords, r=radius_micro)

        for i, nearby_poi_indices in enumerate(micro_indices):
            if len(nearby_poi_indices) == 0: 
                continue

            out_id = chunk_outlet_ids[i]
            out_pt = chunk_outlets_coords[i]
            local_multiplier = density_multipliers[i]

            nearby_coords = poi_coords[nearby_poi_indices]
            n_ids = poi_ids[nearby_poi_indices]
            n_types = poi_types[nearby_poi_indices]
            n_footfalls = poi_footfalls[nearby_poi_indices]

            # Euclidean distance in metric CRS
            distances = np.linalg.norm(nearby_coords - out_pt, axis=1)
            
            # Non-linear Gaussian Distance-Decay formula
            weights = np.exp(-(distances**2) / (2 * sigma**2))

            # Apply regional multiplier scale to calculate final impact score
            final_impacts = n_footfalls * local_multiplier * weights

            for j in range(len(nearby_poi_indices)):
                edges.append({
                    'Outlet_ID': out_id,
                    'POI_ID': n_ids[j],
                    'POI_Type': n_types[j],
                    'Distance_Meters': round(distances[j], 2),
                    'Base_Footfall': float(n_footfalls[j]),
                    'Regional_Multiplier': round(local_multiplier, 2),
                    'Decay_Weight': round(weights[j], 4),
                    'Footfall_Impact_Score': round(final_impacts[j], 2)
                })

        total_edges += len(edges)
        logger.debug(f"Processed chunk {start_idx}-{end_idx}. Total edges collected: {len(edges)}")
        gc.collect()

    # 6. Save final results as Parquet
    if len(edges) == 0:
        logger.warning("No POI edges found! Writing an empty DataFrame.")
        result_df = pd.DataFrame(columns=[
            'Outlet_ID', 'POI_ID', 'POI_Type', 'Distance_Meters', 
            'Base_Footfall', 'Regional_Multiplier', 'Decay_Weight', 'Footfall_Impact_Score'
        ])
    else:
        result_df = pd.DataFrame(edges)

    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    result_df.to_parquet(output_parquet_path, engine='pyarrow', index=False)
    logger.info(f"Saved {len(result_df)} POI edges to {output_parquet_path}")

    del edges, result_df
    gc.collect()
