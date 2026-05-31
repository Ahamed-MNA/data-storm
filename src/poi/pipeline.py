import os
import logging
from poi.competitor_graph import generate_competitor_graph
from poi.poi_decay import generate_poi_decay_edges

logger = logging.getLogger(__name__)

def run_poi_and_competitor_pipeline(use_legacy_coords: bool = True):
    """
    Main pipeline orchestrator for Spatial Distance-Decay POI modeling 
    and Competitive Catchment Density graph generation.
    
    Parameters:
    -----------
    use_legacy_coords : bool, default True
        If True, runs using the uncorrected coordinates from the bronze stage 
        to reproduce the legacy Colab outputs exactly.
        If False, runs using the corrected coordinates from dim_outlets.parquet.
    """
    logger.info("=" * 60)
    logger.info("STARTING POI & COMPETITIVE CATCHMENT PIPELINE")
    logger.info("=" * 60)
    
    # Locate project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # Define standardized file paths
    pbf_file_path = os.path.join(root_dir, 'data', 'bronze', 'sri-lanka-260529.osm.pbf')
    outlets_silver_path = os.path.join(root_dir, 'data', 'silver', 'dim_outlets.parquet')
    coordinates_bronze_path = os.path.join(root_dir, 'data', 'bronze', 'outlet_coordinates.csv')
    
    competitor_graph_output_path = os.path.join(root_dir, 'data', 'silver', 'silver_layer_competitor_graph.parquet')
    poi_edges_output_path = os.path.join(root_dir, 'data', 'silver', 'silver_layer_poi_edges.parquet')

    logger.info(f"Target Coordinate Mode: {'LEGACY (Uncorrected)' if use_legacy_coords else 'CORRECTED (Silver)'}")
    
    # 1. Run Competitor Graph pipeline
    try:
        generate_competitor_graph(
            outlets_silver_path=outlets_silver_path,
            coordinates_bronze_path=coordinates_bronze_path,
            output_parquet_path=competitor_graph_output_path,
            use_legacy_coords=use_legacy_coords
        )
    except Exception as e:
        logger.error(f"Failed to generate competitor graph: {e}", exc_info=True)
        raise

    # 2. Run POI Decay Edges pipeline
    try:
        generate_poi_decay_edges(
            pbf_file_path=pbf_file_path,
            outlets_silver_path=outlets_silver_path,
            coordinates_bronze_path=coordinates_bronze_path,
            output_parquet_path=poi_edges_output_path,
            use_legacy_coords=use_legacy_coords
        )
    except Exception as e:
        logger.error(f"Failed to generate POI decay edges: {e}", exc_info=True)
        raise

    logger.info("=" * 60)
    logger.info("POI & COMPETITIVE CATCHMENT PIPELINE COMPLETED SUCCESSFULY")
    logger.info("=" * 60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Run in legacy mode by default for testing reproducibility
    run_poi_and_competitor_pipeline(use_legacy_coords=True)
