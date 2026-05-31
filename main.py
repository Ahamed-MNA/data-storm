"""
DataStorm — Main Pipeline Orchestrator (Round 2)
==============================================
Runs the marketing spend optimization task.

Usage:
  uv run python main.py [--step STEP]

STEP options:
  optimize    Run the marketing spend optimization (default)
  all         Run the marketing spend optimization
"""
import os
import sys
import logging
import argparse

# Path setup
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(root_dir, 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def step_poi(use_legacy_coords: bool):
    """Run POI and Competitor Graph pipeline."""
    from poi.pipeline import run_poi_and_competitor_pipeline
    run_poi_and_competitor_pipeline(use_legacy_coords=use_legacy_coords)


def step_features():
    """Run Silver to Gold feature engineering pipeline."""
    from features.build_gold_sfa import build_refined_gold_layer
    logger.info("═" * 60)
    logger.info("RUNNING FEATURE ENGINEERING PIPELINE (SILVER -> GOLD)")
    logger.info("═" * 60)
    build_refined_gold_layer()


def step_optimize():
    """Run marketing spend optimization."""
    from optimization.market_spend_optim import run_marketing_spend_optimization
    logger.info("═" * 60)
    logger.info("RUNNING MARKETING SPEND OPTIMIZATION")
    logger.info("═" * 60)
    run_marketing_spend_optimization()


def main():
    parser = argparse.ArgumentParser(
        description='DataStorm Round 2 - Spend Optimization Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--step',
        default='optimize',
        choices=['optimize', 'poi', 'features', 'all'],
        help='Which pipeline step to run (default: optimize)',
    )
    parser.add_argument(
        '--corrected-coords',
        action='store_true',
        help='Use corrected coordinates from dim_outlets.parquet instead of uncorrected bronze coordinates',
    )
    args = parser.parse_args()

    use_legacy_coords = not args.corrected_coords

    try:
        if args.step == 'poi':
            step_poi(use_legacy_coords)
            logger.info("Pipeline step 'poi' completed successfully [OK].")
        elif args.step == 'features':
            step_features()
            logger.info("Pipeline step 'features' completed successfully [OK].")
        elif args.step == 'optimize':
            step_optimize()
            logger.info("Pipeline step 'optimize' completed successfully [OK].")
        elif args.step == 'all':
            step_poi(use_legacy_coords)
            step_features()
            step_optimize()
            logger.info("All pipeline steps completed successfully [OK].")
    except Exception as exc:
        logger.error(f"Pipeline step execution FAILED: {exc}")
        raise


if __name__ == '__main__':
    main()
