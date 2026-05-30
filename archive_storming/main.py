"""
DataStorm Pipeline Orchestrator
================================
Runs the full Bronze -> Silver -> Gold -> Model -> Predictions pipeline end-to-end.

Usage
-----
    python main.py [--step STEP]

    STEP options:
        all           Run the entire pipeline (default)
        bronze        Bronze ingestion only
        silver        Bronze -> Silver cleaning only
        validate      Silver validation + anomaly report
        gold          Silver -> Gold feature engineering only
        model         Gold -> SFA model training only
        predict       SFA report -> competition predictions CSV only

Examples
--------
    # Run everything from scratch
    python main.py

    # Re-run just the model after changing SFA parameters
    python main.py --step model

    # Generate a fresh predictions CSV without re-training
    python main.py --step predict
"""
import os
import sys
import logging
import argparse
import time

# ── Path setup ────────────────────────────────────────────────────────────────
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(root_dir, 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def step_bronze():
    """Bronze: Copy raw CSVs → data/bronze (no transformations)."""
    from ingestion.ingest_bronze import ingest_bronze
    logger.info("═" * 60)
    logger.info("STEP 1 — BRONZE  (Raw Ingestion)")
    logger.info("═" * 60)
    # If data is already in data/bronze/, this is a no-op.
    bronze_dir = os.path.join(root_dir, 'data', 'bronze')
    ingest_bronze(bronze_dir=bronze_dir)


def step_silver():
    """Silver: Clean + quarantine rejected records."""
    from cleaning.bronze_to_silver import run_bronze_to_silver
    logger.info("═" * 60)
    logger.info("STEP 2 — SILVER  (Cleaning & Quality Checks)")
    logger.info("═" * 60)
    run_bronze_to_silver()


def step_validate():
    """Validate Silver layer data quality and run anomaly report."""
    from validation.validate_silver import validate_silver_layer
    from validation.run_anomaly_detection import run_anomaly_report
    logger.info("═" * 60)
    logger.info("STEP 3 — VALIDATE  (Silver QC + Anomaly Detection)")
    logger.info("═" * 60)
    validate_silver_layer()
    run_anomaly_report()


def step_gold():
    """Gold: Feature engineering → SFA-ready parquet."""
    from features.build_gold_sfa import build_refined_gold_layer
    logger.info("═" * 60)
    logger.info("STEP 4 — GOLD  (Feature Engineering)")
    logger.info("═" * 60)
    build_refined_gold_layer()


def step_model():
    """Train SFA model and save outlet potential report."""
    from modeling.train_and_report_potential import run_refined_sfa_pipeline
    logger.info("═" * 60)
    logger.info("STEP 5 — MODEL  (Stochastic Frontier Analysis)")
    logger.info("═" * 60)
    run_refined_sfa_pipeline()


def step_predict():
    """Generate the final competition predictions CSV."""
    from modeling.generate_predictions import generate_predictions
    logger.info("═" * 60)
    logger.info("STEP 6 — PREDICT  (Generate Submission CSV)")
    logger.info("═" * 60)
    generate_predictions()


# ── Pipeline registry ─────────────────────────────────────────────────────────
STEPS = {
    'bronze':   step_bronze,
    'silver':   step_silver,
    'validate': step_validate,
    'gold':     step_gold,
    'model':    step_model,
    'predict':  step_predict,
}

FULL_PIPELINE = ['bronze', 'silver', 'validate', 'gold', 'model', 'predict']


def main():
    parser = argparse.ArgumentParser(
        description='DataStorm Latent Potential Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--step',
        default='all',
        choices=['all'] + FULL_PIPELINE,
        help='Which pipeline step to run (default: all)',
    )
    args = parser.parse_args()

    steps_to_run = FULL_PIPELINE if args.step == 'all' else [args.step]

    total_start = time.time()
    for step_name in steps_to_run:
        t0 = time.time()
        try:
            STEPS[step_name]()
            elapsed = time.time() - t0
            logger.info(f"  [OK] {step_name.upper()} completed in {elapsed:.1f}s\n")
        except Exception as exc:
            logger.error(f"  [FAIL] {step_name.upper()} FAILED: {exc}")
            raise

    total = time.time() - total_start
    logger.info("═" * 60)
    logger.info(f"Pipeline finished in {total:.1f}s")
    logger.info("═" * 60)


if __name__ == '__main__':
    main()
