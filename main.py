"""
DataStorm - Main Pipeline Orchestrator
=======================================
End-to-end pipeline runner for the DataStorm Round 2 challenge.

Stages (run in order for a full pipeline):
  1. ingest      - Copy raw CSVs from data/raw/ -> data/bronze/ (no transforms)
  2. clean       - Bronze CSVs -> cleaned Silver Parquet files
  3. poi         - Spatial POI distance-decay + competitor graph -> Silver Parquets
  4. features    - Silver -> Gold feature engineering (SFA-ready)
  5. model       - Train Stochastic Frontier Analysis (SFA) model on Gold layer
  6. predict     - Generate January 2026 latent-potential predictions (submission CSV)
  7. optimize    - Solve non-linear marketing spend optimization for Western Province
  8. xai         - Generate Explainable AI narrative for a specific outlet (requires --outlet-id)
  9. all         - Run stages 1-7 sequentially (excludes xai; use --step xai separately)

Usage examples:
  uv run python main.py                            # default: optimize
  uv run python main.py --step all                 # full pipeline (ingest -> optimize)
  uv run python main.py --step ingest
  uv run python main.py --step clean
  uv run python main.py --step poi
  uv run python main.py --step poi --corrected-coords
  uv run python main.py --step features
  uv run python main.py --step model
  uv run python main.py --step predict
  uv run python main.py --step optimize
  uv run python main.py --step xai --outlet-id OUT_00001
  uv run python main.py --step xai --outlet-id OUT_00001 --output-json
"""
from __future__ import annotations

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# ── Path Setup ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

# Load .env early so every module that reads env-vars benefits
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DIVIDER = "=" * 62


# ─────────────────────────────────────────────────────────────────────────────
# Stage functions
# ─────────────────────────────────────────────────────────────────────────────

def step_ingest() -> None:
    """Stage 1 - Raw CSV -> Bronze layer (zero-transform copy)."""
    from ingestion.ingest_bronze import ingest_bronze
    logger.info(DIVIDER)
    logger.info("STAGE 1 | INGEST   (raw -> bronze)")
    logger.info(DIVIDER)
    ingest_bronze()


def step_clean() -> None:
    """Stage 2 - Bronze -> Silver (cleaning, validation, Parquet output)."""
    from cleaning.bronze_to_silver import run_bronze_to_silver
    logger.info(DIVIDER)
    logger.info("STAGE 2 | CLEAN    (bronze -> silver)")
    logger.info(DIVIDER)
    run_bronze_to_silver()


def step_poi(use_legacy_coords: bool) -> None:
    """Stage 3 - Spatial POI decay + competitor graph -> Silver Parquets."""
    from poi.pipeline import run_poi_and_competitor_pipeline
    logger.info(DIVIDER)
    coord_mode = "LEGACY (bronze)" if use_legacy_coords else "CORRECTED (silver)"
    logger.info(f"STAGE 3 | POI      (spatial graph - coords: {coord_mode})")
    logger.info(DIVIDER)
    run_poi_and_competitor_pipeline(use_legacy_coords=use_legacy_coords)


def step_features() -> None:
    """Stage 4 - Silver -> Gold feature engineering (SFA-ready dataset)."""
    from features.build_gold_sfa import build_refined_gold_layer
    logger.info(DIVIDER)
    logger.info("STAGE 4 | FEATURES (silver -> gold)")
    logger.info(DIVIDER)
    build_refined_gold_layer()


def step_model() -> None:
    """Stage 5 - Train Stochastic Frontier Analysis model on Gold layer."""
    from modeling.train_and_report_potential import run_refined_sfa_pipeline
    logger.info(DIVIDER)
    logger.info("STAGE 5 | MODEL    (train SFA on gold layer)")
    logger.info(DIVIDER)
    run_refined_sfa_pipeline()


def step_predict() -> None:
    """Stage 6 - Generate January 2026 latent-potential predictions."""
    from modeling.generate_predictions import generate_predictions
    logger.info(DIVIDER)
    logger.info("STAGE 6 | PREDICT  (generate submission predictions)")
    logger.info(DIVIDER)
    generate_predictions()


def step_optimize() -> None:
    """Stage 7 - Solve non-linear marketing spend optimisation."""
    from optimization.market_spend_optim import run_marketing_spend_optimization
    logger.info(DIVIDER)
    logger.info("STAGE 7 | OPTIMIZE (marketing spend allocation)")
    logger.info(DIVIDER)
    run_marketing_spend_optimization()


def step_xai(outlet_id: str, output_json: bool = False) -> None:
    """Stage 8 - XAI narrative generation for a single outlet."""
    from x_ai.main import generate_outlet_explanation
    logger.info(DIVIDER)
    logger.info(f"STAGE 8 | XAI      (explain outlet: {outlet_id})")
    logger.info(DIVIDER)

    response = generate_outlet_explanation(outlet_id=outlet_id)

    if output_json:
        print(response.model_dump_json(indent=2))
    else:
        print("\n" + "=" * 62)
        print(f"  EXPLAINABLE AI REPORT  |  OUTLET: {response.outlet_id}")
        print("=" * 62)
        print(f"  Historical Actual Volume  : {response.actual_volume:>12.2f} L")
        print(f"  Predicted Latent Ceiling  : {response.predicted_potential:>12.2f} L")
        print(f"  True Opportunity Gap      : {response.opportunity_gap:>12.2f} L")
        print(f"  Technical Efficiency      : {response.efficiency_score * 100:>11.1f}%")
        print(f"  Inefficiency Penalty      : {response.inefficiency_pct:>11.1f}%")
        print("-" * 62)
        print("  BUSINESS INTERPRETATION NARRATIVE:")
        print("-" * 62)
        print(response.explanation)
        print("=" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="DataStorm Round 2 - End-to-End Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--step",
        default="optimize",
        choices=["ingest", "clean", "poi", "features", "model", "predict", "optimize", "xai", "all"],
        help=(
            "Pipeline stage to run (default: optimize). "
            "Use 'all' to run stages 1-7 sequentially."
        ),
    )
    parser.add_argument(
        "--corrected-coords",
        action="store_true",
        help=(
            "POI stage only: use corrected coordinates from dim_outlets.parquet "
            "(silver layer) instead of the raw bronze coordinates."
        ),
    )
    parser.add_argument(
        "--outlet-id",
        type=str,
        default=None,
        help="XAI stage only: Outlet ID to explain (e.g. OUT_00001).",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="XAI stage only: print the full response as raw JSON instead of a formatted report.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    use_legacy_coords = not args.corrected_coords

    try:
        if args.step == "ingest":
            step_ingest()

        elif args.step == "clean":
            step_clean()

        elif args.step == "poi":
            step_poi(use_legacy_coords)

        elif args.step == "features":
            step_features()

        elif args.step == "model":
            step_model()

        elif args.step == "predict":
            step_predict()

        elif args.step == "optimize":
            step_optimize()

        elif args.step == "xai":
            if not args.outlet_id:
                parser.error("--outlet-id is required when --step xai is used.")
            step_xai(outlet_id=args.outlet_id, output_json=args.output_json)

        elif args.step == "all":
            # Full pipeline — stages 1-7 (XAI is outlet-specific; run separately)
            step_ingest()
            step_clean()
            step_poi(use_legacy_coords)
            step_features()
            step_model()
            step_predict()
            step_optimize()

        logger.info(DIVIDER)
        logger.info(f"Pipeline step '{args.step}' completed successfully [OK]")
        logger.info(DIVIDER)

    except Exception as exc:
        logger.error(f"Pipeline step '{args.step}' FAILED: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
