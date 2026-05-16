"""
Bronze Layer: Raw Ingestion
===========================
Ingests all flat files AS-IS into data/bronze/ with zero transformations.
This layer preserves the original data exactly as provided by the source systems.
"""
import os
import sys
import shutil
import logging

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir   = os.path.abspath(os.path.join(script_dir, '..', '..'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Source → Bronze mapping ───────────────────────────────────────────────────
# Place raw CSVs in `data/raw/` (or adjust SOURCE_DIR to wherever you keep them)
SOURCE_DIR  = os.path.join(root_dir, 'data', 'raw')
BRONZE_DIR  = os.path.join(root_dir, 'data', 'bronze')

EXPECTED_FILES = [
    'transactions_history_final.csv',
    'outlet_master.csv',
    'outlet_coordinates.csv',
    'distributor_seasonality_details.csv',
    'holiday_list.csv',
]


def ingest_bronze(source_dir: str = SOURCE_DIR, bronze_dir: str = BRONZE_DIR) -> None:
    """
    Copy raw flat files from *source_dir* into *bronze_dir* without any
    modification.  If a file already exists in bronze, it is skipped
    (idempotent run behaviour).

    Parameters
    ----------
    source_dir : str
        Directory that holds the original CSV exports.
    bronze_dir : str
        Target bronze layer directory.
    """
    os.makedirs(bronze_dir, exist_ok=True)
    logger.info("Starting Bronze ingestion...")
    logger.info(f"  Source : {source_dir}")
    logger.info(f"  Target : {bronze_dir}")

    missing = []
    for filename in EXPECTED_FILES:
        src  = os.path.join(source_dir, filename)
        dest = os.path.join(bronze_dir, filename)

        if not os.path.exists(src):
            logger.warning(f"  MISSING source file: {filename} — skipping.")
            missing.append(filename)
            continue

        if os.path.exists(dest):
            logger.info(f"  SKIP  (already exists): {filename}")
        else:
            shutil.copy2(src, dest)
            logger.info(f"  COPIED: {filename}")

    if missing:
        logger.warning(
            f"\n{len(missing)} expected file(s) not found in source directory:\n"
            + "\n".join(f"  - {f}" for f in missing)
            + f"\n\nIf the raw files are already in '{bronze_dir}', you can skip "
              "this step and proceed directly to the Silver layer."
        )
    else:
        logger.info("Bronze ingestion complete — all files present.")


if __name__ == '__main__':
    # Allow overriding source dir from CLI:  python ingest_bronze.py <source_dir>
    src = sys.argv[1] if len(sys.argv) > 1 else SOURCE_DIR
    ingest_bronze(source_dir=src)
