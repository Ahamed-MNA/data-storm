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
        choices=['optimize', 'all'],
        help='Which pipeline step to run (default: optimize)',
    )
    args = parser.parse_args()

    try:
        step_optimize()
        logger.info("Pipeline step 'optimize' completed successfully [OK].")
    except Exception as exc:
        logger.error(f"Pipeline step 'optimize' FAILED: {exc}")
        raise


if __name__ == '__main__':
    main()
