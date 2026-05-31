import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up paths to import other local modules
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from x_ai.engine import get_engine_data, compute_xai_metrics
from x_ai.llm_service import generate_explanation
from x_ai.schemas import OutletXAIResponse

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_outlet_explanation(
    outlet_id: str,
    model_path: str = "outputs/sfa_model.pkl",
    data_path: str = "data/gold/sfa_refined.parquet"
) -> OutletXAIResponse:
    """Orchestrates loading data, running the SFA engine, and generating the LLM explanation."""
    logger.info(f"Step 1: Loading model and features for outlet: {outlet_id}")
    model, outlet_row = get_engine_data(
        model_path=PROJECT_ROOT / model_path,
        data_path=PROJECT_ROOT / data_path,
        outlet_id=outlet_id
    )
    
    logger.info("Step 2: Computing local SFA drivers and constraints")
    payload = compute_xai_metrics(model, outlet_row)
    
    logger.info("Step 3: Translating SFA metrics to business narrative via LLM")
    explanation = generate_explanation(payload)
    
    response = OutletXAIResponse(
        outlet_id=payload.outlet_id,
        actual_volume=payload.actual_volume,
        predicted_potential=payload.predicted_potential,
        opportunity_gap=payload.opportunity_gap,
        efficiency_score=payload.efficiency_score,
        inefficiency_pct=payload.inefficiency_pct,
        explanation=explanation,
        payload=payload
    )
    return response

def main():
    # Load dotenv from project root
    load_dotenv(PROJECT_ROOT / ".env")
    
    parser = argparse.ArgumentParser(description="Functional XAI Generator for SFA Model")
    parser.add_argument("--outlet-id", type=str, required=True, help="Outlet ID to explain (e.g. OUT_00001)")
    parser.add_argument("--model-path", type=str, default="outputs/sfa_model.pkl", help="Path to SFA model pickle")
    parser.add_argument("--data-path", type=str, default="data/gold/sfa_refined.parquet", help="Path to gold dataset parquet")
    parser.add_argument("--output-json", action="store_true", help="Print response as raw JSON instead of text report")
    
    args = parser.parse_args()
    
    try:
        response = generate_outlet_explanation(
            outlet_id=args.outlet_id,
            model_path=args.model_path,
            data_path=args.data_path
        )
        
        if args.output_json:
            print(response.model_dump_json(indent=2))
        else:
            print("\n" + "=" * 60)
            print(f"EXPLAINABLE AI REPORT FOR OUTLET: {response.outlet_id}")
            print("=" * 60)
            print(f"Historical Actual Volume: {response.actual_volume:.2f} L")
            print(f"Predicted Latent Ceiling: {response.predicted_potential:.2f} L")
            print(f"True Opportunity Gap:    {response.opportunity_gap:.2f} L")
            print(f"Technical Efficiency:     {response.efficiency_score * 100:.1f}%")
            print(f"Inefficiency Penalty:     {response.inefficiency_pct:.1f}%")
            print("-" * 60)
            print("BUSINESS INTERPRETATION NARRATIVE:")
            print("-" * 60)
            print(response.explanation)
            print("=" * 60 + "\n")
            
    except Exception as e:
        logger.error(f"XAI Generation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
