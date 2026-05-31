import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from .schemas import OutletXAIPayload

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert FMCG (Fast-Moving Consumer Goods) commercial consultant and data translator.
Your task is to analyze the provided JSON payload containing a retail outlet's sales potential metrics, physical attributes, geospatial signals, and operational constraints, and translate them into a clear, human-readable business narrative.

Strictly adhere to the following Structural Guidelines:
- Write exactly a 3-paragraph narrative. Do not include titles, labels, or greetings (such as "Dear Regional Sales Manager" or "Paragraph 1:"). Start directly with the narrative.
- Paragraph 1 (The Score): Explain the latent ceiling (predicted maximum potential) vs. historical reality (actual baseline sales). Frame this opportunity gap as a clear investment/revenue opportunity. If the actual sales exceed the predicted potential (resulting in a technical efficiency score greater than 100% / 1.0), explain that the outlet is an extraordinary success story and outlier that outperforms its model-predicted frontier.
- Paragraph 2 (The Drivers): Explain what environmental or local factors naturally elevate this store (e.g., presence of commercial centers, educational institutions, or residential density based on the geospatial and POI signals).
- Paragraph 3 (The Bottlenecks & Action): Explain how operational bottlenecks (such as lack of cooler space, high flatline score, or demand volatility) are choking potential, and explicitly suggest how to allocate the marketing/promotional or operational budget to unlock this performance (e.g., deploying extra coolers, easing credit lines, or localized promotional support). Even if the outlet is already performing above its predicted frontier, suggest how resolving bottlenecks can help sustain or further elevate this success.

Guardrails & Style:
- Use professional, executive-ready, and persuasive business language.
- Do NOT use mathematical jargon (such as beta, coefficients, log-space, SFA, frontier, stochastic, etc.).
- Never hallucinate features, attributes, or numbers not present in the JSON payload.
- Focus on translating the metrics into actionable business context.
"""

def generate_narrative_prompt(payload: OutletXAIPayload) -> str:
    """Formats the payload into a clean, token-efficient JSON and constructs the user prompt."""
    # Filter drivers into positive and negative to present them clearly
    positive_drivers = [d for d in payload.top_drivers if d.local_driver_strength > 0][:3]
    negative_drivers = [d for d in payload.top_drivers if d.local_driver_strength < 0][:2]
    
    drivers_summary = []
    for d in positive_drivers:
        # Simplify feature names for easier LLM reading
        friendly_name = d.feature_name.replace("_", " ")
        drivers_summary.append({
            "feature": friendly_name,
            "value": d.feature_value,
            "direction": "Positive Impact",
            "compounding_lift": f"+{d.local_driver_strength:.2f}%"
        })
    for d in negative_drivers:
        friendly_name = d.feature_name.replace("_", " ")
        drivers_summary.append({
            "feature": friendly_name,
            "value": d.feature_value,
            "direction": "Negative Impact",
            "compounding_lift": f"{d.local_driver_strength:.2f}%"
        })

    # Prepare structured summary
    payload_dict = {
        "outlet_id": payload.outlet_id,
        "historical_actual_sales_liters": round(payload.actual_volume, 2),
        "predicted_maximum_potential_liters": round(payload.predicted_potential, 2),
        "opportunity_gap_liters": round(payload.opportunity_gap, 2),
        "technical_efficiency_score": round(payload.efficiency_score, 2),
        "inefficiency_percentage": round(payload.inefficiency_pct, 2),
        "key_compounding_drivers": drivers_summary,
        "local_environmental_signals": {k.replace("_", " "): round(v, 4) for k, v in payload.local_signals.items() if v != 0},
        "operational_constraints": {k.replace("_", " "): round(v, 4) for k, v in payload.operational_constraints.items()}
    }
    
    return f"""
    Please generate the 3-paragraph executive narrative for the following outlet:
    
    JSON Payload:
    {json.dumps(payload_dict, indent=2)}
    
    Remember:
    - Write exactly a 3-paragraph narrative. No markdown header tags for paragraphs.
    - No statistical jargon. Keep it completely in FMCG/business terms.
    - Ground all claims and numbers strictly in the provided JSON payload.
    """

def run_gemini(api_key: str, prompt: str) -> str:
    """Uses the official google-genai library to call gemini-2.5-flash."""
    from google import genai
    from google.genai import types
    
    logger.info("Invoking Gemini 2.5 Flash...")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.5,
        )
    )
    return response.text.strip()

def run_groq(api_key: str, prompt: str) -> str:
    """Uses the groq library to call llama-3.3-70b-specdec or llama-3.1-8b-instant."""
    from groq import Groq
    
    # Try llama-3.3-70b-specdec first, fall back to llama-3.1-8b-instant
    model_name = "llama-3.3-70b-specdec"
    logger.info(f"Invoking Groq with model: {model_name}...")
    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Failed calling Groq model {model_name}: {e}. Trying fallback model...")
        fallback_model = "llama-3.1-8b-instant"
        response = client.chat.completions.create(
            model=fallback_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()

def generate_explanation(payload: OutletXAIPayload) -> str:
    """Generates the business explanation using either Gemini or Groq API based on environment configuration."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    prompt = generate_narrative_prompt(payload)
    
    if gemini_key:
        try:
            return run_gemini(gemini_key, prompt)
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            if groq_key:
                logger.info("Attempting to fallback to Groq...")
            else:
                raise
                
    if groq_key:
        try:
            return run_groq(groq_key, prompt)
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise
            
    raise ValueError(
        "No LLM API keys found in the environment!\n"
        "Please define GEMINI_API_KEY or GROQ_API_KEY in your .env file."
    )
