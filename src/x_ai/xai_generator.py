import os
import sys
import pickle
import pandas as pd
import warnings
from pathlib import Path
from dotenv import load_dotenv

# --- HIDE WARNINGS ---
warnings.filterwarnings("ignore")

# ==========================================
# 1. SMART PATH ROUTING
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = CURRENT_DIR.parent.parent           
sys.path.append(str(PROJECT_ROOT / "src"))

# Now this perfectly matches what the pickle file is looking for!
from modeling.sfa_model import SFAModel

# ==========================================
# 2. SETUP & SECURITY (GROQ + LLaMA 3)
# ==========================================
from groq import Groq

load_dotenv(PROJECT_ROOT / ".env")
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found! Make sure it is in your .env file.")

# Initialize the blazing fast Groq client
client = Groq(api_key=api_key)

# ==========================================
# 3. THE SFA MATH EXTRACTOR
# ==========================================
def extract_model_reasons(model, outlet_features, actual_volume):
    X_matrix = pd.DataFrame([outlet_features])
    predicted_potential = model.predict_potential(X_matrix)[0]
    
    # --- PRO GUARDRAILS ---
    # If pure luck pushes actuals above the baseline frontier, we cap it for the business UI.
    if actual_volume > predicted_potential:
        predicted_potential = actual_volume
        inefficiency_penalty_pct = 0.0 # They operated at 100% efficiency
    else:
        efficiency = actual_volume / predicted_potential
        inefficiency_penalty_pct = (1 - efficiency) * 100

    feature_contributions = {}
    for feat, beta in zip(model.feature_names, model.beta):
        # SKIP THE INTERCEPT
        if feat in outlet_features and feat != "Intercept":
            impact = outlet_features[feat] * beta 
            feature_contributions[feat] = impact
            
    sorted_drivers = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
    top_drivers = sorted_drivers[:2] 

    return predicted_potential, inefficiency_penalty_pct, top_drivers

# ==========================================
# 4. THE LLM TRANSLATOR
# ==========================================
def generate_business_explanation(outlet_id, actual_vol, predicted_vol, inefficiency_pct, top_drivers):
    driver_1_name = top_drivers[0][0].replace("_", " ")
    driver_2_name = top_drivers[1][0].replace("_", " ")
    
    # --- DYNAMIC PROMPT LOGIC ---
    if inefficiency_pct > 0.5:
        constraint_text = f"OPERATIONAL CONSTRAINTS (Why actuals fell short):\n- Systemic Inefficiency Penalty: {inefficiency_pct:.1f}% (Lost volume due to supply limits or credit caps)."
        action_instruction = "3. The historical constraints holding the outlet back and how to unlock the rest of the volume."
    else:
        constraint_text = "OPERATIONAL STATUS: The outlet is operating at PEAK efficiency (0% penalty) and is maximizing its current market conditions."
        action_instruction = "3. Praise the outlet for operating at peak efficiency with no current operational bottlenecks."

    prompt = f"""
    You are an expert FMCG Business Consultant. Explain the sales potential of a retail outlet to a non-technical Regional Sales Manager. Keep the tone professional, clear, and actionable. Do not use statistical jargon.

    DATA FOR OUTLET: {outlet_id}
    - Historical Actual Sales: {actual_vol:.1f} Liters
    - PREDICTED MAXIMUM POTENTIAL: {predicted_vol:.1f} Liters
    
    KEY DRIVERS (Why the potential is high):
    1. {driver_1_name}
    2. {driver_2_name}
    
    {constraint_text}

    Write a single, 4-sentence paragraph explaining:
    1. The uncapped potential score compared to actual sales.
    2. The main local/environmental factors driving this potential.
    {action_instruction}
    """
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )
    
    return response.choices[0].message.content

# ==========================================
# 5. TEST EXECUTION (WITH REAL DATA)
# ==========================================
if __name__ == "__main__":
    print("Finding and Loading SFA Model...")
    MODEL_PATH = PROJECT_ROOT / "outputs" / "sfa_model.pkl"
    
    with open(MODEL_PATH, "rb") as f:
        my_model = pickle.load(f)
    print("Model loaded successfully!")
    
    print("Loading Real Phase 1 Dataset...")
    DATA_PATH = PROJECT_ROOT / "data" / "gold" / "sfa_refined.parquet"
    
    # Load your team's actual dataset
    df = pd.read_parquet(DATA_PATH)
    
    # Grab the very first row (Outlet) to test
    sample_shop = df.iloc[0]
    
    # Automatically find the ID and Volume columns (handling different naming styles)
    id_col = [c for c in df.columns if 'id' in c.lower()][0]
    vol_col = [c for c in df.columns if 'vol' in c.lower() or 'sales' in c.lower()][0]
    
    test_outlet_id = sample_shop[id_col]
    test_actual_volume = sample_shop[vol_col]
    

    # Pull the exact real values, but manually supply the Intercept if missing
    test_features = {}
    for feat in my_model.feature_names:
        if feat == "Intercept" or feat not in sample_shop:
            test_features[feat] = 1.0  # The baseline mathematical constant
        else:
            test_features[feat] = sample_shop[feat]
    
    print(f"\nAnalyzing Outlet {test_outlet_id} with REAL data...")
    
    predicted_vol, inefficiency, top_drivers = extract_model_reasons(my_model, test_features, test_actual_volume)
    final_explanation = generate_business_explanation(test_outlet_id, test_actual_volume, predicted_vol, inefficiency, top_drivers)
    
    print("\n================ FINAL XAI OUTPUT ================")
    print(final_explanation)
    print("==================================================\n")