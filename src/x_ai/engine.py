import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from pathlib import Path

from modeling.sfa_model import SFAModel
from .schemas import FeatureImpact, OutletXAIPayload

def get_engine_data(
    model_path: Path,
    data_path: Path,
    outlet_id: str
) -> Tuple[SFAModel, pd.Series]:
    """Loads the model and retrieves the feature row for a specific outlet."""
    # Load model
    if not model_path.exists():
        raise FileNotFoundError(f"SFA model not found at {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Load dataset
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_parquet(data_path)
    
    # Find outlet row
    id_col = [c for c in df.columns if 'id' in c.lower()][0]
    outlet_rows = df[df[id_col] == outlet_id]
    if outlet_rows.empty:
        raise ValueError(f"Outlet ID {outlet_id} not found in the dataset.")
        
    # Sort by Year and Month to get the latest state if multiple records exist
    if 'Year' in outlet_rows.columns and 'Month' in outlet_rows.columns:
        outlet_rows = outlet_rows.sort_values(by=['Year', 'Month'], ascending=False)
        
    outlet_row = outlet_rows.iloc[0]
    return model, outlet_row

def compute_xai_metrics(model: SFAModel, outlet_row: pd.Series) -> OutletXAIPayload:
    """Extracts SFA metrics and computes local feature contributions and inefficiency."""
    # 1. Map features to model feature names with appropriate defaults
    outlet_features = {}
    for feat in model.feature_names:
        if feat == "Intercept":
            outlet_features[feat] = 1.0
        elif feat not in outlet_row:
            outlet_features[feat] = 0.0  # Default to 0.0 for binary/numeric features if missing
        else:
            outlet_features[feat] = float(outlet_row[feat])
            
    # 2. Predict potential (frontier) and get actual
    X_df = pd.DataFrame([outlet_features])
    predicted_potential = float(model.predict_potential(X_df)[0])
    
    vol_col = [c for c in outlet_row.index if 'vol' in c.lower() or 'sales' in c.lower()][0]
    actual_volume = float(outlet_row[vol_col])
    
    efficiency_score = actual_volume / predicted_potential
    opportunity_gap = max(0.0, predicted_potential - actual_volume)
    
    if actual_volume > predicted_potential:
        inefficiency_pct = 0.0
    else:
        inefficiency_pct = (1.0 - efficiency_score) * 100.0
    
    # 3. Calculate feature contributions and translate weights
    feature_impacts = []
    for feat, beta in zip(model.feature_names, model.beta):
        if feat == "Intercept":
            continue
        feat_val = outlet_features[feat]
        
        # SFA percentage impact: (exp(beta) - 1) * 100
        percentage_impact = (np.exp(beta) - 1.0) * 100.0
        
        # Local driver strength: feature_value * percentage_impact
        local_driver_strength = feat_val * percentage_impact
        
        feature_impacts.append(FeatureImpact(
            feature_name=feat,
            coefficient=float(beta),
            percentage_impact=float(percentage_impact),
            feature_value=float(feat_val),
            local_driver_strength=float(local_driver_strength)
        ))
        
    # Sort features by local driver strength descending
    feature_impacts.sort(key=lambda x: x.local_driver_strength, reverse=True)
    
    # 4. Classify environment signals and operational constraints
    local_signals = {}
    operational_constraints = {}
    
    env_keywords = ["competitor", "poi", "distance", "friction", "province", "holiday"]
    constraint_keywords = ["cooler", "flatline", "bias", "rigidity", "cv_"]
    
    for imp in feature_impacts:
        name_lower = imp.feature_name.lower()
        if any(kw in name_lower for kw in env_keywords):
            local_signals[imp.feature_name] = imp.feature_value
        elif any(kw in name_lower for kw in constraint_keywords):
            operational_constraints[imp.feature_name] = imp.feature_value
            
    # Include dynamic defaults for Cooler Count if present in raw row but not in features
    if "Cooler_Count" not in operational_constraints and "Cooler_Count" in outlet_row:
        operational_constraints["Cooler_Count"] = float(outlet_row["Cooler_Count"])
        
    id_col = [c for c in outlet_row.index if 'id' in c.lower()][0]
    outlet_id = str(outlet_row[id_col])
    
    return OutletXAIPayload(
        outlet_id=outlet_id,
        actual_volume=actual_volume,
        predicted_potential=predicted_potential,
        opportunity_gap=opportunity_gap,
        efficiency_score=efficiency_score,
        inefficiency_pct=inefficiency_pct,
        top_drivers=feature_impacts,
        local_signals=local_signals,
        operational_constraints=operational_constraints
    )
