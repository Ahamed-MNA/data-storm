from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class FeatureImpact(BaseModel):
    feature_name: str = Field(description="Name of the model feature")
    coefficient: float = Field(description="Raw SFA model coefficient (beta)")
    percentage_impact: float = Field(description="Global percentage impact ((exp(beta) - 1) * 100)")
    feature_value: float = Field(description="Value of this feature for the specific outlet")
    local_driver_strength: float = Field(description="Local compounding driver strength (feature_value * percentage_impact)")

class OutletXAIPayload(BaseModel):
    outlet_id: str = Field(description="Unique identifier of the retail outlet")
    actual_volume: float = Field(description="Historical actual/baseline sales volume in Liters")
    predicted_potential: float = Field(description="SFA predicted maximum potential sales volume in Liters")
    opportunity_gap: float = Field(description="Absolute difference in Liters between predicted potential and actual baseline")
    efficiency_score: float = Field(description="Technical efficiency score (actual_volume / predicted_potential)")
    inefficiency_pct: float = Field(description="Inefficiency percentage ((1 - efficiency_score) * 100)")
    top_drivers: List[FeatureImpact] = Field(description="Sorted list of feature impacts for the outlet")
    local_signals: Dict[str, float] = Field(description="Local environment signals (e.g., POI density, competitor count)")
    operational_constraints: Dict[str, float] = Field(description="Operational constraints (e.g., cooler count, supply limits)")

class OutletXAIResponse(BaseModel):
    outlet_id: str = Field(description="Unique identifier of the retail outlet")
    actual_volume: float = Field(description="Historical actual sales volume in Liters")
    predicted_potential: float = Field(description="SFA predicted potential sales volume in Liters")
    opportunity_gap: float = Field(description="True opportunity gap in Liters")
    efficiency_score: float = Field(description="Technical efficiency score")
    inefficiency_pct: float = Field(description="Inefficiency percentage")
    explanation: str = Field(description="3-paragraph business narrative explanation")
    payload: OutletXAIPayload = Field(description="Structured JSON data payload used for generation")
