import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="NHS Outpatient DNA Risk & Explainability API",
    description="Decision-support service for booking teams. Not for automated appointment cancellation.",
    version="1.0.0",
)

bundle = joblib.load("outputs/model_bundle.joblib")
calibrated_model = bundle["calibrated_model"]
explainer_model = bundle["explainer_model"]
FEATURES = bundle["features"]
explainer = shap.TreeExplainer(explainer_model)

FEATURE_LABELS = {
    "lead_time_days": "Days booked in advance",
    "age_band": "Patient age bracket",
    "imd_decile": "Local deprivation index (IMD)",
    "prev_dna_count": "Previous missed appointments",
    "prev_attended_count": "Previous attended appointments",
    "is_morning_slot": "Morning appointment slot",
    "distance_km": "Travel distance to clinic (km)",
    "specialty_code": "Clinic specialty",
}


class AppointmentRequest(BaseModel):
    lead_time_days: int = Field(..., ge=0, le=365)
    age_band: int = Field(..., ge=1, le=4)
    imd_decile: int = Field(..., ge=1, le=10)
    prev_dna_count: int = Field(..., ge=0)
    prev_attended_count: int = Field(..., ge=0)
    is_morning_slot: int = Field(..., ge=0, le=1)
    distance_km: float = Field(..., ge=0.0)
    specialty_code: int


@app.get("/health")
def health_check():
    return {"status": "healthy", "governance_compliant": True}


@app.post("/predict")
def predict_dna_risk(req: AppointmentRequest):
    input_df = pd.DataFrame([req.model_dump()])[FEATURES]
    prob = float(calibrated_model.predict_proba(input_df)[0, 1])

    # Compute local SHAP feature attributions
    shap_vals = explainer.shap_values(input_df)
    row_shap = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

    drivers = sorted(
        zip(FEATURES, row_shap, input_df.iloc[0].values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]

    top_reasons = [
        {
            "factor": FEATURE_LABELS[feat],
            "value": float(val),
            "direction": "Increases risk" if impact > 0 else "Decreases risk",
            "impact_weight": round(float(impact), 3),
        }
        for feat, impact, val in drivers
    ]

    tier = "High" if prob >= 0.35 else ("Medium" if prob >= 0.20 else "Low")
    recommended_action = (
        "Personal phone call + transport check 5 days prior"
        if tier == "High"
        else ("Automated 2-way SMS reminder 72h prior" if tier == "Medium" else "Standard booking confirmation")
    )

    return {
        "dna_probability": round(prob, 4),
        "risk_tier": tier,
        "recommended_support_action": recommended_action,
        "top_explainability_drivers": top_reasons,
        "clinical_safety_notice": "Advisory only. Do not cancel or overbook patient slots.",
    }
