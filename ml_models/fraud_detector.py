"""
fraud_detector.py
Inference wrapper for the GradientBoosting fraud detection pipeline.
Loads the single-pkl pipeline once at module level.
"""

import joblib
import pandas as pd
from pathlib import Path

_MODEL_PATH = Path("ml_models/fraud_model.pkl")
_pipeline   = None

# ── Column defaults matching carclaims.csv ───────────────────────────────────
_DEFAULTS = {
    'AccidentArea': 'Urban',
    'AddressChange-Claim': 'no change',
    'Age': 38,
    'AgeOfPolicyHolder': '31 to 35',
    'AgeOfVehicle': '7 years',
    'AgentType': 'External',
    'BasePolicy': 'Collision',
    'DayOfWeek': 'Monday',
    'DayOfWeekClaimed': 'Monday',
    'Days:Policy-Accident': 'more than 30',
    'Days:Policy-Claim': 'more than 30',
    'Deductible': 400,
    'DriverRating': 2,
    'Fault': 'Policy Holder',
    'Make': 'Pontiac',
    'MaritalStatus': 'Married',
    'Month': 'Jan',
    'MonthClaimed': 'Jan',
    'NumberOfCars': '1 vehicle',
    'NumberOfSuppliments': 'none',
    'PastNumberOfClaims': '2 to 4',
    'PoliceReportFiled': 'No',
    'PolicyType': 'Sedan - Collision',
    'Sex': 'Male',
    'VehicleCategory': 'Sedan',
    'VehiclePrice': '20,000 to 29,000',
    'WeekOfMonth': 3,
    'WeekOfMonthClaimed': 3,
    'WitnessPresent': 'No',
    'Year': 1995
}


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return
    try:
        _pipeline = joblib.load(str(_MODEL_PATH))
    except Exception as e:
        print(f"[fraud_detector] WARNING: Could not load model: {e}")
        _pipeline = None


_load_pipeline()


def assess_fraud_risk(claim_data: dict) -> dict:
    """
    Assess fraud risk for a single claim.

    Args:
        claim_data: dict of raw feature values. Missing keys are filled
                    with sensible defaults from the carclaims.csv dataset.

    Returns:
        {"fraud_probability": 0.847, "risk_level": "HIGH", "flag": True}
        risk_level: LOW (<0.30), MEDIUM (0.30-0.60), HIGH (>0.60)
    On any error returns a safe default (flag=False).
    """
    if _pipeline is None:
        return {
            "fraud_probability": 0.0,
            "risk_level": "LOW",
            "flag": False,
            "error": "model_not_loaded",
        }

    try:
        # Build row with defaults for missing keys
        row = {**_DEFAULTS, **{k: v for k, v in claim_data.items() if k in _DEFAULTS}}
        df  = pd.DataFrame([row])

        prob  = float(_pipeline.predict_proba(df)[0][1])
        prob  = round(prob, 4)

        if prob < 0.30:
            risk_level = "LOW"
            flag       = False
        elif prob < 0.60:
            risk_level = "MEDIUM"
            flag       = False
        else:
            risk_level = "HIGH"
            flag       = True

        return {"fraud_probability": prob, "risk_level": risk_level, "flag": flag}

    except Exception as e:
        return {
            "fraud_probability": 0.0,
            "risk_level": "LOW",
            "flag": False,
            "error": str(e),
        }
