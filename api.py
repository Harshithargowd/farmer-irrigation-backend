from fastapi import FastAPI, Query
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# -----------------------------------
# Initialize FastAPI
# -----------------------------------
app = FastAPI(title="Hyperlocal Farming AI API")

# -----------------------------------
# Global model variable
# -----------------------------------
model = None

# -----------------------------------
# Load model ONCE (cloud-safe)
# -----------------------------------
@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("models/downscaling_model.pkl")
    print("✅ Model loaded successfully")

# -----------------------------------
# Health check
# -----------------------------------
@app.get("/")
def root():
    return {"status": "Backend is running"}

# -----------------------------------
# SHAP explanation (lazy import)
# -----------------------------------
def generate_shap_explanation(model, X):
    import shap  # ✅ lazy-loaded (important for cloud)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    return shap_values

# -----------------------------------
# Prediction API
# -----------------------------------
@app.get("/predict")
def predict(
    latitude: float,
    longitude: float,
    crop: str,
    field_area: float,
    area_unit: str = "Acre",
    ndvi: float = 0.5,
    temperature: float = 30.0,
    wind: float = 2.0,
    rain: float = 0.0,
    explain: bool = False
):
    """
    Main irrigation prediction endpoint
    """

    # -----------------------------------
    # Prepare input features
    # -----------------------------------
    X = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "ndvi": ndvi,
        "temperature": temperature,
        "wind": wind,
        "rain": rain,
        "field_area": field_area
    }])

    # -----------------------------------
    # Model prediction
    # -----------------------------------
    prediction = model.predict(X)[0]

    # -----------------------------------
    # Base response
    # -----------------------------------
    response = {
        "crop": crop,
        "soil_type": "Loamy",  # placeholder (your soil logic can replace this)
        "field_area": field_area,
        "area_unit": area_unit,
        "water_required_litres": round(float(prediction), 2),
        "irrigation_advice": "Irrigate if soil moisture is low",
        "crop_adjusted_soil_moisture": round(100 - ndvi * 40, 2),
        "rain_expected_tomorrow": rain > 5,
        "today_action": "Irrigate" if rain < 5 else "Wait",
        "next_irrigation_in_days": 1 if rain < 5 else 3,
        "soil_message": "Water requirement calculated based on NDVI and weather",
        "auto_detected_inputs": {
            "ndvi": round(ndvi, 2),
            "temperature": temperature,
            "wind": wind,
            "rain_next_24h_mm": round(rain, 2)
        },
        "reasoning": [
            "NDVI indicates vegetation health",
            "Weather conditions affect evapotranspiration",
            "Rain forecast reduces irrigation need"
        ]
    }

    # -----------------------------------
    # Optional SHAP explanation
    # -----------------------------------
    if explain:
        shap_values = generate_shap_explanation(model, X)
        response["shap_summary"] = shap_values.values.tolist()

    return response