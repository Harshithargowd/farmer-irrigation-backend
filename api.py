from fastapi import FastAPI
import pandas as pd
import joblib
import math

# -----------------------------------
# Initialize FastAPI
# -----------------------------------
app = FastAPI(title="Hyperlocal Farming AI API")

# -----------------------------------
# Global model
# -----------------------------------
model = None

# -----------------------------------
# Load model on startup (cloud-safe)
# -----------------------------------
@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("models/downscaling_model.pkl")
    print("✅ Model loaded successfully")
    print("Model expects:", model.feature_names_in_)

# -----------------------------------
# Health check
# -----------------------------------
@app.get("/")
def root():
    return {"status": "Backend is running"}

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
    rain: float = 0.0
):
    """
    Irrigation prediction endpoint
    """

    # -----------------------------------
    # Convert wind speed → components
    # (simple assumption: wind equally split)
    # -----------------------------------
    u_wind = wind / math.sqrt(2)
    v_wind = wind / math.sqrt(2)

    # -----------------------------------
    # Prepare EXACT model input
    # -----------------------------------
    X = pd.DataFrame([{
        "NDVI": ndvi,
        "temperature_2m": temperature,
        "u_component_of_wind_10m": u_wind,
        "v_component_of_wind_10m": v_wind,
        "total_precipitation": rain
    }])

    # -----------------------------------
    # Prediction
    # -----------------------------------
    prediction = float(model.predict(X)[0])

    # -----------------------------------
    # Business logic (irrigation decision)
    # -----------------------------------
    if rain > 5:
        today_action = "Do not irrigate"
        next_days = 2
    elif ndvi < 0.4:
        today_action = "Irrigate today"
        next_days = 1
    else:
        today_action = "Monitor soil moisture"
        next_days = 2

    # -----------------------------------
    # Response
    # -----------------------------------
    return {
        "crop": crop,
        "field_area": field_area,
        "area_unit": area_unit,
        "water_required_litres": round(prediction, 2),
        "today_action": today_action,
        "next_irrigation_in_days": next_days,
        "auto_detected_inputs": {
            "ndvi": round(ndvi, 2),
            "temperature": temperature,
            "wind_speed": wind,
            "rain_next_24h_mm": rain
        },
        "reasoning": [
            "NDVI represents crop health",
            "Temperature and wind affect evapotranspiration",
            "Rain forecast reduces irrigation requirement"
        ]
    }