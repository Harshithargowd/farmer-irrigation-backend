from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import joblib
import os

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
    model_path = "models/downscaling_model.pkl"

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
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
    import shap  # ✅ lazy-loaded (cloud-safe)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # RETURN ONLY NUMERIC VALUES (SAFE)
    return shap_values.values.tolist()

# -----------------------------------
# Prediction API
# -----------------------------------
@app.get("/predict")
def predict(
    latitude: float = Query(...),
    longitude: float = Query(...),
    crop: str = Query(...),
    field_area: float = Query(...),
    area_unit: str = Query("Acre"),
    ndvi: float = Query(0.5),
    temperature: float = Query(30.0),
    wind: float = Query(2.0),
    rain: float = Query(0.0),
    explain: bool = Query(False)
):
    """
    Main irrigation prediction endpoint
    """

    # -----------------------------------
    # Safety check
    # -----------------------------------
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # -----------------------------------
    # Prepare input features (NUMERIC ONLY)
    # Must match training features
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
    try:
        prediction = float(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # -----------------------------------
    # Base response
    # -----------------------------------
    response = {
        "crop": crop,
        "soil_type": "Loamy",  # placeholder
        "field_area": field_area,
        "area_unit": area_unit,
        "water_required_litres": round(prediction, 2),
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
        response["shap_summary"] = generate_shap_explanation(model, X)

    return response