from fastapi import FastAPI, Query
import pandas as pd
import joblib
import shap
import requests
import base64
from datetime import datetime, timedelta


# ----------------------------------
# Initialize FastAPI
# ----------------------------------
app = FastAPI(title="Hyperlocal Farming AI API")

# ----------------------------------
# Load trained model
# ----------------------------------
model = joblib.load("models/downscaling_model.pkl")

# ----------------------------------
# Crop adjustment factors
# ----------------------------------
CROP_WATER_FACTOR = {
    "wheat": 1.0,
    "corn": 0.85,
    "rice": 1.2,
    "sugarcane": 1.5,
    "maize": 1.1,
    "cotton": 1.3,
    "millets": 0.75
}
SENTINEL_CLIENT_ID = "4f8e6c96-b480-4cf2-be9e-1e937e9303a9"
SENTINEL_CLIENT_SECRET ="tmCaRs1UbwEwAVuWEzKV5GsblxHuwkK4"

# ----------------------------------
# Soil water retention factors
# ----------------------------------
SOIL_FACTOR = {
    "black": 1.2,      # holds more water
    "loamy": 1.0,
    "red": 0.85,
    "alluvial": 1.1
}

# ----------------------------------
# Crop water requirement (mm per crop cycle)
# ----------------------------------
CROP_WATER_MM = {
    "rice": 1200,
    "wheat": 450,
    "maize": 600,
    "groundnut": 500,
    "millets": 350
}

def get_sentinel_token():
    auth_string = f"{SENTINEL_CLIENT_ID}:{SENTINEL_CLIENT_SECRET}"
    encoded = base64.b64encode(auth_string.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(
        "https://services.sentinel-hub.com/oauth/token",
        headers=headers,
        data={"grant_type": "client_credentials"}
    )

    return response.json()["access_token"]

def fetch_real_ndvi(latitude, longitude):
    token = get_sentinel_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    today = datetime.utcnow().date()
    start_date = (today - timedelta(days=10)).isoformat()
    end_date = today.isoformat()

    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B04", "B08"],
        output: { bands: 1 }
      };
    }

    function evaluatePixel(sample) {
      let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
      return [ndvi];
    }
    """

    payload = {
        "input": {
            "bounds": {
                "geometry": {
                    "type": "Point",
                    "coordinates": [longitude, latitude]
                }
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": start_date + "T00:00:00Z",
                        "to": end_date + "T23:59:59Z"
                    }
                }
            }]
        },
        "output": {
            "responses": [{
                "identifier": "default",
                "format": {"type": "application/json"}
            }]
        },
        "evalscript": evalscript
    }

    response = requests.post(
        "https://services.sentinel-hub.com/api/v1/process",
        headers=headers,
        json=payload
    )

    data = response.json()

    try:
        ndvi = data["data"][0]["outputs"]["default"]["bands"][0]["stats"]["mean"]
        return round(ndvi, 3)
    except:
        return 0.2  # fallback

def fetch_weather(latitude, longitude):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&current=temperature_2m,wind_speed_10m"
        "&hourly=precipitation"
    )

    response = requests.get(url)
    data = response.json()

    temperature = data["current"]["temperature_2m"]
    wind = data["current"]["wind_speed_10m"]
    rain = sum(data["hourly"]["precipitation"][:24])  # next 24 hrs rain

    return temperature, wind, rain
def estimate_ndvi(rain, crop):
    if rain > 10:
        return 0.7
    elif rain > 5:
        return 0.5
    elif rain > 0:
        return 0.3
    else:
        return 0.15
# ----------------------------------
# SHAP explanation function
# ----------------------------------
def explain_prediction(model, input_df):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)[0]

    explanation = {}
    for feature, value in zip(input_df.columns, shap_vals):
        explanation[feature] = round(float(value), 2)

    return explanation

# ----------------------------------
# Reasoning generator
# ----------------------------------
def generate_reasoning(shap_values):
    reasoning = []

    for feature, value in shap_values.items():
        if abs(value) < 0.1:
            continue

        if feature == "NDVI":
            reasoning.append(
                "Healthy vegetation helped retain soil moisture"
                if value > 0 else
                "Low vegetation cover reduced soil moisture"
            )

        elif feature == "temperature_2m":
            reasoning.append(
                "High temperature increased evaporation"
                if value < 0 else
                "Lower temperature helped retain moisture"
            )

        elif "wind" in feature:
            reasoning.append(
                "Strong wind increased evaporation"
                if value < 0 else
                "Low wind speed conserved moisture"
            )

        elif feature == "total_precipitation":
            reasoning.append(
                "Recent rainfall increased soil moisture"
                if value > 0 else
                "Lack of rainfall reduced soil moisture"
            )

    if not reasoning:
        reasoning.append("Weather and vegetation conditions are balanced")

    return reasoning

# ----------------------------------
# Rule-based soil detection
# ----------------------------------
def detect_soil_type(latitude, longitude):
    if 15 <= latitude <= 25 and 74 <= longitude <= 78:
        return "Black"
    elif 20 <= latitude <= 30 and 75 <= longitude <= 85:
        return "Alluvial"
    elif 8 <= latitude <= 20 and 70 <= longitude <= 80:
        return "Red"
    else:
        return "Loamy"

# ----------------------------------
# Root endpoint
# ----------------------------------
@app.get("/")
def home():
    return {"message": "Hyperlocal Farming AI API running successfully"}

# ----------------------------------
# Prediction endpoint (FINAL)
# ----------------------------------
@app.get("/predict")
def predict(
    latitude: float = Query(...),
    longitude: float = Query(...),
    crop: str = Query("wheat"),
    field_area: float = Query(...),
    area_unit: str = Query(...),
    soil_type: str = Query("")
):
    # AUTO weather detection
    temp, wind, rain = fetch_weather(latitude, longitude)

# AUTO NDVI detection
    ndvi = fetch_real_ndvi(latitude, longitude)
    # ----------------------------------
    # Auto-detect soil type
    # ----------------------------------
    if not soil_type.strip():
        soil_type = detect_soil_type(latitude, longitude)

    soil_type_lower = soil_type.lower()

    # ----------------------------------
    # Prepare model input
    # ----------------------------------
    input_df = pd.DataFrame([{
        "NDVI": ndvi,
        "temperature_2m": temp,
        "u_component_of_wind_10m": wind,
        "v_component_of_wind_10m": 0.0,
        "total_precipitation": rain
    }])

    # ----------------------------------
    # Model prediction
    # ----------------------------------
    predicted_soil_moisture = float(model.predict(input_df)[0])

    # ----------------------------------
    # Crop adjustment
    # ----------------------------------
    crop_factor = CROP_WATER_FACTOR.get(crop.lower(), 1.0)
    crop_adjusted_moisture = predicted_soil_moisture * crop_factor

    # ----------------------------------
    # Convert field area to square meters
    # ----------------------------------
    area_m2 = field_area * 4047 if area_unit.lower() == "acre" else field_area * 10000

    # ----------------------------------
    # Moisture deficit
    # ----------------------------------
    ideal_moisture = 40.0
    moisture_deficit_ratio = max(0, ideal_moisture - crop_adjusted_moisture) / ideal_moisture

    # ----------------------------------
    # Soil factor
    # ----------------------------------
    soil_factor = SOIL_FACTOR.get(soil_type_lower, 1.0)

    # ----------------------------------
    # Water requirement calculation
    # ----------------------------------
    crop_mm = CROP_WATER_MM.get(crop.lower(), 500)

    base_water_litres = crop_mm * area_m2 * 0.001
    water_required_litres = base_water_litres * moisture_deficit_ratio * soil_factor

    # ----------------------------------
    # Irrigation schedule logic
    # ----------------------------------
    rain_forecast_next_day = rain

    if rain_forecast_next_day > 5:
        today_action = "Do not irrigate"
        next_irrigation_days = 2
    elif crop_adjusted_moisture < 20:
        today_action = "Irrigate today"
        next_irrigation_days = 3
    else:
        today_action = "Monitor soil moisture"
        next_irrigation_days = 1

    # ----------------------------------
    # Soil explanation
    # ----------------------------------
    soil_explanation = (
        f"Soil type at your location is {soil_type}. "
        f"This soil has {'high' if soil_factor > 1 else 'moderate'} "
        f"water holding capacity, so irrigation is adjusted accordingly."
    )

    # ----------------------------------
    # SHAP explanation + reasoning
    # ----------------------------------
    shap_explanation = explain_prediction(model, input_df)
    reasoning = generate_reasoning(shap_explanation)

    # ----------------------------------
    # Final API response
    # ----------------------------------
    return {
        "crop": crop,
        "soil_type": soil_type,
        "field_area": field_area,
        "area_unit": area_unit,
        "predicted_soil_moisture": round(predicted_soil_moisture, 2),
        "crop_adjusted_soil_moisture": round(crop_adjusted_moisture, 2),
        "water_required_litres": round(water_required_litres, 2),

        "irrigation_advice": today_action,
        "today_action": today_action,
        "next_irrigation_in_days": next_irrigation_days,
        "rain_expected_tomorrow": rain_forecast_next_day > 5,

        "shap_values": shap_explanation,
        "reasoning": reasoning,

        "latitude": latitude,
        "longitude": longitude,
        "soil_message": soil_explanation,
        "auto_detected_inputs": {
            "ndvi": round(ndvi, 2),
            "temperature": temp,
            "wind": wind,
            "rain_next_24h_mm": round(rain, 2)
        }
    }