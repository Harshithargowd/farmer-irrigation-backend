# ==================================================
# Downscaling Model Training Script
# ==================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# --------------------------------------------------
# STEP 0: Load merged NDVI + ERA5 data
# --------------------------------------------------
df = pd.read_csv("data/merged_downscaling_data.csv")

print("Dataset loaded successfully")
print("Rows:", len(df))

# --------------------------------------------------
# STEP 1: CREATE PROXY SOIL MOISTURE (PHYSICS-BASED)
# --------------------------------------------------

# Calculate wind speed from u and v components
df["temperature_2m"] = df["temperature_2m"] - 273.15
df["wind_speed"] = np.sqrt(
    df["u_component_of_wind_10m"] ** 2 +
    df["v_component_of_wind_10m"] ** 2
)

# Physics-informed proxy soil moisture equation
df["soil_moisture"] = (
    80 * df["NDVI"] +                       # Vegetation retention
    10 * df["total_precipitation"] -       # Rain contribution
    0.15 * df["temperature_2m"] -             # Evaporation due to heat
    0.2 * df["wind_speed"]                   # Evaporation due to wind
)

# Clip to realistic range (0–100%)
df["soil_moisture"] = df["soil_moisture"].clip(0, 100)
print(df[["NDVI", "soil_moisture"]].head())
print("Proxy soil moisture column created")

# --------------------------------------------------
# STEP 2: TRAIN THE DOWNSCALING MODEL
# --------------------------------------------------
# ❗ THIS IS WHERE MODEL TRAINING MUST BE DONE
# ❗ Training ALWAYS happens AFTER proxy creation
# ❗ NEVER inside api.py

# Input features (must match API input exactly)
features = [
    "NDVI",
    "temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "total_precipitation"
]

X = df[features]
y = df["soil_moisture"]

# Initialize Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X, y)

print("Downscaling model trained successfully")

# --------------------------------------------------
# STEP 3: SAVE TRAINED MODEL
# --------------------------------------------------
joblib.dump(model, "models/downscaling_model.pkl")

print("Model saved to models/downscaling_model.pkl")
test_sample = X.iloc[[0]]
test_pred = model.predict(test_sample)[0]
print("DEBUG test prediction:", test_pred)