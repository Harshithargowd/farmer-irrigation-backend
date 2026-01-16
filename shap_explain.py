import os
import pandas as pd
import joblib
import shap
import matplotlib

# IMPORTANT: non-GUI backend so plots save correctly
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# -------------------------------
# Ensure output directory exists
# -------------------------------
os.makedirs("outputs", exist_ok=True)

# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("models/downscaling_model.pkl")

# -------------------------------
# Load merged dataset
# -------------------------------
df = pd.read_csv("data/merged_downscaling_data.csv")

# -------------------------------
# Feature columns (MUST match CSV)
# -------------------------------
features = [
    "NDVI",
    "temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "total_precipitation"
]

X = df[features]

# -------------------------------
# Create SHAP explainer
# -------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# =====================================================
# GLOBAL EXPLANATION (Feature importance for all data)
# =====================================================
plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    X,
    feature_names=features,
    show=False
)

plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=300)
plt.close()

print("SHAP summary plot saved successfully!")

# =====================================================
# LOCAL EXPLANATION (Single prediction)
# =====================================================
sample = X.iloc[0:1]

plt.figure(figsize=(12, 4))

shap.force_plot(
    explainer.expected_value,
    explainer.shap_values(sample),
    sample,
    matplotlib=True,
    show=False
)

plt.savefig("outputs/shap_local_example.png", dpi=300, bbox_inches="tight")
plt.close()

print("Local SHAP explanation saved successfully!")