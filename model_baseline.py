import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Inputs (features)
X = df[['NDVI', 'temp', 'wind', 'rain']]

# Output (target)
y = df['soil_moisture']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("Baseline Random Forest MAE:", mae)

# ✅ SAVE THE MODEL
joblib.dump(model, "models/random_forest_model.pkl")
print("Model saved successfully!")