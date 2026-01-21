import joblib
model = joblib.load("models/downscaling_model.pkl")
print("model expects these features:")
print(model.feature_names_in_)