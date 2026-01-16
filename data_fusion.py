import pandas as pd

ndvi = pd.read_csv("data/ndvi_timeseries.csv")
weather = pd.read_csv("data/era5_weather.csv")

ndvi["date"] = pd.to_datetime(ndvi["time"]).dt.date
weather["date"] = pd.to_datetime(weather["time"]).dt.date

# Aggregate ERA5 hourly → daily (NUMERIC ONLY)
numeric_cols = weather.select_dtypes(include="number").columns

weather_daily = (
    weather.groupby("date")[numeric_cols]
    .mean()
    .reset_index()
)

# Merge NDVI + Weather
merged = pd.merge(ndvi, weather_daily, on="date")

# -------- PROXY TARGET --------
merged["soil_moisture"] = (
    merged["NDVI"] * 40
    + merged["total_precipitation"] * 5
    - merged["temperature_2m"] * 0.5
)
merged["soil_moisture"] = merged["soil_moisture"].clip(0, 100)

merged.to_csv("data/merged_downscaling_data.csv", index=False)

print("Fusion completed successfully!")