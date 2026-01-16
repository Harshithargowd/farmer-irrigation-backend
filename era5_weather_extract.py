import ee
import pandas as pd

ee.Initialize(project="gen-lang-client-0582455559")

farm = ee.Geometry.Point([77.59, 12.97])

era5 = (
    ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
    .filterBounds(farm)
    .filterDate("2024-01-01", "2024-03-01")
    .select([
        "temperature_2m",
        "total_precipitation",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m"
    ])
)

weather_data = era5.getRegion(farm, 10000).getInfo()

headers = weather_data[0]
rows = weather_data[1:]
df = pd.DataFrame(rows, columns=headers)

df["time"] = pd.to_datetime(df["time"], unit="ms")

df.to_csv("data/era5_weather.csv", index=False)

print("ERA5 weather data saved successfully!")