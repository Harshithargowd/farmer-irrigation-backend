import ee
import pandas as pd

ee.Initialize(project="gen-lang-client-0582455559")

farm = ee.Geometry.Point([77.59, 12.97])

s2 = (
    ee.ImageCollection("COPERNICUS/S2")
    .filterBounds(farm)
    .filterDate("2024-01-01", "2024-03-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)

def add_ndvi(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)

ndvi_collection = s2.map(add_ndvi)

# Extract NDVI time series
ndvi_data = ndvi_collection.select("NDVI").getRegion(farm, 10).getInfo()

# Convert to DataFrame
headers = ndvi_data[0]
rows = ndvi_data[1:]
df = pd.DataFrame(rows, columns=headers)

df["time"] = pd.to_datetime(df["time"], unit="ms")

# Save CSV
df.to_csv("data/ndvi_timeseries.csv", index=False)

print("NDVI time series saved successfully!")