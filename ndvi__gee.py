import ee
ee.Authenticate()
ee.Initialize()

# Define area (example farm)
farm = ee.Geometry.Point([77.59, 12.97])  # Bangalore example

# Load Sentinel-2
s2 = ee.ImageCollection("COPERNICUS/S2") \
        .filterBounds(farm) \
        .filterDate("2024-01-01", "2024-02-01") \
        .median()

# NDVI
ndvi = s2.normalizedDifference(['B8', 'B4'])

print("NDVI object created successfully")