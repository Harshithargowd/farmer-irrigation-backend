import pandas as pd

df = pd.read_csv("data/merged_data.csv")
df = df.dropna()
df['date'] = pd.to_datetime(df['date'])
df.to_csv("data/cleaned_data.csv", index=False)
print("Data cleaned and saved successfully")