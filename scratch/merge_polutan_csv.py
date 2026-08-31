import pandas as pd
import os
from functools import reduce

csv_dir = "data/csv"

# File paths
files = {
    'CH4': os.path.join(csv_dir, "CH4_gresik_timeseries.csv"),
    'CO':  os.path.join(csv_dir, "CO_gresik_timeseries.csv"),
    'NO2': os.path.join(csv_dir, "NO2_gresik_timeseries.csv"),
    'SO2': os.path.join(csv_dir, "SO2_gresik_timeseries.csv")
}

# Read each CSV
dfs = []
for name, filepath in files.items():
    df = pd.read_csv(filepath)
    # Ensure date is string format YYYY-MM-DD
    df['date'] = df['date'].astype(str)
    dfs.append(df)

# Merge all dataframes on 'date'
df_merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dfs)

# Sort by date
df_merged['date'] = pd.to_datetime(df_merged['date'])
df_merged = df_merged.sort_values('date').reset_index(drop=True)

# Format date back to string YYYY-MM-DD
df_merged['date'] = df_merged['date'].dt.strftime('%Y-%m-%d')

# Save to polutan_gresik.csv
output_path = os.path.join(csv_dir, "polutan_gresik.csv")
df_merged.to_csv(output_path, index=False)

print(f"Successfully created: {output_path}")
print(f"Shape: {df_merged.shape}")
print("\nFirst 5 rows:")
print(df_merged.head())
print("\nNull counts:")
print(df_merged.isnull().sum())
