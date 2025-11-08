import pandas as pd
import os

# Read the parquet file
file_path = "/home/kurtt/cocktail-research/data/original_ingredients_with_rationalization_20250720_144843.parquet"
df = pd.read_parquet(file_path)

print("File Information:")
print(f"File path: {file_path}")
print(f"File size: {os.path.getsize(file_path):,} bytes")
print(f"DataFrame shape: {df.shape}")
print()

print("Column Information:")
print("Columns:", list(df.columns))
print()
print("Data types:")
print(df.dtypes)
print()

print("First 5 rows:")
print(df.head())
print()

print("Sample of unique values in key columns (if they exist):")
if 'original_ingredient' in df.columns:
    print(f"\nUnique original_ingredient count: {df['original_ingredient'].nunique()}")
    print("Sample original_ingredient values:")
    print(df['original_ingredient'].head(10).tolist())

if 'rationalized_ingredient' in df.columns:
    print(f"\nUnique rationalized_ingredient count: {df['rationalized_ingredient'].nunique()}")
    print("Sample rationalized_ingredient values:")
    print(df['rationalized_ingredient'].head(10).tolist())

print("\nBasic statistics:")
print(df.describe(include='all'))