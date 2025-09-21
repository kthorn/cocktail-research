#!/usr/bin/env python3
"""
Create a simple table of recipe-ingredient data without rationalization.
Exports raw data as it appears in the database.
"""

import argparse
import datetime
import pandas as pd
from cocktail_utils.database import get_recipe_ingredient_data


def main():
    """Main function to create raw recipe-ingredient table."""
    parser = argparse.ArgumentParser(
        description="Create raw recipe-ingredient table without rationalization"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/recipes.db",
        help="Path to the database file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for table file",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Output file format (parquet or csv)",
    )
    args = parser.parse_args()

    # Get recipe-ingredient data
    print("Fetching recipe-ingredient relationships...")
    recipe_ingredient_df = get_recipe_ingredient_data(args.db_path)

    if recipe_ingredient_df.empty:
        print("No recipe-ingredient data found. Exiting.")
        return

    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to specified format
    if args.output_format == "parquet":
        output_file = f"{args.output_dir}/raw_recipe_ingredients_{timestamp}.parquet"
        recipe_ingredient_df.to_parquet(output_file, index=False)
    else:  # csv
        output_file = f"{args.output_dir}/raw_recipe_ingredients_{timestamp}.csv"
        recipe_ingredient_df.to_csv(output_file, index=False)

    print("Successfully created raw recipe-ingredient data export:")
    print(f"  - File: {output_file}")
    print(f"  - Ingredient entries: {len(recipe_ingredient_df)}")
    print(f"  - Recipes covered: {recipe_ingredient_df['recipe_name'].nunique()}")
    print(
        f"  - Unique ingredients: {recipe_ingredient_df['ingredient_name'].nunique()}"
    )

    # Display column information
    print(f"  - Columns: {list(recipe_ingredient_df.columns)}")
    print(f"  - Data shape: {recipe_ingredient_df.shape}")


if __name__ == "__main__":
    main()
