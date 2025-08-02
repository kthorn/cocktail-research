#!/usr/bin/env python3
"""
Create a table of original ingredient data with rationalization mappings.
Preserves original amounts and units while adding rationalized metadata.
"""

import argparse
import datetime
from tqdm.auto import tqdm
import pandas as pd
from cocktail_utils.database import get_recipe_ingredient_data
from cocktail_utils.ingredients import (
    IngredientParser,
    collect_all_rationalizations,
    normalize_string,
)


def missing_information(row: pd.Series, cols: list[str]) -> bool:
    """Check if all specified columns are missing (NaN or empty string)."""
    return ((row[cols].isna()) | (row[cols] == "")).all()


def create_original_data_export(
    recipe_ingredient_df: pd.DataFrame,
    ingredient_rationalizations: dict,
) -> pd.DataFrame:
    """Create export of original ingredient data with rationalization mappings.

    Args:
        recipe_ingredient_df: DataFrame with recipe-ingredient relationships
        ingredient_rationalizations: Dictionary mapping ingredient names to rationalization results

    Returns:
        DataFrame with original ingredient data plus rationalization columns
    """
    print("Creating original ingredient data export...")
    # Add rationalization columns and create rationalized ingredient names
    for ingredient, rationalization in tqdm(
        ingredient_rationalizations.items(), total=len(ingredient_rationalizations)
    ):
        mask = recipe_ingredient_df["ingredient_name"] == ingredient
        recipe_ingredient_df.loc[mask, "rationalized_category"] = (
            rationalization["category"] or ""
        )
        recipe_ingredient_df.loc[mask, "rationalized_specific_type"] = (
            rationalization["specific_type"] or ""
        )
        recipe_ingredient_df.loc[mask, "rationalized_brand"] = (
            rationalization["brand"] or ""
        )

        # Create rationalized ingredient name
        category = normalize_string(rationalization["category"])
        specific_type = rationalization["specific_type"]
        brand = rationalization["brand"]

        # Build rationalized name: prefer specific_type and brand, only use category if nothing else available
        name_parts = []
        has_specific_type = specific_type and str(specific_type).strip()
        has_brand = brand and str(brand).strip()

        if has_specific_type:
            name_parts.append(normalize_string(specific_type))
        if has_brand:
            name_parts.append(normalize_string(brand))

        # Only use category if we have no specific_type or brand
        if not name_parts:
            name_parts.append(category)

        rationalized_name = " ".join(name_parts)
        recipe_ingredient_df.loc[mask, "rationalized_ingredient_name"] = (
            rationalized_name
        )

    # Reorder columns for better readability
    column_order = [
        "recipe_name",
        "recipe_id",
        "ingredient_name",
        "rationalized_ingredient_name",
        "amount",
        "unit",
        "unit_type",
        "rationalized_category",
        "rationalized_specific_type",
        "rationalized_brand",
    ]

    # Only include columns that exist in the DataFrame
    available_columns = [
        col for col in column_order if col in recipe_ingredient_df.columns
    ]
    remaining_columns = [
        col for col in recipe_ingredient_df.columns if col not in available_columns
    ]
    final_columns = available_columns + remaining_columns
    recipe_ingredient_df = recipe_ingredient_df[final_columns]

    recipe_ingredient_df.to_csv("intermediate.csv")

    # Drop recipes that have ingredients without rationalized_category or rationalized_specific_type
    recipes_to_keep = []
    cols_to_check = [
        "rationalized_category",
        "rationalized_specific_type",
        "rationalized_brand",
    ]
    for recipe_id, recipe_group in recipe_ingredient_df.groupby("recipe_id"):
        has_missing = any(
            missing_information(row, cols_to_check)
            for _, row in recipe_group.iterrows()
        )
        if not has_missing:
            recipes_to_keep.append(recipe_id)

    recipe_ingredient_df = recipe_ingredient_df[
        recipe_ingredient_df["recipe_id"].isin(recipes_to_keep)
    ]

    print(
        f"Created original data export with {len(recipe_ingredient_df)} ingredient entries from {len(recipes_to_keep)} complete recipes"
    )
    return recipe_ingredient_df


def main():
    """Main function to create original ingredients table with rationalization data."""
    parser = argparse.ArgumentParser(
        description="Create original ingredients table with rationalization mappings"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/recipes.db",
        help="Path to the database file",
    )
    parser.add_argument(
        "--min-recipes",
        type=int,
        default=1,
        help="Minimum number of recipes an ingredient must appear in",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="anthropic.claude-3-5-haiku-20241022-v1:0",
        help="Bedrock model ID for LLM rationalization",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of parallel workers for LLM processing",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM rationalization for unmatched ingredients",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for table file",
    )
    args = parser.parse_args()

    # Initialize parser and collect rationalizations
    ingredient_parser = IngredientParser(db_path=args.db_path)
    all_ingredients = ingredient_parser.get_ingredients_from_db(
        min_recipe_count=args.min_recipes
    )
    matched_ingredients = collect_all_rationalizations(
        ingredient_parser,
        all_ingredients,
        args.model_id,
        args.max_workers,
        use_llm=args.use_llm,
    )

    # Get recipe-ingredient data
    print("Fetching recipe-ingredient relationships...")
    recipe_ingredient_df = get_recipe_ingredient_data(args.db_path)

    if recipe_ingredient_df.empty:
        print("No recipe-ingredient data found. Exiting.")
        return

    # Create the original data export
    original_data_df = create_original_data_export(
        recipe_ingredient_df, matched_ingredients
    )
    if original_data_df.empty:
        print("No valid ingredient data created. Exiting.")
        return

    # Save to parquet
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_dir}/original_ingredients_with_rationalization_{timestamp}.parquet"
    original_data_df.to_parquet(output_file, index=False)

    print("Successfully created original ingredient data export:")
    print(f"  - File: {output_file}")
    print(f"  - Ingredient entries: {len(original_data_df)}")
    print(f"  - Recipes covered: {original_data_df['recipe_name'].nunique()}")
    print(f"  - Unique ingredients: {original_data_df['ingredient_name'].nunique()}")


if __name__ == "__main__":
    main()
