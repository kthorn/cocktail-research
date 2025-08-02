#!/usr/bin/env python3
"""
Create a rationalized recipe-ingredient matrix in mL amounts or binary presence.
Uses dictionary lookup first, then Claude Haiku for unmatched ingredients.
Outputs to parquet with multi-indexed columns.
"""

import argparse
import datetime

import pandas as pd
from cocktail_utils.database import get_recipe_ingredient_data
from cocktail_utils.ingredients import (
    IngredientParser,
    collect_all_rationalizations,
    prepare_rationalized_dataframe,
    normalize_string,
)


def create_rationalized_matrix(
    recipe_ingredient_df: pd.DataFrame,
    ingredient_rationalizations: dict,
    matrix_type: str = "amount",
) -> pd.DataFrame:
    """Create a recipe-ingredient matrix with rationalized multi-indexed columns.

    Args:
        recipe_ingredient_df: DataFrame with recipe-ingredient relationships
        ingredient_rationalizations: Dictionary mapping ingredient names to rationalization results
        matrix_type: Type of matrix to create - "amount" for ingredient amounts, "boolean" for presence

    Returns:
        DataFrame with recipes as rows and ingredients as multi-indexed columns.
        For "amount": values are ingredient amounts in ml
        For "boolean": values are True/False indicating ingredient presence
    """
    print(f"Creating {matrix_type} recipe-ingredient matrix...")
    filtered_df = prepare_rationalized_dataframe(
        recipe_ingredient_df, ingredient_rationalizations
    )

    if filtered_df.empty:
        print("No valid recipes found after filtering.")
        return pd.DataFrame()

    if matrix_type == "boolean":
        # Create a presence indicator (1 for any amount > 0, 0 otherwise)
        filtered_df["value"] = (filtered_df["amount_ml"] > 0).astype(int)
        aggfunc = "max"  # Use max to handle duplicates (if present multiple times, still present)
    else:  # amount
        # Use amounts directly
        filtered_df["value"] = filtered_df["amount_ml"]
        aggfunc = "sum"  # Sum amounts for duplicates

    # Group by recipe and rationalized columns
    grouped_df = (
        filtered_df.groupby(["recipe_name", "category", "specific_type", "brand"])[
            "value"
        ]
        .agg(aggfunc)
        .reset_index()
    )

    # Create pivot table with multi-index columns
    matrix_df = grouped_df.pivot_table(
        index="recipe_name",
        columns=["category", "specific_type", "brand"],
        values="value",
        fill_value=0,
        aggfunc=aggfunc,
    )

    # Convert to boolean if requested
    if matrix_type == "boolean":
        matrix_df = matrix_df.astype(bool)

    # Sort columns for better organization
    matrix_df = matrix_df.sort_index(axis=1)
    print(
        f"Created {matrix_type} matrix with {len(matrix_df)} recipes and {len(matrix_df.columns)} ingredient columns"
    )

    return matrix_df


def main():
    """Main function to create rationalized recipe-ingredient matrix."""
    parser = argparse.ArgumentParser(
        description="Create rationalized recipe-ingredient matrix with multi-indexed columns"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/punch_recipes.db",
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
        "--matrix-type",
        type=str,
        choices=["amount", "boolean"],
        default="amount",
        help="Type of matrix to create: amount (mL values) or boolean (presence)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for matrix files",
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

    # Create the rationalized matrix
    matrix_df = create_rationalized_matrix(
        recipe_ingredient_df, matched_ingredients, matrix_type=args.matrix_type
    )
    if matrix_df.empty:
        print(f"No valid {args.matrix_type} matrix created. Exiting.")
        return

    # Save to parquet
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_dir}/rationalized_matrix_{args.matrix_type}_{timestamp}.parquet"
    matrix_df.to_parquet(output_file, index=True)

    print("Successfully created rationalized matrix:")
    print(f"  - File: {output_file}")
    print(f"  - Type: {args.matrix_type}")
    print(f"  - Recipes: {len(matrix_df)}")
    print(f"  - Ingredient columns: {len(matrix_df.columns)}")


if __name__ == "__main__":
    main()