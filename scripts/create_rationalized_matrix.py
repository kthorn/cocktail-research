#!/usr/bin/env python3
"""
Create a rationalized recipe-ingredient matrix using dictionary lookup first,
then Claude Haiku for unmatched ingredients. Outputs to parquet with multi-indexed columns.
"""

import argparse
import csv
import datetime
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
from cocktail_utils.database import get_recipe_ingredient_data
from cocktail_utils.ingredients import IngredientParser
from tqdm import tqdm


def llm_rationalize(
    parser: IngredientParser,
    unmatched_ingredients: List[str],
    model_id: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
    max_workers: int = 1,  # Reduced to help with rate limiting
) -> Dict[str, Optional[dict]]:
    """
    Batch process unmatched ingredients using LLM with parallel processing.

    Args:
        parser: IngredientParser instance
        unmatched_ingredients: List of ingredient names to rationalize
        model_id: Bedrock model ID to use
        max_workers: Maximum number of parallel workers

    Returns:
        Dictionary mapping ingredient names to rationalization results
    """
    print(
        f"Rationalizing {len(unmatched_ingredients)} unmatched ingredients using {model_id}..."
    )

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all futures and collect them properly
        future_to_ingredient = {}
        for ingredient in unmatched_ingredients:
            # Add small random delay to stagger requests
            time.sleep(random.uniform(0.1, 0.5))
            future = executor.submit(parser.llm_lookup, ingredient, model_id)
            future_to_ingredient[future] = ingredient

        # Collect results as they complete
        for future in tqdm(
            as_completed(future_to_ingredient),
            total=len(unmatched_ingredients),
            desc="LLM rationalization progress",
        ):
            ingredient = future_to_ingredient[future]
            try:
                match = future.result()
                if match:
                    results[ingredient] = {
                        "brand": match.brand,
                        "specific_type": match.specific_type,
                        "category": match.category,
                        "confidence": match.confidence,
                        "source": match.source,
                    }
                else:
                    results[ingredient] = None
            except Exception as e:
                print(f"Error processing {ingredient}: {e}")
                results[ingredient] = None

    successful_matches = sum(1 for v in results.values() if v is not None)
    print(
        f"Successfully rationalized {successful_matches}/{len(unmatched_ingredients)} ingredients via LLM"
    )

    return results


def write_llm_rationalized_csv(
    llm_results: Dict[str, Optional[dict]], output_file: str
) -> None:
    """
    Write LLM-rationalized ingredients to CSV file.

    Args:
        llm_results: Dictionary mapping ingredient names to rationalization results
        output_file: Path to output CSV file
    """
    successful_results = {
        ingredient: result
        for ingredient, result in llm_results.items()
        if result is not None
    }

    if not successful_results:
        print("No successful LLM rationalizations to write.")
        return

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(
            [
                "original_ingredient",
                "category",
                "specific_type",
                "brand",
                "confidence",
                "source",
            ]
        )

        # Write data, sorted by original ingredient name
        for ingredient in sorted(successful_results.keys()):
            result = successful_results[ingredient]
            writer.writerow(
                [
                    ingredient,
                    result["category"],
                    result["specific_type"] or "",
                    result["brand"] or "",
                    result["confidence"],
                    result["source"],
                ]
            )

    print(
        f"Wrote {len(successful_results)} LLM-rationalized ingredients to {output_file}"
    )


def create_multi_index_column(
    category: str, specific_type: Optional[str], brand: Optional[str]
) -> Tuple[str, str, str]:
    """
    Create a multi-index column tuple with proper handling of None values.
    """
    return (category or "unknown", specific_type or "generic", brand or "generic")


def create_rationalized_matrix(
    recipe_ingredient_df: pd.DataFrame,
    ingredient_rationalizations: Dict[str, dict],
    min_recipe_count: int = 1,
) -> pd.DataFrame:
    """
    Create a recipe-ingredient matrix with rationalized multi-indexed columns.

    Args:
        recipe_ingredient_df: DataFrame with recipe-ingredient relationships
        ingredient_rationalizations: Dictionary mapping ingredient names to rationalization data
        min_recipe_count: Minimum number of recipes an ingredient must appear in

    Returns:
        DataFrame with recipes as rows and rationalized ingredients as multi-indexed columns
    """
    print("Creating rationalized recipe-ingredient matrix...")

    # Filter ingredients by minimum recipe count
    ingredient_counts = recipe_ingredient_df["ingredient_name"].value_counts()
    valid_ingredients = ingredient_counts[
        ingredient_counts >= min_recipe_count
    ].index.tolist()

    filtered_df = recipe_ingredient_df[
        recipe_ingredient_df["ingredient_name"].isin(valid_ingredients)
    ]
    print(
        f"Using {len(valid_ingredients)} ingredients that appear in at least {min_recipe_count} recipes"
    )

    # Create mapping from original ingredient names to rationalized column names
    ingredient_to_column = {}
    column_tuples = []

    for ingredient in valid_ingredients:
        if ingredient in ingredient_rationalizations:
            rationalization = ingredient_rationalizations[ingredient]
            column_tuple = create_multi_index_column(
                rationalization["category"],
                rationalization["specific_type"],
                rationalization["brand"],
            )
            ingredient_to_column[ingredient] = column_tuple
            column_tuples.append(column_tuple)
        else:
            # Unmatched ingredient - use generic categorization
            column_tuple = create_multi_index_column("unknown", "unknown", ingredient)
            ingredient_to_column[ingredient] = column_tuple
            column_tuples.append(column_tuple)

    # Remove duplicates while preserving order
    unique_columns = []
    seen = set()
    for col in column_tuples:
        if col not in seen:
            unique_columns.append(col)
            seen.add(col)

    print(f"Created {len(unique_columns)} unique rationalized ingredient columns")

    # Create the matrix
    recipes = sorted(filtered_df["recipe_name"].unique())
    matrix_data = []

    for recipe in tqdm(recipes, desc="Building matrix"):
        recipe_data = filtered_df[filtered_df["recipe_name"] == recipe]
        row = {}

        for column_tuple in unique_columns:
            # Find all ingredients that map to this column
            matching_ingredients = [
                ing for ing, col in ingredient_to_column.items() if col == column_tuple
            ]

            # Sum amounts for all matching ingredients in this recipe
            total_amount = 0.0
            for ingredient in matching_ingredients:
                ingredient_rows = recipe_data[
                    recipe_data["ingredient_name"] == ingredient
                ]
                if not ingredient_rows.empty:
                    # Handle None amounts by treating them as 0
                    amounts = ingredient_rows["amount"].fillna(0)
                    total_amount += amounts.sum()

            row[column_tuple] = total_amount

        matrix_data.append(row)

    # Create DataFrame with multi-index columns
    matrix_df = pd.DataFrame(matrix_data, index=recipes)

    # Create MultiIndex for columns
    multi_index = pd.MultiIndex.from_tuples(
        unique_columns, names=["category", "specific_type", "brand"]
    )
    matrix_df.columns = multi_index

    # Sort columns for better organization
    matrix_df = matrix_df.sort_index(axis=1)

    print(
        f"Created matrix with {len(matrix_df)} recipes and {len(matrix_df.columns)} ingredient columns"
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
        "--output-file",
        type=str,
        default=None,
        help="Output parquet file path (default: rationalized_matrix_TIMESTAMP.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated files (default: current directory)",
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

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize ingredient parser
    ingredient_parser = IngredientParser(db_path=args.db_path)

    # Step 1: Get all ingredients and rationalize using dictionary lookup
    print("Step 1: Dictionary-based ingredient rationalization...")
    all_ingredients = ingredient_parser.get_ingredients_from_db(
        min_recipe_count=args.min_recipes
    )

    matched_ingredients = {}
    unmatched_ingredients = []
    llm_csv_file = None

    for ing_usage in tqdm(all_ingredients, desc="Dictionary lookup"):
        ingredient_name = ing_usage.ingredient_name
        match = ingredient_parser.rationalize_ingredient(ingredient_name)

        if match:
            matched_ingredients[ingredient_name] = {
                "brand": match.brand,
                "specific_type": match.specific_type,
                "category": match.category,
                "confidence": match.confidence,
                "source": match.source,
            }
        else:
            unmatched_ingredients.append(ingredient_name)

    print(f"Dictionary matched: {len(matched_ingredients)}")
    print(f"Unmatched ingredients: {len(unmatched_ingredients)}")

    # Step 2: Use LLM to rationalize unmatched ingredients
    if unmatched_ingredients:
        print("Step 2: LLM-based rationalization for unmatched ingredients...")
        llm_results = llm_rationalize(
            ingredient_parser,
            unmatched_ingredients,
            model_id=args.model_id,
            max_workers=args.max_workers,
        )

        # Write LLM results to CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        llm_csv_file = f"{args.output_dir}/llm_rationalized_ingredients_{timestamp}.csv"
        write_llm_rationalized_csv(llm_results, llm_csv_file)

        # Add successful LLM results to matched ingredients
        for ingredient, result in llm_results.items():
            if result:
                matched_ingredients[ingredient] = result

    total_rationalized = len(matched_ingredients)
    total_ingredients = len(all_ingredients)
    print(
        f"Total rationalized: {total_rationalized}/{total_ingredients} ({total_rationalized / total_ingredients * 100:.1f}%)"
    )

    # Step 3: Get recipe-ingredient relationships
    print("Step 3: Fetching recipe-ingredient relationships...")
    recipe_ingredient_df = get_recipe_ingredient_data(args.db_path)

    if recipe_ingredient_df.empty:
        print("No recipe-ingredient data found. Exiting.")
        return

    # Step 4: Create the rationalized matrix
    print("Step 4: Creating rationalized matrix...")
    matrix_df = create_rationalized_matrix(
        recipe_ingredient_df, matched_ingredients, min_recipe_count=args.min_recipes
    )

    # Step 5: Save to parquet
    if args.output_file:
        output_file = args.output_file
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{args.output_dir}/rationalized_matrix_{timestamp}.parquet"

    print(f"Step 5: Saving matrix to {output_file}...")
    matrix_df.to_parquet(output_file, index=True)

    print("Successfully created rationalized matrix:")
    print(f"  - {len(matrix_df)} recipes")
    print(f"  - {len(matrix_df.columns)} ingredient columns")
    print(f"  - Multi-index levels: {matrix_df.columns.names}")
    print(f"  - Matrix file: {output_file}")
    if llm_csv_file:
        print(f"  - LLM results CSV: {llm_csv_file}")


if __name__ == "__main__":
    main()
