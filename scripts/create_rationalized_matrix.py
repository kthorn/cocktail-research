#!/usr/bin/env python3
"""
Create a rationalized recipe-ingredient matrix using dictionary lookup first,
then Claude Haiku for unmatched ingredients. Outputs to parquet with multi-indexed columns.
"""

import argparse
import datetime
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
from cocktail_utils.database import get_recipe_ingredient_data
from cocktail_utils.ingredients import IngredientParser
from tqdm import tqdm


def normalize_string(value, default: str = "unknown") -> str:
    """Normalize values to non-empty strings."""
    if value is None:
        return default
    elif isinstance(value, list):
        return value[0] if value else default
    else:
        return str(value).strip() if str(value).strip() else default


def find_most_recent_rationalized_csv() -> Optional[str]:
    csv_files = glob.glob("llm_rationalized_ingredients_*.csv")

    if not csv_files:
        return None

    # Sort by modification time, most recent first
    csv_files.sort(key=os.path.getmtime, reverse=True)
    most_recent = csv_files[0]

    print(f"Found most recent rationalized CSV: {most_recent}")
    return most_recent


def load_previous_rationalizations(csv_file: str) -> Dict[str, dict]:
    """Load previously rationalized ingredients from CSV file."""
    if not os.path.exists(csv_file):
        return {}

    print(f"Loading previous rationalizations from {csv_file}...")

    try:
        # Use pandas for faster CSV reading
        df = pd.read_csv(csv_file)

        previous_rationalizations = {}
        for _, row in df.iterrows():
            ingredient_name = row["original_ingredient"]
            previous_rationalizations[ingredient_name] = {
                "category": row["category"],
                "specific_type": row["specific_type"]
                if pd.notna(row["specific_type"])
                else None,
                "brand": row["brand"] if pd.notna(row["brand"]) else None,
                "confidence": float(row["confidence"])
                if pd.notna(row["confidence"])
                else 0.0,
                "source": row["source"],
            }

        print(f"Loaded {len(previous_rationalizations)} previous rationalizations")
        return previous_rationalizations

    except Exception as e:
        print(f"Error loading previous rationalizations: {e}")
        return {}


def write_rationalized_csv(
    rationalizations: Dict[str, Optional[dict]], output_file: str
) -> None:
    """
    Write all rationalized ingredients (both previous and new) to CSV file.

    Args:
        all_rationalizations: Dictionary mapping ingredient names to rationalization results
        output_file: Path to output CSV file
    """
    successful_results = {
        ingredient: result
        for ingredient, result in rationalizations.items()
        if result is not None
    }

    if not successful_results:
        print("No successful rationalizations to write.")
        return

    # Use pandas for faster CSV writing
    data = []
    for ingredient, result in successful_results.items():
        data.append(
            {
                "original_ingredient": ingredient,
                "category": result["category"],
                "specific_type": result["specific_type"] or "",
                "brand": result["brand"] or "",
                "confidence": result["confidence"],
                "source": result["source"],
            }
        )

    df = pd.DataFrame(data)
    df.sort_values("original_ingredient", inplace=True)
    df.to_csv(output_file, index=False)

    print(f"Wrote {len(successful_results)} rationalized ingredients to {output_file}")


def llm_rationalize(
    parser: IngredientParser,
    unmatched_ingredients: List[str],
    model_id: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
    max_concurrent: int = 10,
) -> Dict[str, Optional[dict]]:
    """
    Batch process unmatched ingredients using LLM with parallel processing.

    Args:
        parser: IngredientParser instance
        unmatched_ingredients: List of ingredient names to rationalize
        model_id: Bedrock model ID to use
        max_concurrent: Maximum number of parallel workers

    Returns:
        Dictionary mapping ingredient names to rationalization results
    """
    print(
        f"Rationalizing {len(unmatched_ingredients)} unmatched ingredients using {model_id}..."
    )

    results = {}

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit all tasks at once for better efficiency
        future_to_ingredient = {
            executor.submit(parser.llm_lookup, ingredient, model_id): ingredient
            for ingredient in unmatched_ingredients
        }

        # Process completions
        with tqdm(total=len(unmatched_ingredients), desc="LLM rationalization") as pbar:
            for future in as_completed(future_to_ingredient):
                ingredient = future_to_ingredient[future]
                try:
                    match = future.result()
                    results[ingredient] = {
                        "brand": match.brand,
                        "specific_type": match.specific_type,
                        "category": match.category,
                        "confidence": match.confidence,
                        "source": match.source,
                    }
                except Exception as e:
                    print(f"Error processing {ingredient}: {e}")
                    results[ingredient] = None

                pbar.update(1)

    successful_matches = sum(1 for v in results.values() if v is not None)
    print(
        f"Successfully rationalized {successful_matches}/{len(unmatched_ingredients)} ingredients via LLM"
    )

    return results


def prepare_rationalized_dataframe(
    recipe_ingredient_df: pd.DataFrame,
    ingredient_rationalizations: Dict[str, dict],
) -> pd.DataFrame:
    """Prepare dataframe with rationalized ingredient information."""
    print("Preparing rationalized dataframe...")

    # Filter to only include recipes where ALL ingredients are rationalized
    valid_ingredients = set(ingredient_rationalizations.keys())
    recipe_ingredient_counts = recipe_ingredient_df.groupby("recipe_name")[
        "ingredient_name"
    ].apply(set)
    valid_recipes = recipe_ingredient_counts[
        recipe_ingredient_counts.apply(lambda x: x.issubset(valid_ingredients))
    ].index

    filtered_df = recipe_ingredient_df[
        recipe_ingredient_df["recipe_name"].isin(valid_recipes)
    ].copy()

    print(
        f"Kept {len(valid_recipes)} recipes out of {recipe_ingredient_df['recipe_name'].nunique()} original recipes"
    )

    # Create rationalization mappings
    category_map = {}
    specific_type_map = {}
    brand_map = {}

    for ingredient, rationalization in ingredient_rationalizations.items():
        category_map[ingredient] = normalize_string(rationalization["category"])
        specific_type_map[ingredient] = normalize_string(
            rationalization["specific_type"], "generic"
        )
        brand_map[ingredient] = normalize_string(rationalization["brand"], "generic")

    # Apply mappings vectorized
    filtered_df["category"] = filtered_df["ingredient_name"].map(category_map)
    filtered_df["specific_type"] = filtered_df["ingredient_name"].map(specific_type_map)
    filtered_df["brand"] = filtered_df["ingredient_name"].map(brand_map)

    # Handle missing amounts
    filtered_df["amount_ml"] = filtered_df["amount_ml"].fillna(0)

    return filtered_df


def create_rationalized_matrix(
    recipe_ingredient_df: pd.DataFrame,
    ingredient_rationalizations: Dict[str, dict],
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


def collect_all_rationalizations(
    parser: IngredientParser,
    all_ingredients: List,
    model_id: str,
    max_workers: int,
    use_llm: bool = True,
) -> Dict[str, dict]:
    """Collect all ingredient rationalizations from dictionary, previous CSV, and LLM."""
    print("Collecting ingredient rationalizations...")

    # Step 1: Dictionary-based rationalization
    print("Step 1: Dictionary-based ingredient rationalization...")
    matched_ingredients = {}
    unmatched_ingredients = []

    for ing_usage in tqdm(all_ingredients, desc="Dictionary lookup"):
        ingredient_name = ing_usage.ingredient_name
        match = parser.rationalize_ingredient(ingredient_name)

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

    # Step 2: Load previous rationalizations
    print("Step 2: Loading previous rationalizations...")
    previous_csv = find_most_recent_rationalized_csv()
    previous_rationalizations = (
        load_previous_rationalizations(previous_csv) if previous_csv else {}
    )
    still_unmatched = []
    for ingredient in unmatched_ingredients:
        if ingredient in previous_rationalizations:
            matched_ingredients[ingredient] = previous_rationalizations[ingredient]
        else:
            still_unmatched.append(ingredient)
    print(
        f"Previously rationalized: {len(unmatched_ingredients) - len(still_unmatched)}"
    )
    print(f"Still unmatched: {len(still_unmatched)}")

    if use_llm and still_unmatched:
        # Step 3: LLM rationalization
        llm_results = {}
        print("Step 3: LLM-based rationalization for unmatched ingredients...")
        llm_results = llm_rationalize(parser, still_unmatched, model_id, max_workers)

        # Add successful LLM results to matched ingredients
        for ingredient, result in llm_results.items():
            if result:
                matched_ingredients[ingredient] = result

        # Write combined results to CSV
        csv_file = None
        rationalized_to_write = {**previous_rationalizations}
        rationalized_to_write.update({k: v for k, v in llm_results.items() if v})

        if rationalized_to_write:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = f"llm_rationalized_ingredients_{timestamp}.csv"
            write_rationalized_csv(rationalized_to_write, csv_file)

    total_rationalized = len(matched_ingredients)
    total_ingredients = len(all_ingredients)
    print(
        f"Total rationalized: {total_rationalized}/{total_ingredients} ({total_rationalized / total_ingredients * 100:.1f}%)"
    )
    return matched_ingredients


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
    args = parser.parse_args()

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
    print("Fetching recipe-ingredient relationships...")
    recipe_ingredient_df = get_recipe_ingredient_data(args.db_path)

    if recipe_ingredient_df.empty:
        print("No recipe-ingredient data found. Exiting.")
        return

    # Create the rationalized matrix
    for matrix_type in ["amount", "boolean"]:
        matrix_df = create_rationalized_matrix(
            recipe_ingredient_df, matched_ingredients, matrix_type=matrix_type
        )
        if matrix_df.empty:
            print(f"No valid {matrix_type} matrix created. Exiting.")
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/rationalized_matrix_{matrix_type}_{timestamp}.parquet"
        matrix_df.to_parquet(output_file, index=True)

    print("Successfully created rationalized matrices:")
    print(f"  - {len(matrix_df)} recipes")
    print(f"  - {len(matrix_df.columns)} ingredient columns")


if __name__ == "__main__":
    main()
