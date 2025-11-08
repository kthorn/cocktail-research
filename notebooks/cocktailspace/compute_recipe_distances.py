#!/usr/bin/env python3
"""
Compute Distance Matrices for Recipe Subsets

This script calculates both weighted (substitution-aware) and unweighted (standard Manhattan)
distance matrices for a subset of recipes specified by name, and exports them to CSV files.

Usage:
    python compute_recipe_distances.py --recipes "Negroni,Manhattan,Martini" --output-prefix subset
    python compute_recipe_distances.py --recipe-file recipes.txt --output-prefix my_recipes
    python compute_recipe_distances.py --all --output-prefix all_recipes
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional

from distance_calculations import (
    build_recipe_ingredient_matrix,
    calculate_weighted_ingredient_distance,
    calculate_manhattan_distance,
)


# Configuration
DB_PATH = "backup-2025-10-17_08-00-45.db"


def load_recipes_from_db(
    db_path: str, recipe_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load recipes and their ingredients from SQLite database.

    Args:
        db_path: Path to SQLite database
        recipe_names: Optional list of recipe names to filter by

    Returns:
        DataFrame with recipe and ingredient information
    """
    conn = sqlite3.connect(db_path)

    if recipe_names:
        # Create parameterized query for recipe name filtering
        placeholders = ",".join(["?" for _ in recipe_names])
        query = f"""
        SELECT
            r.id as recipe_id,
            r.name as recipe_name,
            i.id as ingredient_id,
            i.name as ingredient_name,
            i.path as ingredient_path,
            i.substitution_level,
            ri.amount,
            ri.unit_id,
            u.conversion_to_ml
        FROM recipes r
        JOIN recipe_ingredients ri ON r.id = ri.recipe_id
        JOIN ingredients i ON ri.ingredient_id = i.id
        LEFT JOIN units u ON ri.unit_id = u.id
        WHERE r.name IN ({placeholders})
        ORDER BY r.id, i.id
        """
        df = pd.read_sql_query(query, conn, params=recipe_names)
    else:
        # Load all recipes
        query = """
        SELECT
            r.id as recipe_id,
            r.name as recipe_name,
            i.id as ingredient_id,
            i.name as ingredient_name,
            i.path as ingredient_path,
            i.substitution_level,
            ri.amount,
            ri.unit_id,
            u.conversion_to_ml
        FROM recipes r
        JOIN recipe_ingredients ri ON r.id = ri.recipe_id
        JOIN ingredients i ON ri.ingredient_id = i.id
        LEFT JOIN units u ON ri.unit_id = u.id
        ORDER BY r.id, i.id
        """
        df = pd.read_sql_query(query, conn)

    conn.close()

    if df.empty:
        raise ValueError("No recipes found matching the specified names")

    return df


def load_recipe_names_from_file(filepath: str) -> List[str]:
    """Load recipe names from a text file (one per line).

    Args:
        filepath: Path to text file with recipe names

    Returns:
        List of recipe names
    """
    with open(filepath, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def compute_and_save_distance_matrices(
    recipes_df: pd.DataFrame,
    output_prefix: str = "distance",
    batch_size: int = 100,
    substitution_weight: float = 0.5,
) -> None:
    """Compute both weighted and unweighted distance matrices and save to CSV.

    Args:
        recipes_df: DataFrame with recipe and ingredient information
        output_prefix: Prefix for output CSV files
        batch_size: Batch size for weighted distance calculation
        substitution_weight: Weight applied to substitutable ingredients (0.0 = no penalty, 1.0 = full penalty)
    """
    print(f"Processing {recipes_df['recipe_id'].nunique()} recipes...")

    # Build normalized recipe-ingredient matrix
    print("\nBuilding recipe-ingredient matrix...")
    normalized_matrix, _ = build_recipe_ingredient_matrix(recipes_df)
    print(
        f"Matrix shape: {normalized_matrix.shape[0]} recipes x {normalized_matrix.shape[1]} ingredients"
    )
    normalized_matrix.to_csv("normalized_ingredients_matrix.csv")

    # Calculate unweighted Manhattan distance
    print("\nCalculating standard Manhattan distance...")
    manhattan_df = calculate_manhattan_distance(normalized_matrix)

    # Calculate weighted Manhattan distance (substitution-aware)
    print(
        f"\nCalculating weighted Manhattan distance (substitution-aware, weight={substitution_weight})..."
    )
    weighted_manhattan_df = calculate_weighted_ingredient_distance(
        recipes_df,
        normalized_matrix=normalized_matrix,
        batch_size=batch_size,
        substitution_weight=substitution_weight,
    )

    # Save to CSV
    manhattan_output = f"{output_prefix}_manhattan.csv"
    weighted_output = f"{output_prefix}_weighted_manhattan.csv"

    print(f"\nSaving distance matrices...")
    manhattan_df.to_csv(manhattan_output)
    print(f"  Standard Manhattan distance saved to: {manhattan_output}")

    weighted_manhattan_df.to_csv(weighted_output)
    print(f"  Weighted Manhattan distance saved to: {weighted_output}")

    # Print statistics
    print("\n=== Distance Matrix Statistics ===")
    print("\nStandard Manhattan Distance:")
    print(f"  Shape: {manhattan_df.shape}")
    print(f"  Min:   {manhattan_df.min().min():.4f}")
    print(f"  Max:   {manhattan_df.max().max():.4f}")
    print(f"  Mean:  {manhattan_df.mean().mean():.4f}")

    print("\nWeighted Manhattan Distance (Substitution-Aware):")
    print(f"  Shape: {weighted_manhattan_df.shape}")
    print(f"  Min:   {weighted_manhattan_df.min().min():.4f}")
    print(f"  Max:   {weighted_manhattan_df.max().max():.4f}")
    print(f"  Mean:  {weighted_manhattan_df.mean().mean():.4f}")

    # Show sample of both matrices
    print("\n=== Sample (First 5x5) ===")
    print("\nStandard Manhattan Distance:")
    print(manhattan_df.iloc[:5, :5].round(4))

    print("\nWeighted Manhattan Distance:")
    print(weighted_manhattan_df.iloc[:5, :5].round(4))


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Compute distance matrices for recipe subsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute distances for specific recipes (comma-separated)
  python compute_recipe_distances.py --recipes "Negroni,Manhattan,Martini"

  # Load recipe names from a file
  python compute_recipe_distances.py --recipe-file my_recipes.txt

  # Compute distances for all recipes in the database
  python compute_recipe_distances.py --all

  # Specify custom output prefix and database path
  python compute_recipe_distances.py --recipes "Negroni,Manhattan" --output-prefix cocktails --db my_db.db
  
  # Control substitutability penalty (0.0 = treat substitutable ingredients as identical)
  python compute_recipe_distances.py --recipes "Manhattan,Jeremy Oertel's Manhattan" --substitution-weight 0.0
  
  # Full penalty for substitutable ingredients (no weighting, like unweighted distance)
  python compute_recipe_distances.py --recipes "Manhattan,Jeremy Oertel's Manhattan" --substitution-weight 1.0
        """,
    )

    parser.add_argument(
        "--recipes",
        type=str,
        help="Comma-separated list of recipe names to analyze",
    )
    parser.add_argument(
        "--recipe-file",
        type=str,
        help="Path to text file containing recipe names (one per line)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compute distances for all recipes in the database",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="distance",
        help="Prefix for output CSV files (default: distance)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DB_PATH,
        help=f"Path to SQLite database (default: {DB_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for weighted distance calculation (default: 100)",
    )
    parser.add_argument(
        "--substitution-weight",
        type=float,
        default=0.5,
        help="Weight for substitutable ingredients (0.0=no penalty, 1.0=full penalty, default: 0.5)",
    )

    args = parser.parse_args()

    # Validate input arguments
    input_methods = sum([bool(args.recipes), bool(args.recipe_file), args.all])
    if input_methods == 0:
        parser.error("Must specify one of: --recipes, --recipe-file, or --all")
    elif input_methods > 1:
        parser.error("Cannot specify multiple input methods simultaneously")

    # Load recipe names
    recipe_names = None
    if args.recipes:
        recipe_names = [name.strip() for name in args.recipes.split(",")]
        print(f"Computing distances for {len(recipe_names)} recipes: {recipe_names}")
    elif args.recipe_file:
        recipe_names = load_recipe_names_from_file(args.recipe_file)
        print(f"Loaded {len(recipe_names)} recipe names from {args.recipe_file}")
    elif args.all:
        print("Computing distances for ALL recipes in the database")

    # Check if database exists
    if not Path(args.db).exists():
        raise FileNotFoundError(f"Database file not found: {args.db}")

    # Load recipe data
    print(f"\nLoading recipe data from database: {args.db}")
    recipes_df = load_recipes_from_db(args.db, recipe_names)

    # Show which recipes were found
    found_recipes = sorted(recipes_df["recipe_name"].unique())
    print(f"\nFound {len(found_recipes)} recipes:")
    for recipe in found_recipes[:10]:
        print(f"  - {recipe}")
    if len(found_recipes) > 10:
        print(f"  ... and {len(found_recipes) - 10} more")

    # Compute and save distance matrices
    compute_and_save_distance_matrices(
        recipes_df,
        output_prefix=args.output_prefix,
        batch_size=args.batch_size,
        substitution_weight=args.substitution_weight,
    )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
