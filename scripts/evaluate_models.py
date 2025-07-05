#!/usr/bin/env python3
"""
Evaluate different LLM models for ingredient rationalization.
"""

import argparse
import csv
import datetime
import sys
from typing import List

from cocktail_utils.ingredients import IngredientParser


def find_unmatched_ingredients(parser: IngredientParser, n: int) -> List[str]:
    """
    Find N ingredients that are not matched by the dictionary lookup.
    """
    print("Searching for ingredients not matched by dictionary...")
    unmatched = []
    # Get all ingredients, even those in just one recipe
    all_ingredients = parser.get_ingredients_from_db(min_recipe_count=1)

    for ing_usage in all_ingredients:
        if len(unmatched) >= n:
            break
        if parser.rationalize_ingredient(ing_usage.ingredient_name) is None:
            unmatched.append(ing_usage.ingredient_name)

    print(f"Found {len(unmatched)} ingredients to test.")
    return unmatched


def evaluate_models(
    ingredients: List[str],
    model_ids: List[str],
    db_path: str,
    output_file: str,
):
    """
    Submit ingredients to a list of models and write results to CSV file.
    """
    parser = IngredientParser(db_path=db_path)

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["input_ingredient"]
        for model_id in model_ids:
            header.append(f"{model_id}_output")
        writer.writerow(header)

        for i, ingredient in enumerate(ingredients):
            print(f"Processing ingredient {i + 1}/{len(ingredients)}: {ingredient}")
            row = [ingredient]
            for model_id in model_ids:
                match = parser.llm_lookup(ingredient, model_id)
                if match:
                    output = f"brand: {match.brand}, specific_type: {match.specific_type}, category: {match.category}"
                else:
                    output = "ERROR"
                row.append(output)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM models for ingredient rationalization."
    )
    parser.add_argument(
        "-n",
        "--num-ingredients",
        type=int,
        default=20,
        help="Number of ingredients to test.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["anthropic.claude-3-5-haiku-20241022-v1:0", "amazon.nova-lite-v1:0"],
        help="List of Bedrock model IDs to evaluate.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/punch_recipes.db",
        help="Path to the SQLite database.",
    )

    args = parser.parse_args()

    # Create a parser to find unmatched ingredients
    initial_parser = IngredientParser(
        db_path=args.db_path,
    )

    unmatched = find_unmatched_ingredients(initial_parser, args.num_ingredients)
    # Write unmatched ingredients to a file for inspection
    output_file = (
        f"unmatched_ingredients_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(output_file, "w") as f:
        f.write("Unmatched ingredients for LLM evaluation:\n")
        f.write("=" * 50 + "\n\n")
        for i, ingredient in enumerate(unmatched, 1):
            f.write(f"{i:3d}. {ingredient}\n")

    print(f"Wrote {len(unmatched)} unmatched ingredients to {output_file}")

    if not unmatched:
        print("Could not find any ingredients that failed the dictionary lookup.")
        return

    # Generate output CSV filename
    csv_output_file = (
        f"model_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    evaluate_models(unmatched, args.models, args.db_path, csv_output_file)
    print(f"Model evaluation results written to {csv_output_file}")


if __name__ == "__main__":
    main()
