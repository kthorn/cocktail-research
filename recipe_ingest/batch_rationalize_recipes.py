#!/usr/bin/env python3
"""
Batch recipe rationalization script.

Processes recipes from the database and automatically identifies those that can be
fully rationalized (all ingredients mapped) without human intervention. Creates
batch files for upload and tracks progress in validation_log.json.
"""

import argparse
import glob
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests


def find_most_recent_raw_ingredients_file(
    data_dir="/home/kurtt/cocktail-research/data",
):
    """Find the most recent raw ingredients parquet file based on timestamp."""
    pattern = os.path.join(data_dir, "raw_recipe_ingredients_*.parquet")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No raw ingredients files found matching pattern: {pattern}"
        )

    # Extract timestamps and find the most recent
    file_timestamps = []
    for file_path in files:
        filename = os.path.basename(file_path)
        match = re.search(r"raw_recipe_ingredients_(\d{8}_\d{6})\.parquet", filename)
        if match:
            timestamp_str = match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                file_timestamps.append((timestamp, file_path))
            except ValueError:
                print(f"Warning: Could not parse timestamp from {filename}")
                continue

    if not file_timestamps:
        raise ValueError("No valid timestamp files found")

    file_timestamps.sort(key=lambda x: x[0], reverse=True)
    most_recent_file = file_timestamps[0][1]

    print(f"Using most recent raw ingredients file: {most_recent_file}")
    return most_recent_file


class BatchRecipeRationalizer:
    def __init__(
        self,
        db_path="/home/kurtt/cocktail-research/data/recipes.db",
        mappings_file="/home/kurtt/cocktail-research/recipe_ingest/ingredient_rationalizer/ingredient_mappings.json",
        validation_log_file="/home/kurtt/cocktail-research/data/validation_log.json",
        output_dir="/home/kurtt/cocktail-research/output",
        batch_size=100,
    ):
        self.db_path = db_path
        self.mappings_file = mappings_file
        self.validation_log_file = validation_log_file
        self.output_dir = output_dir
        self.batch_size = batch_size

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load data
        self.load_ingredient_mappings()
        self.load_validation_log()
        self.load_raw_ingredients_data()
        self.load_database_recipes()

    def load_ingredient_mappings(self):
        """Load ingredient mappings from file."""
        if os.path.exists(self.mappings_file):
            try:
                with open(self.mappings_file, "r") as f:
                    self.mappings = json.load(f)
                print(f"Loaded {len(self.mappings)} ingredient mappings")
            except Exception as e:
                print(f"Error loading mappings: {e}")
                self.mappings = {}
        else:
            print(f"Warning: Mappings file not found at {self.mappings_file}")
            self.mappings = {}

    def load_validation_log(self):
        """Load validation progress log."""
        if os.path.exists(self.validation_log_file):
            with open(self.validation_log_file, "r") as f:
                self.validation_log = json.load(f)
        else:
            self.validation_log = {
                "current_index": 0,
                "reviewed": [],
                "accepted": [],
                "rejected": [],
            }

        # Ensure auto_ingested field exists
        if "auto_ingested" not in self.validation_log:
            self.validation_log["auto_ingested"] = []

        # Ensure needs_review field exists
        if "needs_review" not in self.validation_log:
            self.validation_log["needs_review"] = []

    def save_validation_log(self):
        """Save validation progress log."""
        with open(self.validation_log_file, "w") as f:
            json.dump(self.validation_log, f, indent=2)

    def load_raw_ingredients_data(self):
        """Load raw ingredients data from parquet file."""
        try:
            raw_ingredients_file = find_most_recent_raw_ingredients_file()
            self.raw_ingredients_df = pd.read_parquet(raw_ingredients_file)
            print(
                f"Loaded raw ingredients data: {self.raw_ingredients_df.shape[0]} rows"
            )
        except Exception as e:
            print(f"Error loading raw ingredients data: {e}")
            self.raw_ingredients_df = None
            raise

    def load_database_recipes(self):
        """Load recipe metadata from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, source_file FROM recipe")

            self.recipes = {}
            for row in cursor.fetchall():
                recipe_id, name, source_file = row
                self.recipes[recipe_id] = {
                    "id": recipe_id,
                    "name": name,
                    "source_file": source_file,
                }
            conn.close()
            print(f"Loaded {len(self.recipes)} recipes from database")
        except Exception as e:
            print(f"Error loading recipes from database: {e}")
            self.recipes = {}
            raise

    def is_recipe_already_processed(self, recipe_name: str) -> bool:
        """Check if recipe has already been processed."""
        return (
            recipe_name in self.validation_log.get("accepted", [])
            or recipe_name in self.validation_log.get("rejected", [])
            or recipe_name in self.validation_log.get("auto_ingested", [])
        )

    def get_recipe_ingredients(self, recipe_id: int) -> List[Dict]:
        """Get ingredients for a recipe from parquet data."""
        if self.raw_ingredients_df is None:
            return []

        recipe_ingredients = self.raw_ingredients_df[
            self.raw_ingredients_df["recipe_id"] == recipe_id
        ]

        ingredients = []
        for _, row in recipe_ingredients.iterrows():
            ingredients.append(
                {
                    "ingredient_name": row["ingredient_name"],
                    "amount": row.get("original_amount", 0),
                    "unit_name": row.get("original_unit", ""),
                }
            )

        return ingredients

    def can_rationalize_recipe(self, ingredients: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Check if all ingredients in a recipe can be rationalized.

        Returns:
            (can_rationalize, unmapped_ingredients)
        """
        unmapped = []

        for ingredient in ingredients:
            ingredient_name = ingredient["ingredient_name"].strip()
            if not ingredient_name:
                continue

            # Check if ingredient exists in mappings
            if ingredient_name not in self.mappings:
                unmapped.append(ingredient_name)

        return len(unmapped) == 0, unmapped

    def rationalize_ingredients(self, ingredients: List[Dict]) -> List[Dict]:
        """Replace ingredient names with mapped API names."""
        rationalized = []

        for ingredient in ingredients:
            rationalized_ing = ingredient.copy()
            ingredient_name = ingredient["ingredient_name"].strip()

            # Replace with mapped name if available
            if ingredient_name in self.mappings:
                mapping = self.mappings[ingredient_name]
                rationalized_ing["ingredient_name"] = mapping.get("name")

            rationalized.append(rationalized_ing)

        return rationalized

    def derive_source_url(self, recipe_name: str) -> str:
        """Derive Punch source URL from recipe name."""
        clean_name = recipe_name.lower()

        # Replace unicode characters
        unicode_replacements = {
            "\u2019": "'",
            "\u2018": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u00a0": " ",
        }
        for unicode_char, replacement in unicode_replacements.items():
            clean_name = clean_name.replace(unicode_char, replacement)

        # Remove special characters
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in " -&")
        clean_name = clean_name.replace(" ", "-").replace("&", "and")
        clean_name = "-".join(filter(None, clean_name.split("-")))

        return f"https://punchdrink.com/recipes/{clean_name}/"

    def format_recipe_for_output(self, recipe_id: int, recipe_name: str) -> Dict:
        """Format recipe in the output format."""
        ingredients = self.get_recipe_ingredients(recipe_id)
        rationalized_ingredients = self.rationalize_ingredients(ingredients)

        return {
            "name": recipe_name,
            "description": "",
            "instructions": "Shake all ingredients with ice and strain into a cocktail or coupe glass",
            "ingredients": rationalized_ingredients,
            "source_url": self.derive_source_url(recipe_name),
        }

    def process_recipes(self):
        """Process all recipes and create batches."""
        auto_ingested = []
        needs_review = []

        print("\nProcessing recipes...")

        for recipe_id, recipe_data in self.recipes.items():
            recipe_name = recipe_data["name"]

            # Skip if already processed
            if self.is_recipe_already_processed(recipe_name):
                continue

            # Get ingredients
            ingredients = self.get_recipe_ingredients(recipe_id)

            if not ingredients:
                print(f"  Skipping '{recipe_name}' - no ingredients found")
                continue

            # Check if all ingredients can be rationalized
            can_rationalize, unmapped = self.can_rationalize_recipe(ingredients)

            if can_rationalize:
                # Add to auto-ingest batch
                formatted_recipe = self.format_recipe_for_output(recipe_id, recipe_name)
                auto_ingested.append(formatted_recipe)
                self.validation_log["auto_ingested"].append(recipe_name)
                print(f"  ✓ Auto-ingestible: {recipe_name}")
            else:
                # Track for later human review
                needs_review.append(
                    {"recipe_name": recipe_name, "unmapped_ingredients": unmapped}
                )
                print(
                    f"  ✗ Needs review: {recipe_name} ({len(unmapped)} unmapped ingredients)"
                )

        return auto_ingested, needs_review

    def create_batch_files(self, recipes: List[Dict]):
        """Create batch files with up to batch_size recipes each."""
        if not recipes:
            print("\nNo recipes to batch.")
            return []

        batch_files = []
        total_batches = (len(recipes) + self.batch_size - 1) // self.batch_size

        print(f"\nCreating {total_batches} batch file(s)...")

        for i in range(0, len(recipes), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch = recipes[i : i + self.batch_size]

            batch_data = {"recipes": batch}

            # Create filename with zero-padded batch number
            filename = f"rationalized-recipes-batch-{batch_num:03d}.json"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, "w") as f:
                json.dump(batch_data, f, indent=2)

            batch_files.append(filepath)
            print(f"  Created {filename} with {len(batch)} recipes")

        return batch_files

    def run(self):
        """Run the batch rationalization process."""
        print("=" * 70)
        print("Batch Recipe Rationalization")
        print("=" * 70)

        # Process recipes
        auto_ingested, needs_review = self.process_recipes()

        # Update needs_review in validation log
        self.validation_log["needs_review"] = needs_review

        # Save validation log
        self.save_validation_log()

        # Create batch files
        batch_files = self.create_batch_files(auto_ingested)

        # Print summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Auto-ingested recipes: {len(auto_ingested)}")
        print(f"Needs review: {len(needs_review)}")
        print(f"Batch files created: {len(batch_files)}")

        if needs_review:
            print("\nTop unmapped ingredients:")
            # Count frequency of unmapped ingredients
            unmapped_counts = {}
            for item in needs_review:
                for ing in item["unmapped_ingredients"]:
                    unmapped_counts[ing] = unmapped_counts.get(ing, 0) + 1

            # Show top 10
            sorted_unmapped = sorted(
                unmapped_counts.items(), key=lambda x: x[1], reverse=True
            )
            for ing, count in sorted_unmapped[:10]:
                print(f"  {ing}: {count} recipes")

        print("=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Batch rationalize recipes for auto-ingestion"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="/home/kurtt/cocktail-research/data/recipes.db",
        help="Path to the database file (default: ../data/recipes.db)",
    )
    parser.add_argument(
        "--mappings-file",
        type=str,
        default="/home/kurtt/cocktail-research/recipe_ingest/ingredient_rationalizer/ingredient_mappings.json",
        help="Path to ingredient mappings file (default: ingredient_rationalizer/ingredient_mappings.json)",
    )
    parser.add_argument(
        "--validation-log",
        type=str,
        default="/home/kurtt/cocktail-research/data/validation_log.json",
        help="Path to validation log file (default: ../data/validation_log.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/kurtt/cocktail-research/output",
        help="Directory to write batch files (default: ../output)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of recipes per batch file (default: 100)",
    )

    args = parser.parse_args()

    try:
        rationalizer = BatchRecipeRationalizer(
            db_path=args.db_path,
            mappings_file=args.mappings_file,
            validation_log_file=args.validation_log,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

        rationalizer.run()

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
