#!/usr/bin/env python3
"""
Batch recipe rationalization script.

Processes recipes from HTML files and automatically identifies those that can be
fully rationalized (all ingredients mapped) without human intervention. Creates
batch files for upload and tracks progress in validation_log.json.

Usage:
    python batch_rationalize_recipes.py --source punch
    python batch_rationalize_recipes.py --source diffords --batch-size 50
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

# Add cocktail-utils to path
sys.path.insert(0, "/home/kurtt/cocktail-research/cocktail-utils/src")

from cocktail_utils.recipes import get_recipe_source


class BatchRecipeRationalizer:
    def __init__(
        self,
        source_name: str,
        mappings_file: str = "/home/kurtt/cocktail-research/recipe_ingest/ingredient_rationalizer/ingredient_mappings.json",
        validation_log_file: str = "/home/kurtt/cocktail-research/data/validation_log.json",
        output_dir: str = "/home/kurtt/cocktail-research/output",
        batch_size: int = 100,
    ):
        self.source_name = source_name
        self.recipe_source = get_recipe_source(source_name)

        if not self.recipe_source:
            raise ValueError(f"Unknown recipe source: {source_name}")

        self.mappings_file = mappings_file
        self.validation_log_file = validation_log_file
        self.output_dir = output_dir
        self.batch_size = batch_size

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load data
        self.load_ingredient_mappings()
        self.load_validation_log()
        self.load_html_files()

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
                all_progress = json.load(f)

            # Get progress for this source
            if self.source_name in all_progress:
                self.validation_log = all_progress[self.source_name]
            else:
                # Initialize empty progress for new source
                self.validation_log = {
                    "current_index": 0,
                    "accepted": [],
                    "rejected": [],
                    "auto_ingested": [],
                    "needs_review": [],
                }
                all_progress[self.source_name] = self.validation_log
                with open(self.validation_log_file, "w") as f:
                    json.dump(all_progress, f, indent=2)
        else:
            self.validation_log = {
                "current_index": 0,
                "accepted": [],
                "rejected": [],
                "auto_ingested": [],
                "needs_review": [],
            }

    def save_validation_log(self):
        """Save validation progress log."""
        # Load entire file, update our source, save back
        with open(self.validation_log_file, "r") as f:
            all_progress = json.load(f)

        all_progress[self.source_name] = self.validation_log

        with open(self.validation_log_file, "w") as f:
            json.dump(all_progress, f, indent=2)

    def load_html_files(self):
        """Load HTML files from recipe source."""
        self.html_files = self.recipe_source.find_html_files()
        print(f"Loaded {len(self.html_files)} HTML files for {self.source_name}")

    def is_recipe_already_processed(self, recipe_name: str) -> bool:
        """Check if recipe has already been processed."""
        return (
            recipe_name in self.validation_log.get("accepted", [])
            or recipe_name in self.validation_log.get("rejected", [])
            or recipe_name in self.validation_log.get("auto_ingested", [])
        )

    def parse_recipe_from_html(self, html_file: str) -> Dict:
        """Parse recipe from HTML file.

        Args:
            html_file: Path to HTML file

        Returns:
            Parsed recipe dict or None if parsing fails
        """
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            return self.recipe_source.parse_recipe_from_html(html_content, html_file)
        except Exception as e:
            print(f"Error reading/parsing {html_file}: {e}")
            return None

    def can_rationalize_recipe(
        self, parsed_recipe: Dict
    ) -> Tuple[bool, List[str]]:
        """
        Check if all ingredients in a recipe can be rationalized.

        Args:
            parsed_recipe: Parsed recipe dict with ingredients

        Returns:
            (can_rationalize, unmapped_ingredients)
        """
        unmapped = []

        for ingredient in parsed_recipe["ingredients"]:
            ingredient_name = ingredient["ingredient_name"].strip()
            if not ingredient_name:
                continue

            # Skip ingredients marked as optional
            ingredient_name_lower = ingredient_name.lower()
            if "optional" in ingredient_name_lower or "omit if" in ingredient_name_lower:
                continue

            # Check if ingredient exists in mappings
            if ingredient_name not in self.mappings:
                unmapped.append(ingredient_name)

        return len(unmapped) == 0, unmapped

    def rationalize_recipe(self, parsed_recipe: Dict) -> Dict:
        """Rationalize ingredients in a parsed recipe.

        Args:
            parsed_recipe: Parsed recipe dict

        Returns:
            Rationalized recipe dict ready for output
        """
        rationalized_ingredients = []

        for ingredient in parsed_recipe["ingredients"]:
            ingredient_name = ingredient["ingredient_name"].strip()

            # Skip ingredients marked as optional
            ingredient_name_lower = ingredient_name.lower()
            if "optional" in ingredient_name_lower or "omit if" in ingredient_name_lower:
                print(f"    Filtering out optional ingredient: '{ingredient_name}'")
                continue

            # Replace with mapped name if available
            if ingredient_name in self.mappings:
                mapping = self.mappings[ingredient_name]
                rationalized_ingredients.append({
                    "ingredient_name": mapping.get("name", ingredient_name),
                    "amount": ingredient.get("amount", ""),
                    "unit_name": ingredient.get("unit_name", ""),
                })
            else:
                rationalized_ingredients.append(ingredient)

        # Use extracted source_url if available, otherwise derive it
        if "source_url" in parsed_recipe and parsed_recipe["source_url"]:
            source_url = parsed_recipe["source_url"]
        else:
            source_url = self.recipe_source.derive_source_url(parsed_recipe["name"])

        result = {
            "name": parsed_recipe["name"],
            "description": "",
            "instructions": "Shake all ingredients with ice and strain into a cocktail or coupe glass",
            "ingredients": rationalized_ingredients,
            "source_url": source_url,
        }

        # Include source field if present
        if "source" in parsed_recipe:
            result["source"] = parsed_recipe["source"]

        return result

    def process_recipes(self):
        """Process all recipes and create batches."""
        auto_ingested = []
        needs_review = []

        print("\\nProcessing recipes...")

        for html_file in self.html_files:
            # Parse recipe from HTML
            parsed_recipe = self.parse_recipe_from_html(html_file)

            if not parsed_recipe:
                print(f"  Skipping {html_file} - parsing failed")
                continue

            recipe_name = parsed_recipe["name"]

            # Skip if already processed
            if self.is_recipe_already_processed(recipe_name):
                continue

            # Skip recipes we don't want to process
            # 1. Skip shot recipes (SHOT GLASS in instructions or "Shot" in name)
            instructions = parsed_recipe.get("instructions", "")
            if "SHOT GLASS" in instructions.upper():
                print(f"  Skipping '{recipe_name}' - contains SHOT GLASS")
                continue

            if "Shot" in recipe_name:
                print(f"  Skipping '{recipe_name}' - has 'Shot' in name")
                continue

            # 2. Skip recipes with 2 or fewer ingredients
            ingredients = parsed_recipe.get("ingredients", [])
            if len(ingredients) <= 2:
                print(f"  Skipping '{recipe_name}' - has {len(ingredients)} ingredient(s)")
                continue

            # Check if all ingredients can be rationalized
            can_rationalize, unmapped = self.can_rationalize_recipe(parsed_recipe)

            if can_rationalize:
                # Add to auto-ingest batch
                rationalized_recipe = self.rationalize_recipe(parsed_recipe)
                auto_ingested.append(rationalized_recipe)
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
            print("\\nNo recipes to batch.")
            return []

        batch_files = []
        total_batches = (len(recipes) + self.batch_size - 1) // self.batch_size

        print(f"\\nCreating {total_batches} batch file(s)...")

        for i in range(0, len(recipes), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch = recipes[i : i + self.batch_size]

            batch_data = {"recipes": batch}

            # Create filename with source and zero-padded batch number
            filename = f"rationalized-recipes-{self.source_name}-batch-{batch_num:03d}.json"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, "w") as f:
                json.dump(batch_data, f, indent=2)

            batch_files.append(filepath)
            print(f"  Created {filename} with {len(batch)} recipes")

        return batch_files

    def run(self):
        """Run the batch rationalization process."""
        print("=" * 70)
        print(f"Batch Recipe Rationalization - {self.source_name.upper()}")
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
        print("\\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Auto-ingested recipes: {len(auto_ingested)}")
        print(f"Needs review: {len(needs_review)}")
        print(f"Batch files created: {len(batch_files)}")

        if needs_review:
            print("\\nTop unmapped ingredients:")
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
        "--source",
        type=str,
        required=True,
        choices=["punch", "diffords"],
        help="Recipe source to process (punch or diffords)",
    )
    parser.add_argument(
        "--mappings-file",
        type=str,
        default="/home/kurtt/cocktail-research/recipe_ingest/ingredient_rationalizer/ingredient_mappings.json",
        help="Path to ingredient mappings file",
    )
    parser.add_argument(
        "--validation-log",
        type=str,
        default="/home/kurtt/cocktail-research/data/validation_log.json",
        help="Path to validation log file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/kurtt/cocktail-research/output",
        help="Directory to write batch files",
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
            source_name=args.source,
            mappings_file=args.mappings_file,
            validation_log_file=args.validation_log,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

        rationalizer.run()

    except Exception as e:
        print(f"\\nError: {e}")
        raise


if __name__ == "__main__":
    main()
