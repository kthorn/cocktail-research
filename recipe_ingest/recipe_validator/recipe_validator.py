#!/usr/bin/env python3
"""
Simple recipe validation web app.
Displays raw HTML recipes alongside rationalized data for validation.

Usage:
    python recipe_validator.py --source punch
    python recipe_validator.py --source diffords
"""

import argparse
import json
import os
import sqlite3
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Add cocktail-utils to path
import sys

sys.path.insert(0, "/home/kurtt/cocktail-research/cocktail-utils/src")

from cocktail_utils.recipes import get_recipe_source, validate_url

app = Flask(__name__)

# Global configuration
VALIDATION_LOG_FILE = "/home/kurtt/cocktail-research/data/validation_log.json"
VALIDATED_RECIPES_FILE = (
    "/home/kurtt/cocktail-research/input_data/validated-recipes.json"
)
DATABASE_FILE = "/home/kurtt/cocktail-research/data/recipes.db"
MAPPINGS_FILE = "/home/kurtt/cocktail-research/recipe_ingest/ingredient_rationalizer/ingredient_mappings.json"

# Global validator instance (set by main)
validator = None


class RecipeValidator:
    def __init__(self, source_name: str):
        """Initialize validator for a specific recipe source.

        Args:
            source_name: Name of recipe source ("punch" or "diffords")
        """
        self.source_name = source_name
        self.recipe_source = get_recipe_source(source_name)

        if not self.recipe_source:
            raise ValueError(f"Unknown recipe source: {source_name}")

        self.load_progress()
        self.load_ingredient_mappings()
        self.load_database_mapping()
        self.load_html_files()

    def load_progress(self):
        """Load validation progress from log file."""
        if os.path.exists(VALIDATION_LOG_FILE):
            with open(VALIDATION_LOG_FILE, "r") as f:
                all_progress = json.load(f)

            # Get progress for this source
            if self.source_name in all_progress:
                self.progress = all_progress[self.source_name]
            else:
                # Initialize empty progress for new source
                self.progress = {
                    "current_index": 0,
                    "accepted": [],
                    "rejected": [],
                    "auto_ingested": [],
                    "needs_review": [],
                }
                all_progress[self.source_name] = self.progress
                # Save it back
                with open(VALIDATION_LOG_FILE, "w") as f:
                    json.dump(all_progress, f, indent=2)
        else:
            # Create new log file
            self.progress = {
                "current_index": 0,
                "accepted": [],
                "rejected": [],
                "auto_ingested": [],
                "needs_review": [],
            }
            all_progress = {self.source_name: self.progress}
            os.makedirs(os.path.dirname(VALIDATION_LOG_FILE), exist_ok=True)
            with open(VALIDATION_LOG_FILE, "w") as f:
                json.dump(all_progress, f, indent=2)

    def save_progress(self):
        """Save validation progress to log file."""
        # Load entire file, update our source, save back
        with open(VALIDATION_LOG_FILE, "r") as f:
            all_progress = json.load(f)

        all_progress[self.source_name] = self.progress

        with open(VALIDATION_LOG_FILE, "w") as f:
            json.dump(all_progress, f, indent=2)

    def load_ingredient_mappings(self):
        """Load ingredient mappings from file."""
        if os.path.exists(MAPPINGS_FILE):
            with open(MAPPINGS_FILE, "r") as f:
                self.mappings = json.load(f)
            print(f"Loaded {len(self.mappings)} ingredient mappings")
        else:
            print(f"Warning: No mappings file found at {MAPPINGS_FILE}")
            self.mappings = {}

    def load_html_files(self):
        """Load list of HTML recipe files for this source."""
        self.html_files = self.recipe_source.find_html_files()
        print(f"Found {len(self.html_files)} HTML files for {self.source_name}")

    def load_database_mapping(self):
        """Load recipe source file to recipe_id mapping from database."""
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, source_file FROM recipe WHERE source_file IS NOT NULL"
            )
            self.db_mapping = {}
            self.recipe_names = {}
            for row in cursor.fetchall():
                recipe_id, name, source_file = row
                self.db_mapping[source_file] = recipe_id  # source_file -> recipe_id
                self.recipe_names[recipe_id] = name  # recipe_id -> name
            conn.close()
            print(f"Loaded {len(self.db_mapping)} recipe mappings from database")
        except Exception as e:
            print(f"Error loading database mapping: {e}")
            self.db_mapping = {}
            self.recipe_names = {}

    def get_current_recipe(self):
        """Get the current recipe to validate."""
        # Skip recipes that have already been processed
        while self.progress["current_index"] < len(self.html_files):
            html_file = self.html_files[self.progress["current_index"]]

            # Parse HTML to get recipe
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            parsed_recipe = self.recipe_source.parse_recipe_from_html(
                html_content, html_file
            )

            if not parsed_recipe:
                # Skip this recipe if parsing failed
                print(f"Skipping {html_file} - parsing failed")
                self.progress["current_index"] += 1
                continue

            recipe_name = parsed_recipe["name"]

            # Check if this recipe has already been processed
            if self._is_recipe_processed(recipe_name):
                # Skip this recipe and move to the next one
                self.progress["current_index"] += 1
                continue

            # Found an unprocessed recipe
            break

        # Check if we've reached the end
        if self.progress["current_index"] >= len(self.html_files):
            return None

        html_file = self.html_files[self.progress["current_index"]]

        # Load HTML content
        with open(html_file, "r", encoding="utf-8") as f:
            raw_html = f.read()

        # Parse recipe
        parsed_recipe = self.recipe_source.parse_recipe_from_html(raw_html, html_file)

        if not parsed_recipe:
            # Move to next if this one failed
            self.progress["current_index"] += 1
            return self.get_current_recipe()

        recipe_name = parsed_recipe["name"]

        # Skip recipes we don't want to validate
        # 1. Skip shot recipes (SHOT GLASS in instructions or "Shot" in name)
        instructions = parsed_recipe.get("instructions", "")
        if "SHOT GLASS" in instructions.upper():
            print(f"Skipping {recipe_name} - contains SHOT GLASS")
            self.progress["current_index"] += 1
            return self.get_current_recipe()

        if "Shot" in recipe_name:
            print(f"Skipping {recipe_name} - has 'Shot' in name")
            self.progress["current_index"] += 1
            return self.get_current_recipe()

        # 2. Skip recipes with 2 or fewer ingredients
        ingredients = parsed_recipe.get("ingredients", [])
        if len(ingredients) <= 2:
            print(f"Skipping {recipe_name} - has {len(ingredients)} ingredients")
            self.progress["current_index"] += 1
            return self.get_current_recipe()

        # Clean HTML for display
        html_content = self.recipe_source.clean_html_content(raw_html)

        # Rationalize ingredients using mappings
        rationalized_data = self._rationalize_recipe(parsed_recipe)

        return {
            "index": self.progress["current_index"],
            "total": len(self.html_files),
            "recipe_name": recipe_name,
            "html_file": html_file,
            "html_content": html_content,
            "rationalized_data": rationalized_data,
        }

    def _rationalize_recipe(self, parsed_recipe: dict) -> dict:
        """Rationalize ingredients in a parsed recipe.

        Args:
            parsed_recipe: Dict with name, ingredients, instructions, description

        Returns:
            Dict in output format with rationalized ingredients
        """
        rationalized_ingredients = []

        for ing in parsed_recipe["ingredients"]:
            ingredient_name = ing["ingredient_name"].strip()

            # Skip ingredients marked as optional
            ingredient_name_lower = ingredient_name.lower()
            if (
                "optional" in ingredient_name_lower
                or "omit if" in ingredient_name_lower
            ):
                continue

            # Try to rationalize using mappings
            if ingredient_name in self.mappings:
                mapping = self.mappings[ingredient_name]
                rationalized_ingredients.append(
                    {
                        "ingredient_name": mapping.get("name", ingredient_name),
                        "amount": ing.get("amount", ""),
                        "unit_name": ing.get("unit_name", ""),
                    }
                )
            else:
                # Keep original if not mapped
                rationalized_ingredients.append(ing)

        # Use extracted source_url if available, otherwise derive it
        if "source_url" in parsed_recipe and parsed_recipe["source_url"]:
            source_url = parsed_recipe["source_url"]
        else:
            source_url = self.recipe_source.derive_source_url(parsed_recipe["name"])

        # Validate URL
        # validate_url(source_url, parsed_recipe["name"])

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

    def _is_recipe_processed(self, recipe_name: str) -> bool:
        """Check if a recipe has already been processed."""
        return (
            recipe_name in self.progress.get("accepted", [])
            or recipe_name in self.progress.get("rejected", [])
            or recipe_name in self.progress.get("auto_ingested", [])
        )

    def accept_recipe(self, recipe_data):
        """Accept a recipe and save to validated recipes file."""
        if recipe_data.get("rationalized_data"):
            # Load existing validated recipes
            validated_recipes = {"recipes": []}
            if os.path.exists(VALIDATED_RECIPES_FILE):
                with open(VALIDATED_RECIPES_FILE, "r") as f:
                    validated_recipes = json.load(f)

            # Add the rationalized recipe
            validated_recipes["recipes"].append(recipe_data["rationalized_data"])

            # Save updated file
            os.makedirs(os.path.dirname(VALIDATED_RECIPES_FILE), exist_ok=True)
            with open(VALIDATED_RECIPES_FILE, "w") as f:
                json.dump(validated_recipes, f, indent=2)
                f.flush()

            # Update progress
            self.progress["accepted"].append(recipe_data["recipe_name"])

        self._advance_to_next()

    def reject_recipe(self, recipe_data):
        """Reject a recipe."""
        self.progress["rejected"].append(recipe_data["recipe_name"])
        self._advance_to_next()

    def _advance_to_next(self):
        """Advance to next recipe and save progress."""
        self.progress["current_index"] += 1
        self.save_progress()

    def reset_progress(self):
        """Reset validation progress to start over."""
        self.progress = {
            "current_index": 0,
            "accepted": [],
            "rejected": [],
            "auto_ingested": [],
            "needs_review": [],
        }
        self.save_progress()


@app.route("/")
def index():
    """Main validation interface."""
    recipe = validator.get_current_recipe()
    if recipe is None:
        return render_template(
            "completed.html",
            accepted_count=len(validator.progress["accepted"]),
            rejected_count=len(validator.progress["rejected"]),
            source=validator.source_name,
        )
    return render_template(
        "validator.html", recipe=recipe, source=validator.source_name
    )


@app.route("/accept", methods=["POST"])
def accept():
    """Accept current recipe."""
    recipe = validator.get_current_recipe()
    if recipe:
        # Check if custom JSON data was provided
        request_data = request.get_json()
        if request_data and "recipe_data" in request_data:
            # Use the edited JSON data from the client
            custom_recipe_data = request_data["recipe_data"]

            # Validate the custom recipe data
            try:
                # Basic validation - ensure required fields exist
                if not isinstance(custom_recipe_data, dict):
                    return jsonify(
                        {"success": False, "error": "Recipe data must be a JSON object"}
                    )

                required_fields = ["name", "ingredients"]
                for field in required_fields:
                    if field not in custom_recipe_data:
                        return jsonify(
                            {
                                "success": False,
                                "error": f"Missing required field: {field}",
                            }
                        )

                if not isinstance(custom_recipe_data["ingredients"], list):
                    return jsonify(
                        {"success": False, "error": "Ingredients must be a list"}
                    )

                # Replace the rationalized_data with the custom data
                recipe["rationalized_data"] = custom_recipe_data

            except Exception as e:
                return jsonify(
                    {"success": False, "error": f"Invalid recipe data: {str(e)}"}
                )

        validator.accept_recipe(recipe)
    return jsonify({"success": True})


@app.route("/reject", methods=["POST"])
def reject():
    """Reject current recipe."""
    recipe = validator.get_current_recipe()
    if recipe:
        validator.reject_recipe(recipe)
    return jsonify({"success": True})


@app.route("/status")
def status():
    """Get current validation status."""
    return jsonify(
        {
            "current_index": validator.progress["current_index"],
            "total": len(validator.html_files),
            "accepted": len(validator.progress["accepted"]),
            "rejected": len(validator.progress["rejected"]),
            "source": validator.source_name,
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    """Reset validation progress."""
    validator.reset_progress()
    return jsonify({"success": True, "message": "Progress reset"})


def main():
    """Run validator."""
    parser = argparse.ArgumentParser(description="Recipe validation web app")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["punch", "diffords"],
        help="Recipe source to validate (punch or diffords)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run Flask app on (default: 5000)",
    )

    args = parser.parse_args()

    # Initialize global validator
    global validator
    validator = RecipeValidator(args.source)

    print("=" * 70)
    print(f"Recipe Validator - {args.source.upper()}")
    print("=" * 70)
    print(f"HTML files found: {len(validator.html_files)}")
    print(f"Ingredient mappings loaded: {len(validator.mappings)}")
    print(f"Database mappings loaded: {len(validator.db_mapping)}")
    print(
        f"Progress: {validator.progress['current_index']}/{len(validator.html_files)} recipes"
    )
    print(
        f"Accepted: {len(validator.progress['accepted'])}, Rejected: {len(validator.progress['rejected'])}"
    )
    print("=" * 70)

    if len(validator.html_files) == 0:
        print(f"⚠️  No HTML files found in {validator.recipe_source.raw_recipes_dir}")
    else:
        print(f"\n🚀 Starting Flask app on http://localhost:{args.port}")
        print(f"   Validating {args.source} recipes\n")

    app.run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
