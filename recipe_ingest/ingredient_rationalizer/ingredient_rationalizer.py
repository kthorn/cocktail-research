#!/usr/bin/env python3
"""
Ingredient rationalization tool for validating and mapping recipe ingredients
against a master ingredients database API.
"""

import json
import os
import requests
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from difflib import SequenceMatcher
from typing import Dict, List

app = Flask(__name__)

# Configuration
VALIDATED_RECIPES_FILE = "../../input_data/validated-recipes.json"
INGREDIENT_MAPPINGS_FILE = "ingredient_mappings.json"
PROGRESS_FILE = "rationalization_progress.json"
INGREDIENTS_API_URL = (
    "https://a5crx5o72d.execute-api.us-east-1.amazonaws.com/api/ingredients"
)
OUTPUT_FILE = "../../input_data/rationalized-recipes.json"
NEW_INGREDIENTS_FILE = "../../input_data/new-ingredients-for-upload.json"


class IngredientRationalizer:
    def __init__(self):
        self.new_ingredients = []  # Initialize new_ingredients list
        self.load_validated_recipes()
        self.load_api_ingredients()
        self.load_mappings()
        self.load_new_ingredients()
        self.load_progress()
        self.extract_unique_ingredients()
        self.auto_match_exact_ingredients()

    def load_validated_recipes(self):
        """Load recipes from validated-recipes.json"""
        try:
            with open(VALIDATED_RECIPES_FILE, "r") as f:
                data = json.load(f)
                self.recipes = data.get("recipes", [])
                print(f"Loaded {len(self.recipes)} recipes")
        except FileNotFoundError:
            print(f"Error: {VALIDATED_RECIPES_FILE} not found")
            self.recipes = []
        except Exception as e:
            print(f"Error loading recipes: {e}")
            self.recipes = []

    def load_api_ingredients(self):
        """Fetch and cache ingredients from API"""
        try:
            print("Fetching ingredients from API...")
            response = requests.get(INGREDIENTS_API_URL, timeout=30)
            response.raise_for_status()
            self.api_ingredients = response.json()

            # Create lookup dictionaries for easier access
            self.api_ingredients_by_id = {
                ing["id"]: ing for ing in self.api_ingredients
            }
            self.api_ingredients_by_name = {
                ing["name"].lower(): ing for ing in self.api_ingredients
            }

            print(f"Loaded {len(self.api_ingredients)} ingredients from API")
        except Exception as e:
            print(f"Error fetching API ingredients: {e}")
            self.api_ingredients = []
            self.api_ingredients_by_id = {}
            self.api_ingredients_by_name = {}

    def load_mappings(self):
        """Load existing ingredient mappings from file"""
        if os.path.exists(INGREDIENT_MAPPINGS_FILE):
            try:
                with open(INGREDIENT_MAPPINGS_FILE, "r") as f:
                    self.mappings = json.load(f)
                print(f"Loaded {len(self.mappings)} existing mappings")
            except Exception as e:
                print(f"Error loading mappings: {e}")
                self.mappings = {}
        else:
            self.mappings = {}

    def save_mappings(self):
        """Save ingredient mappings to file"""
        with open(INGREDIENT_MAPPINGS_FILE, "w") as f:
            json.dump(self.mappings, f, indent=2)

    def load_new_ingredients(self):
        """Load new ingredients from file"""
        if os.path.exists(NEW_INGREDIENTS_FILE):
            try:
                with open(NEW_INGREDIENTS_FILE, "r") as f:
                    data = json.load(f)
                    ingredients = data.get("ingredients", [])
                    self.new_ingredients = [ing["name"] for ing in ingredients]
                print(f"Loaded {len(self.new_ingredients)} new ingredients from file")
            except Exception as e:
                print(f"Error loading new ingredients: {e}")
                self.new_ingredients = []
        else:
            self.new_ingredients = []

    def load_progress(self):
        """Load rationalization progress"""
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, "r") as f:
                    self.progress = json.load(f)
            except Exception as e:
                print(f"Error loading progress: {e}")
                self.progress = self._init_progress()
        else:
            self.progress = self._init_progress()

    def _init_progress(self):
        """Initialize progress tracking"""
        return {
            "current_ingredient": None,
            "processed": [],
            "skipped": [],
            "last_updated": datetime.now().isoformat(),
        }

    def save_progress(self):
        """Save current progress"""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(PROGRESS_FILE, "w") as f:
            json.dump(self.progress, f, indent=2)

    def extract_unique_ingredients(self):
        """Extract all unique ingredient names from recipes"""
        unique_ingredients = set()
        for recipe in self.recipes:
            for ingredient in recipe.get("ingredients", []):
                ingredient_name = ingredient.get("ingredient_name", "").strip()
                if ingredient_name:
                    unique_ingredients.add(ingredient_name)

        # Sort alphabetically for easier processing
        self.unique_ingredients = sorted(list(unique_ingredients))

        # Filter to only unmapped ingredients
        # Exclude ingredients that have mappings OR are marked as new ingredients
        self.unmapped_ingredients = [
            ing
            for ing in self.unique_ingredients
            if ing not in self.mappings and ing not in getattr(self, 'new_ingredients', [])
        ]

        print(f"Found {len(self.unique_ingredients)} unique ingredients")
        print(f"{len(self.unmapped_ingredients)} need mapping")

    def auto_match_exact_ingredients(self):
        """Automatically map ingredients that have exact matches in the API"""
        auto_matched = []
        debug_ingredients = ['ricard pastis', 'creole shrubb, such as hamilton petite shrubb', 'luxardo bitter']

        # Check all unique ingredients that don't have mappings yet
        # This includes both unmapped and previously processed but unmapped ingredients
        for ingredient_name in self.unique_ingredients:
            # Skip if already mapped
            if ingredient_name in self.mappings:
                continue

            ingredient_lower = ingredient_name.lower()
            matched_ingredient = None

            # Debug logging for specific ingredients
            if ingredient_lower in [ing.lower() for ing in debug_ingredients]:
                print(f"\nDEBUG: Checking '{ingredient_name}'")
                # Check if any API ingredient contains this string
                partial_matches = [
                    api_name for api_name in self.api_ingredients_by_name.keys()
                    if ingredient_lower in api_name or api_name in ingredient_lower
                ]
                if partial_matches:
                    print(f"  Partial matches found: {partial_matches[:5]}")
                else:
                    print(f"  No partial matches found")

            # Check for exact match in API ingredients
            if ingredient_lower in self.api_ingredients_by_name:
                matched_ingredient = self.api_ingredients_by_name[ingredient_lower]

            if matched_ingredient:
                # Automatically create mapping
                self.mappings[ingredient_name] = {
                    "id": matched_ingredient["id"],
                    "name": matched_ingredient["name"],
                    "mapped_at": datetime.now().isoformat(),
                    "auto_matched": True,
                }

                # Mark as processed if not already
                if ingredient_name not in self.progress["processed"]:
                    self.progress["processed"].append(ingredient_name)
                auto_matched.append(ingredient_name)

        if auto_matched:
            # Save the auto-matched mappings
            self.save_mappings()

            # Recalculate unmapped ingredients
            self.extract_unique_ingredients()

            # Reset current ingredient to first unmapped (if any)
            if self.unmapped_ingredients:
                self.progress["current_ingredient"] = self.unmapped_ingredients[0]
            else:
                self.progress["current_ingredient"] = None

            self.save_progress()

            print(
                f"Auto-matched {len(auto_matched)} ingredients with exact API matches"
            )
            for name in auto_matched[:10]:  # Show first 10
                print(f"  - {name}")
            if len(auto_matched) > 10:
                print(f"  ... and {len(auto_matched) - 10} more")

    def get_current_ingredient(self):
        """Get the current ingredient to process"""
        # If no unmapped ingredients remain, we're done
        if not self.unmapped_ingredients:
            return None

        # Find the next ingredient to process
        current_ingredient = None
        current_index = 0

        # If we have a current ingredient from progress, try to find it
        if self.progress.get("current_ingredient"):
            try:
                current_index = self.unmapped_ingredients.index(self.progress["current_ingredient"])
                current_ingredient = self.progress["current_ingredient"]
            except ValueError:
                # Current ingredient no longer in unmapped list, start with first
                pass

        # If no valid current ingredient, use the first unmapped one
        if current_ingredient is None:
            current_ingredient = self.unmapped_ingredients[0]
            current_index = 0
            # Update progress to track this ingredient
            self.progress["current_ingredient"] = current_ingredient
            self.save_progress()

        # Find suggested matches based on string similarity
        suggestions = self.find_suggestions(current_ingredient)

        return {
            "index": current_index,
            "total": len(self.unmapped_ingredients),
            "ingredient_name": current_ingredient,
            "suggestions": suggestions,
            "recipes_using": self.get_recipes_using_ingredient(current_ingredient),
        }

    def find_suggestions(
        self, ingredient_name: str, threshold: float = 0.6
    ) -> List[Dict]:
        """Find similar ingredients from API based on string similarity"""
        suggestions = []
        ingredient_lower = ingredient_name.lower()

        # Direct match check first
        if ingredient_lower in self.api_ingredients_by_name:
            api_ing = self.api_ingredients_by_name[ingredient_lower]
            suggestions.append(
                {
                    "id": api_ing["id"],
                    "name": api_ing["name"],
                    "path": api_ing.get("path", ""),
                    "similarity": 1.0,
                    "exact_match": True,
                }
            )

        # Find similar matches
        for api_ing in self.api_ingredients:
            api_name_lower = api_ing["name"].lower()

            # Skip if already added as exact match
            if api_name_lower == ingredient_lower:
                continue

            # Calculate similarity
            similarity = SequenceMatcher(None, ingredient_lower, api_name_lower).ratio()

            # Also check if one contains the other
            contains_match = (
                ingredient_lower in api_name_lower or api_name_lower in ingredient_lower
            )

            if similarity >= threshold or contains_match:
                suggestions.append(
                    {
                        "id": api_ing["id"],
                        "name": api_ing["name"],
                        "path": api_ing.get("path", ""),
                        "similarity": similarity,
                        "exact_match": False,
                        "contains_match": contains_match,
                    }
                )

        # Sort by similarity score
        suggestions.sort(
            key=lambda x: (x.get("exact_match", False), x["similarity"]), reverse=True
        )

        # Limit to top 10 suggestions
        return suggestions[:10]

    def get_recipes_using_ingredient(self, ingredient_name: str) -> List[str]:
        """Get list of recipe names using this ingredient"""
        recipes_using = []
        for recipe in self.recipes:
            for ingredient in recipe.get("ingredients", []):
                if ingredient.get("ingredient_name", "").strip() == ingredient_name:
                    recipes_using.append(recipe.get("name", "Unknown"))
                    break
        return recipes_using[:5]  # Limit to first 5 for display

    def map_ingredient(self, original_name: str, mapping_data: Dict):
        """Map an ingredient to API ingredient or mark as new"""
        mapping_type = mapping_data.get("type")

        if mapping_type == "existing":
            # Map to existing API ingredient
            ingredient_id = mapping_data.get("ingredient_id")
            if ingredient_id:
                # Convert to int if it's a string (from web interface)
                try:
                    ingredient_id = int(ingredient_id)
                except (ValueError, TypeError):
                    pass

                if ingredient_id in self.api_ingredients_by_id:
                    api_ingredient = self.api_ingredients_by_id[ingredient_id]
                    self.mappings[original_name] = {
                        "id": ingredient_id,
                        "name": api_ingredient["name"],
                        "mapped_at": datetime.now().isoformat(),
                    }

        elif mapping_type == "new":
            # Track new ingredients in memory only (not in progress file)
            if not hasattr(self, "new_ingredients"):
                self.new_ingredients = []
            if original_name not in self.new_ingredients:
                self.new_ingredients.append(original_name)
            # Export new ingredients to JSON file
            self.export_new_ingredients()

        elif mapping_type == "skip":
            # Skip this ingredient for now
            self.progress["skipped"].append(original_name)

        # Update progress - add to processed list
        if original_name not in self.progress["processed"]:
            self.progress["processed"].append(original_name)

        # Recalculate unmapped ingredients first
        self.extract_unique_ingredients()

        # Set next ingredient to process (first unmapped ingredient)
        if self.unmapped_ingredients:
            self.progress["current_ingredient"] = self.unmapped_ingredients[0]
        else:
            self.progress["current_ingredient"] = None

        # Save state
        self.save_mappings()
        self.save_progress()

    def export_new_ingredients(self):
        """Export new ingredients to JSON file for upload"""
        if not hasattr(self, "new_ingredients") or not self.new_ingredients:
            return

        new_ingredients = []
        for ingredient_name in self.new_ingredients:
            new_ingredients.append({"name": ingredient_name, "parent_name": ""})

        export_data = {"ingredients": new_ingredients}

        with open(NEW_INGREDIENTS_FILE, "w") as f:
            json.dump(export_data, f, indent=2)

        print(
            f"Exported {len(new_ingredients)} new ingredients to {NEW_INGREDIENTS_FILE}"
        )

    def export_rationalized_recipes(self):
        """Export recipes with rationalized ingredient data"""
        rationalized_recipes = []

        for recipe in self.recipes:
            rationalized_recipe = recipe.copy()
            rationalized_ingredients = []

            for ingredient in recipe.get("ingredients", []):
                ingredient_name = ingredient.get("ingredient_name", "").strip()
                rationalized_ing = ingredient.copy()

                # Replace ingredient_name with mapped name if available
                if ingredient_name in self.mappings:
                    mapping = self.mappings[ingredient_name]
                    rationalized_ing["ingredient_name"] = mapping.get("name")
                # For ingredients processed but not mapped (marked as new in old system or current)
                # Keep original name - they need to be re-processed or added as new

                rationalized_ingredients.append(rationalized_ing)

            rationalized_recipe["ingredients"] = rationalized_ingredients
            rationalized_recipes.append(rationalized_recipe)

        # Save to output file in exact same format as validated-recipes.json
        output_data = {"recipes": rationalized_recipes}

        with open(OUTPUT_FILE, "w") as f:
            json.dump(output_data, f, indent=2)

        # Return metadata for display purposes
        return {
            "total_recipes": len(rationalized_recipes),
            "total_ingredients": len(self.unique_ingredients),
            "mapped_ingredients": len(self.mappings),
            "new_ingredients": len(self.new_ingredients)
            if hasattr(self, "new_ingredients")
            else 0,
            "exported_at": datetime.now().isoformat(),
        }

    def reset_progress(self):
        """Reset rationalization progress"""
        self.progress = self._init_progress()
        self.extract_unique_ingredients()
        # Set current ingredient to first unmapped (if any)
        if self.unmapped_ingredients:
            self.progress["current_ingredient"] = self.unmapped_ingredients[0]
        self.save_progress()

    def get_statistics(self):
        """Get current rationalization statistics"""
        return {
            "total_recipes": len(self.recipes),
            "total_unique_ingredients": len(self.unique_ingredients),
            "mapped_ingredients": len(self.mappings),
            "unmapped_ingredients": len(self.unmapped_ingredients),
            "processed_this_session": len(self.progress["processed"]),
            "skipped": len(self.progress["skipped"]),
            "new_ingredients": len(self.new_ingredients)
            if hasattr(self, "new_ingredients")
            else 0,
            "api_ingredients_available": len(self.api_ingredients),
        }


# Initialize rationalizer
rationalizer = IngredientRationalizer()


@app.route("/")
def index():
    """Main rationalization interface"""
    current = rationalizer.get_current_ingredient()

    if current is None:
        # All ingredients processed
        stats = rationalizer.get_statistics()
        return render_template("completed.html", stats=stats)

    return render_template(
        "rationalizer.html",
        current=current,
        api_ingredients=rationalizer.api_ingredients,
        stats=rationalizer.get_statistics(),
    )


@app.route("/map", methods=["POST"])
def map_ingredient():
    """Map current ingredient"""
    data = request.get_json()

    current = rationalizer.get_current_ingredient()
    if current:
        rationalizer.map_ingredient(current["ingredient_name"], data)

    return jsonify({"success": True})


@app.route("/status")
def status():
    """Get current status"""
    return jsonify(rationalizer.get_statistics())


@app.route("/export", methods=["POST"])
def export():
    """Export rationalized recipes"""
    try:
        metadata = rationalizer.export_rationalized_recipes()
        return jsonify({"success": True, "metadata": metadata, "file": OUTPUT_FILE})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/reset", methods=["POST"])
def reset():
    """Reset progress"""
    rationalizer.reset_progress()
    return jsonify({"success": True})


@app.route("/search_ingredients")
def search_ingredients():
    """Search API ingredients"""
    query = request.args.get("q", "").lower()
    if not query:
        return jsonify([])

    results = []
    for ing in rationalizer.api_ingredients:
        if query in ing["name"].lower():
            results.append(
                {"id": ing["id"], "name": ing["name"], "path": ing.get("path", "")}
            )

    return jsonify(results[:20])  # Limit to 20 results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Ingredient Rationalization Tool")
    print("=" * 60)
    print(f"Recipes loaded: {len(rationalizer.recipes)}")
    print(f"Unique ingredients: {len(rationalizer.unique_ingredients)}")
    print(f"Already mapped: {len(rationalizer.mappings)}")
    print(f"Need mapping: {len(rationalizer.unmapped_ingredients)}")
    print(f"API ingredients: {len(rationalizer.api_ingredients)}")
    print("=" * 60)
    print("Starting web server at http://localhost:5002")
    print("=" * 60 + "\n")

    app.run(debug=True, port=5002)
