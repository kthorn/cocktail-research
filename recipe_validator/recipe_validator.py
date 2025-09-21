#!/usr/bin/env python3
"""
Simple recipe validation web app.
Displays raw HTML recipes alongside rationalized data for validation.
"""

import json
import os
import glob
import sqlite3
import re
import requests
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import pandas as pd
from bs4 import BeautifulSoup

app = Flask(__name__)


def find_most_recent_raw_ingredients_file(data_dir="data"):
    """Find the most recent raw ingredients parquet file based on timestamp."""
    pattern = os.path.join(data_dir, "raw_recipe_ingredients_*.parquet")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No original ingredients files found matching pattern: {pattern}"
        )

    # Extract timestamps and find the most recent
    file_timestamps = []
    for file_path in files:
        filename = os.path.basename(file_path)
        # Extract timestamp from filename using regex
        match = re.search(
            r"raw_recipe_ingredients_(\d{8}_\d{6})\.parquet",
            filename,
        )
        if match:
            timestamp_str = match.group(1)
            try:
                # Parse timestamp (format: YYYYMMDD_HHMMSS)
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                file_timestamps.append((timestamp, file_path))
            except ValueError:
                print(f"Warning: Could not parse timestamp from {filename}")
                continue

    if not file_timestamps:
        raise ValueError("No valid timestamp files found")

    # Sort by timestamp and return the most recent file
    file_timestamps.sort(key=lambda x: x[0], reverse=True)
    most_recent_file = file_timestamps[0][1]

    print(f"Using most recent raw ingredients file: {most_recent_file}")
    return most_recent_file


VALIDATION_LOG_FILE = "validation_log.json"
VALIDATED_RECIPES_FILE = "input_data/validated-recipes.json"
RAW_RECIPES_DIR = "raw_recipes/punch_html"
RAW_INGREDIENTS_FILE = find_most_recent_raw_ingredients_file()
DATABASE_FILE = "data/recipes.db"


class RecipeValidator:
    def __init__(self):
        self.load_progress()
        self.load_raw_ingredients_data()
        self.load_database_mapping()
        self.load_html_files()

    def load_progress(self):
        """Load validation progress from log file."""
        if os.path.exists(VALIDATION_LOG_FILE):
            with open(VALIDATION_LOG_FILE, "r") as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "current_index": 0,
                "reviewed": [],
                "accepted": [],
                "rejected": [],
            }

    def save_progress(self):
        """Save validation progress to log file."""
        with open(VALIDATION_LOG_FILE, "w") as f:
            json.dump(self.progress, f, indent=2)

    def load_html_files(self):
        """Load list of HTML recipe files that have rationalized data."""
        all_html_files = sorted(glob.glob(f"{RAW_RECIPES_DIR}/*.html"))
        print(f"Found {len(all_html_files)} total HTML files")

        # Filter to only include files that have raw ingredients data
        self.html_files = []
        if self.raw_ingredients_df is not None and self.db_mapping:
            for html_file in all_html_files:
                # Convert to the same format as database (raw_recipes/punch_html/filename.html)
                path_parts = html_file.split(os.sep)
                if len(path_parts) >= 2:
                    db_path = "/".join(
                        path_parts[-2:]
                    )  # Take last 2 parts: punch_html/filename.html
                    db_path = f"raw_recipes/{db_path}"  # Add raw_recipes/ prefix

                    # Check if this file is in our database mapping
                    if db_path in self.db_mapping:
                        recipe_id = self.db_mapping[db_path]
                        # Check if this recipe has raw ingredients data
                        if recipe_id in self.raw_ingredients_df["recipe_id"].values:
                            self.html_files.append(html_file)

            print(
                f"Filtered to {len(self.html_files)} HTML files with raw ingredients data"
            )
        else:
            self.html_files = all_html_files
            print(
                "No raw ingredients data or database mapping available, showing all files"
            )

    def load_raw_ingredients_data(self):
        """Load raw ingredients data from parquet file."""
        try:
            self.raw_ingredients_df = pd.read_parquet(RAW_INGREDIENTS_FILE)
            print(f"Loaded raw ingredients data: {self.raw_ingredients_df.shape}")
        except Exception as e:
            print(f"Error loading raw ingredients data: {e}")
            self.raw_ingredients_df = None

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
        if self.progress["current_index"] >= len(self.html_files):
            return None

        html_file = self.html_files[self.progress["current_index"]]

        # Get recipe name from database mapping using same path format as in load_html_files
        path_parts = html_file.split(os.sep)
        if len(path_parts) >= 2:
            db_path = "/".join(path_parts[-2:])  # Take last 2 parts
            db_path = f"raw_recipes/{db_path}"  # Add raw_recipes/ prefix
        else:
            db_path = html_file

        recipe_id = self.db_mapping.get(db_path)
        recipe_name = self.recipe_names.get(
            recipe_id, Path(html_file).stem.replace("-", " ").title()
        )

        # Load HTML content and clean it
        with open(html_file, "r", encoding="utf-8") as f:
            raw_html = f.read()

        html_content = self._clean_html_content(raw_html)

        # Get raw ingredients data
        raw_recipe_data = None
        if self.raw_ingredients_df is not None and recipe_id is not None:
            recipe_ingredients = self.raw_ingredients_df[
                self.raw_ingredients_df["recipe_id"] == recipe_id
            ]
            if not recipe_ingredients.empty:
                raw_recipe_data = self._format_raw_ingredients_recipe(
                    recipe_id, recipe_name, recipe_ingredients
                )

        return {
            "index": self.progress["current_index"],
            "total": len(self.html_files),
            "recipe_name": recipe_name,
            "html_file": html_file,
            "html_content": html_content,
            "rationalized_data": raw_recipe_data,
        }

    def _clean_html_content(self, html_content):
        """Extract only the recipe content section and remove duplicates."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Try to find the main recipe content area
            recipe_content = None

            # Look for common recipe container selectors
            selectors_to_try = [
                "article",
                ".recipe-content",
                ".entry-content",
                ".post-content",
                '[itemtype*="Recipe"]',
                "main",
            ]

            for selector in selectors_to_try:
                recipe_content = soup.select_one(selector)
                if recipe_content:
                    break

            if not recipe_content:
                # Fallback: use the whole body if no specific container found
                recipe_content = soup.find("body") or soup

            # Remove unwanted sections from the recipe content
            unwanted_selectors = [
                "nav",
                "header",
                "footer",
                ".navigation",
                ".nav",
                ".sidebar",
                ".ads",
                ".advertisement",
                ".social",
                ".comments",
                ".comment",
                ".related",
                ".newsletter",
                "script",
                "style",
                ".cookie",
                ".gdpr",
                '[style*="display:none"]',
                '[style*="display: none"]',
                ".share",
                ".sharing",
                ".social-share",
                ".share-buttons",
                "form",
                ".form",
                ".email-form",
                ".subscription",
                ".share-story",
                ".tweet",
                ".email-page",
                "img",
                "picture",
                "figure",
                ".image",
                ".photo",
                ".ico",
            ]

            for selector in unwanted_selectors:
                for element in recipe_content.select(selector):
                    element.decompose()

            # Remove hidden duplicate recipe ingredients
            for element in recipe_content.find_all(
                attrs={"itemprop": "recipeIngredient"}
            ):
                if element.get("style") and "display:none" in element.get("style"):
                    element.decompose()

            # Remove text-based sharing elements
            sharing_text_patterns = [
                "Share story:",
                "Share",
                "Tweet",
                "Email This Page",
                "Email",
                "Facebook",
                "Twitter",
                "Pinterest",
            ]

            # Find and remove elements containing sharing text
            for element in recipe_content.find_all(text=True):
                if any(pattern in element.strip() for pattern in sharing_text_patterns):
                    # Remove the parent element if it only contains sharing text
                    parent = element.parent
                    if parent and parent.get_text().strip() in sharing_text_patterns:
                        parent.decompose()
                    else:
                        # Just remove the text node
                        element.replace_with("")

            # Create a clean HTML structure with just the recipe content
            clean_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Recipe</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .ingredients-list {{ list-style-type: disc; margin-left: 20px; }}
                    .ingredients-list li {{ margin: 5px 0; }}
                </style>
            </head>
            <body>
                {recipe_content}
            </body>
            </html>
            """

            return clean_html

        except Exception as e:
            print(f"Error cleaning HTML: {e}")
            return html_content  # Return original if parsing fails

    def _format_raw_ingredients_recipe(
        self, recipe_id, recipe_name, recipe_ingredients_df
    ):
        """Format raw ingredients data for display."""
        ingredients = []

        for _, row in recipe_ingredients_df.iterrows():
            # Use the raw ingredient name directly
            ingredient_name = row["ingredient_name"]

            ingredients.append(
                {
                    "ingredient_name": ingredient_name,
                    "amount": row.get("original_amount", 0),
                    "unit_name": row.get("original_unit", ""),
                }
            )

        # Derive source URL from recipe name - convert to punch URL format
        # Clean special characters for URL
        clean_name = recipe_name.lower()
        # Replace unicode characters with normal equivalents
        unicode_replacements = {
            "\u2019": "'",  # right single quotation mark
            "\u2018": "'",  # left single quotation mark
            "\u201c": '"',  # left double quotation mark
            "\u201d": '"',  # right double quotation mark
            "\u00a0": " ",  # non-breaking space
        }
        for unicode_char, replacement in unicode_replacements.items():
            clean_name = clean_name.replace(unicode_char, replacement)
        # Remove apostrophes and other special characters for URL formatting
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in " -&")
        # Replace spaces and ampersands
        clean_name = clean_name.replace(" ", "-").replace("&", "and")
        # Remove multiple consecutive dashes
        clean_name = "-".join(filter(None, clean_name.split("-")))
        source_url = f"https://punchdrink.com/recipes/{clean_name}/"

        # Validate URL and report if invalid
        self._validate_url(source_url, recipe_name)

        return {
            "name": recipe_name,
            "description": "",
            "instructions": "Shake all ingredients with ice and strain into a cocktail or coupe glass",
            "ingredients": ingredients,
            "source_url": source_url,
        }

    def _validate_url(self, url, recipe_name):
        """Validate URL by making a HEAD request and report invalid URLs to console."""
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            if response.status_code >= 400:
                print(
                    f"⚠️  INVALID URL for recipe '{recipe_name}': {url} (Status: {response.status_code})"
                )
        except requests.exceptions.RequestException as e:
            print(
                f"⚠️  URL VALIDATION ERROR for recipe '{recipe_name}': {url} (Error: {str(e)})"
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

            # Save updated file and close it explicitly
            os.makedirs(os.path.dirname(VALIDATED_RECIPES_FILE), exist_ok=True)
            with open(VALIDATED_RECIPES_FILE, "w") as f:
                json.dump(validated_recipes, f, indent=2)
                f.flush()  # Ensure data is written to disk
            # File is automatically closed here due to context manager

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
        self.progress["reviewed"].append(
            self.html_files[self.progress["current_index"] - 1]
        )
        self.save_progress()

    def reset_progress(self):
        """Reset validation progress to start over."""
        self.progress = {
            "current_index": 0,
            "reviewed": [],
            "accepted": [],
            "rejected": [],
        }
        self.save_progress()


validator = RecipeValidator()


@app.route("/")
def index():
    """Main validation interface."""
    recipe = validator.get_current_recipe()
    if recipe is None:
        return render_template(
            "completed.html",
            accepted_count=len(validator.progress["accepted"]),
            rejected_count=len(validator.progress["rejected"]),
        )
    return render_template("validator.html", recipe=recipe)


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
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    """Reset validation progress."""
    validator.reset_progress()
    return jsonify({"success": True, "message": "Progress reset"})


if __name__ == "__main__":
    print("Recipe Validator starting...")
    print(f"HTML files found: {len(validator.html_files)}")
    print(f"Raw ingredients data loaded: {validator.raw_ingredients_df is not None}")
    print(
        f"Database mappings loaded: {len(validator.db_mapping) if validator.db_mapping else 0}"
    )
    print(
        f"Progress: {validator.progress['current_index']}/{len(validator.html_files)} recipes"
    )
    print(
        f"Accepted: {len(validator.progress['accepted'])}, Rejected: {len(validator.progress['rejected'])}"
    )
    if len(validator.html_files) == 0:
        print("No HTML files available for validation!")
        print(f"Looking in directory: {RAW_RECIPES_DIR}")
        print(f"Raw ingredients file: {RAW_INGREDIENTS_FILE}")
        print(f"Database file: {DATABASE_FILE}")
    app.run(debug=True, port=5000)
