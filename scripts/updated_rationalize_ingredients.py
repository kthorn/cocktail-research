#!/usr/bin/env python3
"""
Updated rationalize_ingredients.py using cocktail-utils library
"""

import argparse
import csv
import datetime
import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import boto3

# Import from our new library
from cocktail_utils.database import get_connection
from cocktail_utils.ingredients import normalize_ingredient_text, extract_brand


@dataclass
class IngredientMatch:
    brand: Optional[str]
    specific_type: str
    category: str
    confidence: float
    source: str  # 'dictionary', 'llm', 'manual'


@dataclass
class IngredientUsage:
    ingredient_name: str
    recipe_count: int
    sample_recipes: List[str]


class IngredientParser:
    def __init__(
        self,
        taxonomy_file: str = "ingredient_taxonomy.json",
        brand_file: str = "brand_dictionary.json",
        db_path: str = "data/old_punch_recipes.db",
        output_format: str = "json",
        output_file: str = None,
    ):
        """Initialize the IngredientParser with configuration options."""
        self.taxonomy = json.load(
            open("data/ingredient_taxonomy.json", "r", encoding="utf-8")
        )
        self.brand_dictionary = json.load(
            open("data/brand_dictionary.json", "r", encoding="utf-8")
        )
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
        self.llm_cache = {}  # Cache LLM responses
        self.db_path = db_path
        self.output_format = output_format.lower()
        self.output_file = (
            output_file
            or f"ingredient_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.output_format}"
        )
        self.output_handle = None
        self.csv_writer = None
        self.results_written = 0

    def extract_brand_with_dictionary(self, ingredient_text: str) -> Tuple[Optional[str], str]:
        """Extract brand name using brand dictionary and patterns."""
        # Check if there is an exact match in brand_dictionary
        brand_category = self.brand_dictionary.get(ingredient_text)
        if brand_category:
            return ingredient_text, brand_category

        # Use the library's extract_brand function
        return extract_brand(ingredient_text)

    def dictionary_lookup(self, ingredient_text: str) -> Optional[IngredientMatch]:
        """Attempt to match ingredient using the loaded taxonomy dictionary."""
        # Use library's normalization function
        normalized = normalize_ingredient_text(ingredient_text)
        brand, cleaned_text = self.extract_brand_with_dictionary(normalized)

        # Try exact matches first
        for category, types in self.taxonomy.items():
            for specific_type, variations in types.items():
                # Check if cleaned text matches specific type
                if cleaned_text == specific_type:
                    return IngredientMatch(
                        brand, specific_type, category, 1.0, "dictionary"
                    )

                # Check variations
                for variation in variations:
                    if cleaned_text == variation:
                        return IngredientMatch(
                            brand, specific_type, category, 1.0, "dictionary"
                        )

        # Try word boundary matches
        for category, types in self.taxonomy.items():
            for specific_type, variations in types.items():
                if self._contains_as_words(cleaned_text, specific_type):
                    return IngredientMatch(
                        brand, specific_type, category, 0.9, "dictionary"
                    )

                for variation in variations:
                    if self._contains_as_words(cleaned_text, variation):
                        return IngredientMatch(
                            brand, specific_type, category, 0.9, "dictionary"
                        )

        # Partial matches only if meaningful
        for category, types in self.taxonomy.items():
            for specific_type, variations in types.items():
                if len(specific_type) >= 4 and specific_type in cleaned_text:
                    if self._is_meaningful_partial_match(cleaned_text, specific_type):
                        return IngredientMatch(
                            brand, specific_type, category, 0.7, "dictionary"
                        )

                for variation in variations:
                    if len(variation) >= 4 and variation in cleaned_text:
                        if self._is_meaningful_partial_match(cleaned_text, variation):
                            return IngredientMatch(
                                brand, specific_type, category, 0.7, "dictionary"
                            )

        return None

    def _contains_as_words(self, text: str, term: str) -> bool:
        """Check if term exists as complete words in text."""
        import re
        pattern = r"\b" + re.escape(term) + r"\b"
        return bool(re.search(pattern, text, re.IGNORECASE))

    def _is_meaningful_partial_match(self, text: str, term: str) -> bool:
        """Check if partial match is meaningful and not misleading."""
        if self._contains_as_words(text, term):
            return True

        import re
        matches = list(re.finditer(re.escape(term), text, re.IGNORECASE))
        for match in matches:
            start, end = match.span()
            before = text[start - 1] if start > 0 else " "
            after = text[end] if end < len(text) else " "

            if before in " -_" and after in " -_":
                return True

        return False

    def get_ingredients_from_db(self, min_recipe_count: int = 1) -> List[IngredientUsage]:
        """Retrieve ingredients from the database with usage statistics."""
        ingredients = []

        try:
            # Use library's database connection
            conn = get_connection(self.db_path)
            cursor = conn.cursor()

            query = """
            SELECT 
                i.name,
                COUNT(DISTINCT ri.recipe_id) as recipe_count,
                GROUP_CONCAT(DISTINCT r.name) as sample_recipes
            FROM ingredient i
            JOIN recipe_ingredient ri ON i.id = ri.ingredient_id
            JOIN recipe r ON ri.recipe_id = r.id
            GROUP BY i.id, i.name
            HAVING recipe_count >= ?
            ORDER BY recipe_count DESC
            """

            cursor.execute(query, (min_recipe_count,))
            results = cursor.fetchall()

            for name, count, sample_recipes in results:
                # Limit sample recipes to first 3 for display
                sample_list = sample_recipes.split(",")[:3] if sample_recipes else []
                ingredients.append(IngredientUsage(name, count, sample_list))

        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()

        return ingredients

    # ... (rest of the methods remain the same as original, focusing on LLM and output handling)


def main():
    """Main function using the cocktail-utils library for database operations."""
    parser_args = argparse.ArgumentParser(
        description="Rationalize cocktail ingredients from database"
    )
    parser_args.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    parser_args.add_argument(
        "--output",
        type=str,
        help="Output filename (default: auto-generated with timestamp)",
    )
    parser_args.add_argument(
        "--min-recipes",
        type=int,
        default=2,
        help="Minimum number of recipes an ingredient must appear in (default: 2)",
    )
    args = parser_args.parse_args()

    # Initialize parser with output configuration
    parser = IngredientParser(
        output_format=args.format,
        output_file=args.output,
    )

    try:
        # Get ingredients from database using library utilities
        print("Loading ingredients from database...")
        ingredient_usages = parser.get_ingredients_from_db(
            min_recipe_count=args.min_recipes
        )

        if not ingredient_usages:
            print(
                "No ingredients found in database. Make sure punch_recipes.db exists and has data."
            )
            exit(1)

        print(
            f"Found {len(ingredient_usages)} ingredients used in {args.min_recipes}+ recipes"
        )
        print(f"Results will be written to {parser.output_file} as {args.format.upper()}")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        raise


if __name__ == "__main__":
    main()