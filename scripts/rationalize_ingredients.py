import argparse
import csv
import datetime
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import boto3


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
        """Initialize the IngredientParser with configuration options.

        Args:
            taxonomy_file: Path to the JSON file containing ingredient taxonomy.
                Defaults to "ingredient_taxonomy.json".
            brand_file: Path to te JSON file containing brand -> category dictionary.
                Handles ingredients where only a brand name is provided.
            db_path: Path to the SQLite database containing recipe data.
                Defaults to "punch_recipes.db".
            output_format: Format for output file, either "json" or "csv".
                Defaults to "json".
            output_file: Custom output filename. If None, auto-generates filename
                with timestamp. Defaults to None.
        """
        self.taxonomy = json.load(
            open("data/ingredient_taxonomy.json", "r", encoding="utf-8")
        )
        self.brand_dictionary = json.load(
            open("data/brand_dictionary.json", "r", encoding="utf-8")
        )
        self.brand_patterns = self._compile_brand_patterns()
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

    def _compile_brand_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for brand extraction from ingredient text.

        Creates regex patterns to identify and extract brand names from
        ingredient descriptions, looking for patterns like "preferably X",
        "such as X", "like X", and brands mentioned at the end.

        Returns:
            A list of compiled regex patterns for brand extraction.
        """
        return [
            re.compile(r"preferably\s+([A-Z][a-zA-Z\s&\'\-\d]+)", re.IGNORECASE),
            re.compile(r"such as\s+([A-Z][a-zA-Z\s&\'\-\d]+)", re.IGNORECASE),
            re.compile(r"like\s+([A-Z][a-zA-Z\s&\'\-\d]+)", re.IGNORECASE),
            # Pattern for brands at the end without "preferably"
            re.compile(r",\s+([A-Z][a-zA-Z\s&\'\-\d]+)$", re.IGNORECASE),
        ]

    def extract_brand(self, ingredient_text: str) -> Tuple[Optional[str], str]:
        """Extract brand name and return cleaned ingredient text.

        Searches for brand references in ingredient text using compiled patterns
        and removes them to get the clean ingredient name.

        Args:
            ingredient_text: The raw ingredient description text.

        Returns:
            A tuple containing:
                - Optional brand name if found, None otherwise
                - Cleaned ingredient text with brand reference removed
        """
        # Check if there is an exact match in brand_dictionary
        brand_category = self.brand_dictionary.get(ingredient_text)
        if brand_category:
            return ingredient_text, brand_category

        for pattern in self.brand_patterns:
            match = pattern.search(ingredient_text)
            if match:
                brand = match.group(1).strip()
                # Remove the brand reference from the text
                cleaned_text = pattern.sub("", ingredient_text).strip().rstrip(",")
                return brand, cleaned_text
        return None, ingredient_text

    def normalize_ingredient(self, ingredient_text: str) -> str:
        """Normalize ingredient text for consistent matching.

        Removes quantities, measurements, and common prefixes that don't
        affect ingredient categorization. Converts to lowercase and
        normalizes whitespace.

        Args:
            ingredient_text: The raw ingredient description text.

        Returns:
            Normalized ingredient text suitable for taxonomy matching.

        Examples:
            "2 oz fresh lemon juice" → "lemon juice"
            "1 tsp premium vanilla extract" → "vanilla extract"
        """
        # Remove quantities and measurements
        text = re.sub(
            r"^\d+[\s\d/]*\s*(ounces?|oz|cups?|tsp|tbsp|ml|cl|dashes?|drops?)\s+",
            "",
            ingredient_text,
            flags=re.IGNORECASE,
        )

        # Remove common prefixes that don't affect categorization
        text = re.sub(
            r"^(fresh|freshly|good|quality|premium)\s+", "", text, flags=re.IGNORECASE
        )

        # Normalize whitespace and convert to lowercase
        text = " ".join(text.lower().split())

        return text

    def dictionary_lookup(self, ingredient_text: str) -> Optional[IngredientMatch]:
        """Attempt to match ingredient using the loaded taxonomy dictionary.

        Tries multiple matching strategies in order of precision:
        1. Exact matches with specific types and variations
        2. Word boundary matches
        3. Meaningful partial matches

        Args:
            ingredient_text: The ingredient text to match against taxonomy.

        Returns:
            An IngredientMatch object if a match is found, None otherwise.
            Match confidence varies based on matching strategy used.
        """
        normalized = self.normalize_ingredient(ingredient_text)
        brand, cleaned_text = self.extract_brand(normalized)

        # Debug logging for troublesome ingredients

        # Try exact matches first
        for category, types in self.taxonomy.items():
            for specific_type, variations in types.items():
                # Check if cleaned text matches specific type
                if cleaned_text == specific_type:
                    if "campari" in ingredient_text.lower():
                        print(
                            f"DEBUG: Exact match found - cleaned_text='{cleaned_text}' == specific_type='{specific_type}' in category='{category}'"
                        )
                    return IngredientMatch(
                        brand, specific_type, category, 1.0, "dictionary"
                    )

                # Check variations
                for variation in variations:
                    if cleaned_text == variation:
                        if "campari" in ingredient_text.lower():
                            print(
                                f"DEBUG: Variation match found - cleaned_text='{cleaned_text}' == variation='{variation}' -> specific_type='{specific_type}' in category='{category}'"
                            )
                        return IngredientMatch(
                            brand, specific_type, category, 1.0, "dictionary"
                        )

        # Try word boundary matches (more precise than simple contains)
        for category, types in self.taxonomy.items():
            for specific_type, variations in types.items():
                # Check if specific type exists as whole words in cleaned text
                if self._contains_as_words(cleaned_text, specific_type):
                    if "campari" in ingredient_text.lower():
                        print(
                            f"DEBUG: Word boundary match with specific_type - '{specific_type}' found in '{cleaned_text}' for category='{category}'"
                        )
                    return IngredientMatch(
                        brand, specific_type, category, 0.9, "dictionary"
                    )

                # Check variations with word boundaries
                for variation in variations:
                    if self._contains_as_words(cleaned_text, variation):
                        if "campari" in ingredient_text.lower():
                            print(
                                f"DEBUG: Word boundary match with variation - '{variation}' found in '{cleaned_text}' -> specific_type='{specific_type}' in category='{category}'"
                            )
                        return IngredientMatch(
                            brand, specific_type, category, 0.9, "dictionary"
                        )

        # Last resort: partial matches only if they're meaningful
        for category, types in self.taxonomy.items():
            for specific_type, variations in types.items():
                # Only do partial matching for longer terms (4+ chars) to avoid false positives
                if len(specific_type) >= 4 and specific_type in cleaned_text:
                    # Additional check: make sure it's not a misleading match
                    if self._is_meaningful_partial_match(cleaned_text, specific_type):
                        if "campari" in ingredient_text.lower():
                            print(
                                f"DEBUG: Partial match with specific_type - '{specific_type}' found in '{cleaned_text}' for category='{category}'"
                            )
                        return IngredientMatch(
                            brand, specific_type, category, 0.7, "dictionary"
                        )

                for variation in variations:
                    if len(variation) >= 4 and variation in cleaned_text:
                        if self._is_meaningful_partial_match(cleaned_text, variation):
                            if "campari" in ingredient_text.lower():
                                print(
                                    f"DEBUG: Partial match with variation - '{variation}' found in '{cleaned_text}' -> specific_type='{specific_type}' in category='{category}'"
                                )
                            return IngredientMatch(
                                brand, specific_type, category, 0.7, "dictionary"
                            )

        if "campari" in ingredient_text.lower():
            print(
                f"DEBUG: No match found for '{ingredient_text}' (cleaned: '{cleaned_text}')"
            )
        return None

    def _contains_as_words(self, text: str, term: str) -> bool:
        """Check if term exists as complete words in text.

        Uses word boundary regex to ensure the term appears as whole words
        rather than as part of larger words.

        Args:
            text: The text to search in.
            term: The term to search for.

        Returns:
            True if term exists as complete words in text, False otherwise.
        """
        import re

        # Create word boundary pattern
        pattern = r"\b" + re.escape(term) + r"\b"
        return bool(re.search(pattern, text, re.IGNORECASE))

    def _is_meaningful_partial_match(self, text: str, term: str) -> bool:
        """Check if partial match is meaningful and not misleading.

        Analyzes partial matches to avoid false positives like matching "bitter"
        in "bitters" when looking for different ingredients. Uses context
        analysis to determine if the match is semantically meaningful.

        Args:
            text: The text containing the potential match.
            term: The term being matched.

        Returns:
            True if the partial match is meaningful, False otherwise.
        """
        # Avoid matching if the term is at the end of a compound word
        # e.g., avoid matching "bitter" in "bitters" when looking for different ingredients

        # If the match is at word boundaries, it's likely meaningful
        if self._contains_as_words(text, term):
            return True

        # If the term appears as part of a larger word, be more cautious
        # Allow it only if it makes sense (e.g., "rum" in "aged rum")
        import re

        matches = list(re.finditer(re.escape(term), text, re.IGNORECASE))
        for match in matches:
            start, end = match.span()
            # Check characters before and after
            before = text[start - 1] if start > 0 else " "
            after = text[end] if end < len(text) else " "

            # If surrounded by spaces or punctuation, it's likely a good match
            if before in " -_" and after in " -_":
                return True

        return False

    def query_llm_batch(self, ingredients: List[str]) -> List[IngredientMatch]:
        """Query LLM for batch processing of ingredients that couldn't be matched.

        Uses AWS Bedrock to process ingredients that weren't matched by the
        dictionary lookup. Implements caching to avoid repeated API calls
        for the same ingredients.

        Args:
            ingredients: List of ingredient strings to process with LLM.

        Returns:
            List of IngredientMatch objects corresponding to the input ingredients.
            Returns error matches for ingredients that fail processing.
        """
        # Check cache first
        uncached_ingredients = []
        cached_results = {}

        for ingredient in ingredients:
            if ingredient in self.llm_cache:
                cached_results[ingredient] = self.llm_cache[ingredient]
            else:
                uncached_ingredients.append(ingredient)

        results = []

        if uncached_ingredients:
            # Construct batch prompt
            prompt = self._create_batch_prompt(uncached_ingredients)

            try:
                response = self.bedrock_client.invoke_model(
                    modelId="amazon.nova-lite-v1:0",
                    body=json.dumps(
                        {
                            "schemaVersion": "messages-v1",
                            "messages": [
                                {"role": "user", "content": [{"text": prompt}]}
                            ],
                            "inferenceConfig": {
                                "maxTokens": 2000,
                                "temperature": 0.1,
                            },
                        }
                    ),
                )

                response_body = json.loads(response["body"].read())
                llm_results = self._parse_llm_response(
                    response_body["output"]["message"]["content"][0]["text"],
                    uncached_ingredients,
                )

                # Cache results
                for ingredient, result in zip(uncached_ingredients, llm_results):
                    self.llm_cache[ingredient] = result

                results.extend(llm_results)

            except Exception as e:
                print(f"LLM query failed: {e}")
                # Return default results for failed queries
                results.extend(
                    [
                        IngredientMatch(None, ingredient, "unknown", 0.3, "error")
                        for ingredient in uncached_ingredients
                    ]
                )

        # Add cached results
        for ingredient in ingredients:
            if ingredient in cached_results:
                results.append(cached_results[ingredient])

        return results

    def _create_batch_prompt(self, ingredients: List[str]) -> str:
        """Create prompt for batch LLM processing of ingredients.

        Formats a list of ingredients into a structured prompt for the LLM
        that includes examples and specific instructions for parsing.

        Args:
            ingredients: List of ingredient strings to include in the prompt.

        Returns:
            A formatted prompt string ready for LLM processing.
        """
        numbered_ingredients = "\n".join(
            [f'{i + 1}. "{ingredient}"' for i, ingredient in enumerate(ingredients)]
        )

        return f"""Parse these cocktail/recipe ingredients into the format: Brand : Specific Type : Category

Examples:
- "blended aged rum, preferably Banks 7" → "Banks 7 : blended aged rum : rum"
- "dry Curaçao, preferably Pierre Ferrand" → "Pierre Ferrand : dry curaçao : orange liqueur"
- "blanc vermouth" → "None : blanc vermouth : vermouth"
- "Campari" → "None : campari : liqueur"
- "bombay sapphire gin" → "bombay sapphire : gin : gin"

Notes: 
  - For some ingredients, the specific type and category may be the same (e.g., "campari : liqueur" where campari is both the specific type and falls under the liqueur category).
  - Don't use apertif as a category
  - Rich syrup is the same as 2:1 syrup and not the same as simple syrup

Ingredients to parse:
{numbered_ingredients}

Please respond with exactly {len(ingredients)} lines in the format:
1. Brand : Specific Type : Category
2. Brand : Specific Type : Category
...

Use "None" for brand if no specific brand is mentioned. Common categories include: rum, whiskey, gin, vermouth, orange liqueur, bitters, syrup, juice, liqueur, etc."""

    def _parse_llm_response(
        self, response: str, original_ingredients: List[str]
    ) -> List[IngredientMatch]:
        """Parse LLM response text into IngredientMatch objects.

        Converts the structured text response from the LLM into a list of
        IngredientMatch objects, handling malformed responses gracefully.

        Args:
            response: The raw text response from the LLM.
            original_ingredients: List of original ingredient names for fallback.

        Returns:
            List of IngredientMatch objects parsed from the response.
            Uses error matches for malformed or missing responses.
        """
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        results = []

        for i, line in enumerate(lines):
            if i >= len(original_ingredients):
                break

            # Remove numbering if present
            line = re.sub(r"^\d+\.\s*", "", line)

            try:
                parts = [part.strip() for part in line.split(" : ")]
                if len(parts) == 3:
                    brand = parts[0] if parts[0].lower() != "none" else None
                    specific_type = parts[1]
                    category = parts[2]
                    results.append(
                        IngredientMatch(brand, specific_type, category, 0.7, "llm")
                    )
                else:
                    # Fallback for malformed response
                    results.append(
                        IngredientMatch(
                            None, original_ingredients[i], "unknown", 0.3, "llm_error"
                        )
                    )
            except Exception:
                results.append(
                    IngredientMatch(
                        None, original_ingredients[i], "unknown", 0.3, "llm_error"
                    )
                )

        # Handle case where LLM returned fewer results than expected
        while len(results) < len(original_ingredients):
            idx = len(results)
            results.append(
                IngredientMatch(
                    None, original_ingredients[idx], "unknown", 0.3, "llm_missing"
                )
            )

        return results

    def parse_ingredients(self, ingredients: List[str]) -> List[IngredientMatch]:
        """Main method to parse a list of ingredients into structured matches.

        Uses a two-pass approach: first attempts dictionary lookup for fast
        matching, then processes remaining ingredients with LLM in batches.

        Args:
            ingredients: List of raw ingredient description strings.

        Returns:
            List of IngredientMatch objects corresponding to each input ingredient.
            Results maintain the same order as the input list.
        """
        results = [None] * len(ingredients)  # Pre-allocate results list
        llm_queue = []
        llm_indices = []  # Track which positions need LLM processing

        print(f"DEBUG: parse_ingredients called with {len(ingredients)} ingredients")

        # First pass: try dictionary lookup
        for i, ingredient in enumerate(ingredients):
            match = self.dictionary_lookup(ingredient)
            if match and match.confidence >= 0.8:
                print(
                    f"DEBUG: Dictionary match for ingredient #{i} '{ingredient}': {match.specific_type} -> {match.category} (confidence: {match.confidence})"
                )
                results[i] = match  # Store result at original position
            else:
                print(
                    f"DEBUG: No dictionary match for ingredient #{i} '{ingredient}', adding to LLM queue"
                )
                llm_queue.append(ingredient)
                llm_indices.append(i)  # Track the original position

        print(
            f"DEBUG: After dictionary lookup: {len([r for r in results if r is not None])} matches, {len(llm_queue)} for LLM"
        )

        # Second pass: process unknowns with LLM in batches
        if llm_queue:
            batch_size = 10  # Process in batches to manage token limits
            for i in range(0, len(llm_queue), batch_size):
                batch = llm_queue[i : i + batch_size]
                batch_indices = llm_indices[i : i + batch_size]
                batch_results = self.query_llm_batch(batch)
                print(
                    f"DEBUG: LLM batch {i // batch_size + 1} processed {len(batch)} ingredients, got {len(batch_results)} results"
                )

                # Store LLM results at their original positions
                for j, result in enumerate(batch_results):
                    original_index = batch_indices[j]
                    results[original_index] = result

        # Convert to regular list (removing None values if any)
        final_results = [r for r in results if r is not None]

        print(
            f"DEBUG: Final results count: {len(final_results)} for {len(ingredients)} input ingredients"
        )
        return final_results

    def update_taxonomy(self, new_entries: List[IngredientMatch]):
        """Update taxonomy with confirmed entries from manual or high-confidence LLM matches.

        Adds new ingredient classifications to the taxonomy for future use.
        Only processes entries from manual sources or LLM matches with high confidence.

        Args:
            new_entries: List of IngredientMatch objects to potentially add to taxonomy.
        """
        for entry in new_entries:
            if entry.source == "manual" or (
                entry.source == "llm" and entry.confidence >= 0.8
            ):
                category = entry.category
                specific_type = entry.specific_type

                if category not in self.taxonomy:
                    self.taxonomy[category] = {}

                if specific_type not in self.taxonomy[category]:
                    self.taxonomy[category][specific_type] = []

    def save_taxonomy(self, filename: str = "ingredient_taxonomy.json"):
        """Save updated taxonomy to JSON file.

        Persists the current taxonomy dictionary to disk for future use.

        Args:
            filename: Path where to save the taxonomy file.
                Defaults to "ingredient_taxonomy.json".
        """
        with open(filename, "w") as f:
            json.dump(self.taxonomy, f, indent=2)

    def generate_report(self, results: List[IngredientMatch]) -> Dict:
        """Generate processing report with statistics and quality metrics.

        Creates a summary report of the ingredient processing results including
        match counts, confidence levels, and items needing manual review.

        Args:
            results: List of IngredientMatch objects to analyze.

        Returns:
            Dictionary containing processing statistics including:
                - total_processed: Total number of ingredients processed
                - dictionary_matches: Count of dictionary-based matches
                - llm_processed: Count of LLM-processed ingredients
                - manual_review_needed: Count of low-confidence matches
                - categories_found: List of unique categories identified
                - low_confidence_items: List of items needing review
        """
        report = {
            "total_processed": len(results),
            "dictionary_matches": len([r for r in results if r.source == "dictionary"]),
            "llm_processed": len([r for r in results if r.source == "llm"]),
            "manual_review_needed": len([r for r in results if r.confidence < 0.7]),
            "categories_found": list(set(r.category for r in results)),
            "low_confidence_items": [
                f"{r.brand or 'None'} : {r.specific_type} : {r.category}"
                for r in results
                if r.confidence < 0.7
            ],
        }
        return report

    def get_ingredients_from_db(
        self, min_recipe_count: int = 1
    ) -> List[IngredientUsage]:
        """Retrieve ingredients from the database with usage statistics.

        Queries the SQLite database to get all ingredients along with their
        usage frequency and sample recipes. Filters ingredients by minimum
        recipe count to focus on commonly used ingredients.

        Args:
            min_recipe_count: Minimum number of recipes an ingredient must
                appear in to be included. Defaults to 1.

        Returns:
            List of IngredientUsage objects containing ingredient names,
            recipe counts, and sample recipe names. Empty list if database
            error occurs.
        """
        ingredients = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get ingredients with their usage count and sample recipes
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

    def get_ingredient_details_from_db(self, ingredient_name: str) -> Dict:
        """Get detailed information about a specific ingredient from the database.

        Retrieves all recipes that use the specified ingredient along with
        amounts, units, notes, and source URLs for detailed analysis.

        Args:
            ingredient_name: The exact name of the ingredient to query.

        Returns:
            Dictionary containing:
                - ingredient_name: The queried ingredient name
                - recipes: List of recipe details with amounts and notes
                - total_recipes: Count of recipes using this ingredient
            Returns empty dict if database error occurs.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = """
            SELECT 
                r.name as recipe_name,
                ri.amount,
                ri.unit,
                ri.note,
                r.source_url
            FROM ingredient i
            JOIN recipe_ingredient ri ON i.id = ri.ingredient_id
            JOIN recipe r ON ri.recipe_id = r.id
            WHERE i.name = ?
            ORDER BY r.name
            """

            cursor.execute(query, (ingredient_name,))
            results = cursor.fetchall()

            recipes = []
            for recipe_name, amount, unit, note, source_url in results:
                recipes.append(
                    {
                        "recipe_name": recipe_name,
                        "amount": amount,
                        "unit": unit,
                        "note": note,
                        "source_url": source_url,
                    }
                )

            return {
                "ingredient_name": ingredient_name,
                "recipes": recipes,
                "total_recipes": len(recipes),
            }

        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return {}
        finally:
            if conn:
                conn.close()

    def initialize_output_file(self):
        """Initialize the output file for progressive writing of results.

        Sets up the output file format (CSV or JSON) and writes appropriate
        headers. For JSON, starts the array structure. For CSV, writes column headers.

        Raises:
            ValueError: If an unsupported output format is specified.
        """
        if self.output_format == "csv":
            self.output_handle = open(
                self.output_file, "w", newline="", encoding="utf-8"
            )
            self.csv_writer = csv.writer(self.output_handle)
            # Write CSV headers
            headers = [
                "original_name",
                "brand",
                "specific_type",
                "category",
                "confidence",
                "source",
                "recipe_count",
                "sample_recipes",
            ]
            self.csv_writer.writerow(headers)
        elif self.output_format == "json":
            self.output_handle = open(self.output_file, "w", encoding="utf-8")
            # Start JSON array
            self.output_handle.write("[\n")
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

    def write_result(self, result: "IngredientMatch", usage: "IngredientUsage"):
        """Write a single result to the output file.

        Writes one ingredient analysis result to the output file in the
        configured format (CSV or JSON). Handles file initialization
        automatically if needed.

        Args:
            result: IngredientMatch object containing parsed ingredient data.
            usage: IngredientUsage object containing recipe usage statistics.
        """
        if not self.output_handle:
            self.initialize_output_file()

        # Debug logging for campari
        if "campari" in usage.ingredient_name.lower():
            print(
                f"DEBUG: Writing result for '{usage.ingredient_name}': {result.brand} : {result.specific_type} : {result.category} (confidence: {result.confidence}, source: {result.source})"
            )

        if self.output_format == "csv":
            row = [
                usage.ingredient_name,
                result.brand or "",
                result.specific_type,
                result.category,
                result.confidence,
                result.source,
                usage.recipe_count,
                ", ".join(usage.sample_recipes),
            ]
            self.csv_writer.writerow(row)
            self.output_handle.flush()  # Ensure data is written immediately
        elif self.output_format == "json":
            result_data = {
                "original_name": usage.ingredient_name,
                "brand": result.brand,
                "specific_type": result.specific_type,
                "category": result.category,
                "confidence": result.confidence,
                "source": result.source,
                "recipe_count": usage.recipe_count,
                "sample_recipes": usage.sample_recipes,
            }

            # Add comma if not the first result
            if self.results_written > 0:
                self.output_handle.write(",\n")

            json.dump(result_data, self.output_handle, indent=2)
            self.output_handle.flush()  # Ensure data is written immediately

        self.results_written += 1

    def finalize_output_file(self):
        """Finalize and close the output file.

        Completes the output file format (closes JSON array if needed)
        and closes the file handle. Prints confirmation message with filename.
        """
        if self.output_handle:
            if self.output_format == "json":
                self.output_handle.write("\n]")
            self.output_handle.close()
            self.output_handle = None
            print(f"Results written to {self.output_file}")

    def __del__(self):
        """Ensure output file is properly closed when object is destroyed.

        Destructor that safely closes the output file if it's still open
        when the IngredientParser object is garbage collected.
        """
        if hasattr(self, "output_handle") and self.output_handle:
            self.finalize_output_file()


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
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
        # Get ingredients from database
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
        print(f"Writing results to {parser.output_file} as {args.format.upper()}")

        # Extract just the ingredient names for processing
        ingredient_names = [usage.ingredient_name for usage in ingredient_usages]

        # Parse ingredients in batches and write results progressively
        print("Processing ingredients...")
        batch_size = 10  # Process in batches to manage token limits
        total_processed = 0

        for i in range(0, len(ingredient_names), batch_size):
            batch_names = ingredient_names[i : i + batch_size]
            batch_usages = ingredient_usages[i : i + batch_size]

            # Parse this batch
            batch_results = parser.parse_ingredients(batch_names)

            # Write results immediately
            for result, usage in zip(batch_results, batch_usages):
                parser.write_result(result, usage)
                total_processed += 1

                # Show progress
                if total_processed % 50 == 0 or total_processed == len(
                    ingredient_usages
                ):
                    print(
                        f"Processed {total_processed}/{len(ingredient_usages)} ingredients..."
                    )

        # Finalize output file
        parser.finalize_output_file()

        # Generate final report
        print("\nProcessing Complete!")
        print(f"- Total ingredients processed: {total_processed}")
        print(f"- Results saved to: {parser.output_file}")
        print(f"- Format: {args.format.upper()}")

        # Save updated taxonomy
        parser.save_taxonomy()
        print("- Taxonomy saved to: ingredient_taxonomy.json")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        parser.finalize_output_file()
        print("Partial results have been saved")
    except Exception as e:
        print(f"\nError during processing: {e}")
        parser.finalize_output_file()
        raise
