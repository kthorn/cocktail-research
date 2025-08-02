import datetime
import json
import os
import re
import sqlite3
from typing import List, Optional, Tuple

import boto3

from cocktail_utils.database import get_connection
from cocktail_utils.ingredients import normalize_ingredient_text
from cocktail_utils.ingredients.models import IngredientMatch, IngredientUsage


def _get_default_brand_patterns() -> List[re.Pattern]:
    """Get default regex patterns for brand extraction from ingredient text.

    Creates regex patterns to identify and extract brand names from
    ingredient descriptions. Looks for common brand reference patterns
    like "preferably X", "such as X", "like X", and brands mentioned
    at the end of ingredient descriptions.

    Returns:
        A list of compiled regex patterns for brand extraction, ordered
        by specificity with more specific patterns first.

    Examples:
        The patterns will match:
        - "gin, preferably Hendrick's" -> captures "Hendrick's"
        - "vermouth such as Dolin" -> captures "Dolin"
        - "whiskey like Jameson" -> captures "Jameson"
        - "rum, Bacardi" -> captures "Bacardi"
    """
    return [
        # Patterns for parenthetical brand references
        re.compile(
            r"\s*\(preferably\s+([A-Za-z\u00C0-\u024F][\w\s&'\-\d\.]+)\)", re.IGNORECASE
        ),
        re.compile(
            r"\s*\(such as\s+([A-Za-z\u00C0-\u024F][\w\s&'\-\d\.]+)\)", re.IGNORECASE
        ),
        re.compile(
            r"\s*\(like\s+([A-Za-z\u00C0-\u024F][\w\s&'\-\d\.]+)\)", re.IGNORECASE
        ),
        # Patterns for non-parenthetical brand references
        re.compile(
            r",?\s*preferably\s+([A-Za-z\u00C0-\u024F][\w\s&'\-\d\.]+)", re.IGNORECASE
        ),
        re.compile(
            r",?\s*such as\s+([A-Za-z\u00C0-\u024F][\w\s&'\-\d\.]+)", re.IGNORECASE
        ),
        re.compile(
            r",?\s*like\s+([A-Za-z\u00C0-\u024F][\w\s&'\-\d\.]+)", re.IGNORECASE
        ),
        # Pattern for brands at the end without "preferably"
        re.compile(r",\s+([A-Za-z\u00C0-\u024F][\w\s&'\-\d\.]+)$", re.IGNORECASE),
    ]


# Common usage instructions that should not be treated as brands
USAGE_INSTRUCTIONS = {
    "to rinse",
    "to rinse the glass",
    "to float",
    "to top",
    "for spritzing",
    "in atomizer",
    "for rinse",
    "as needed",
    "rinse",
    "for garnish",
    "to taste",
    "to coat",
    "to mist",
    "to spray",
    "for washing",
    "to rim",
    "to rim the glass",
}


class IngredientParser:
    """A parser for analyzing and categorizing cocktail ingredients.

    This class provides functionality to parse ingredient names from cocktail recipes,
    extract brand information, categorize ingredients using a taxonomy, and perform
    LLM-based ingredient analysis.

    Attributes:
        taxonomy (dict): A dictionary containing ingredient categories and their variations.
        brand_dictionary (dict): A dictionary mapping brand names to categories.
        bedrock_client: AWS Bedrock client for LLM operations.
        llm_cache (dict): Cache for storing LLM responses to avoid repeated API calls.
        db_path (str): Path to the SQLite database containing recipe data.
        output_format (str): Format for output files (json, csv, etc.).
        output_file (str): Name of the output file for analysis results.
        output_handle: File handle for writing output.
        csv_writer: CSV writer object for CSV output format.
        results_written (int): Counter for tracking written results.
    """

    def __init__(
        self,
        taxonomy_file: str = os.path.join(
            os.path.dirname(__file__), "data", "ingredient_taxonomy.json"
        ),
        brand_file: str = os.path.join(
            os.path.dirname(__file__), "data", "brand_dictionary.json"
        ),
        db_path: str = "data/punch_recipes.db",
        output_format: str = "json",
        output_file: str = None,
    ):
        """Initialize the IngredientParser with configuration options.

        Args:
            taxonomy_file (str): Path to the JSON file containing ingredient taxonomy.
                Defaults to the bundled taxonomy file.
            brand_file (str): Path to the JSON file containing brand dictionary.
                Defaults to the bundled brand dictionary file.
            db_path (str): Path to the SQLite database containing recipe data.
                Defaults to "data/punch_recipes.db".
            output_format (str): Format for output files. Supported formats: "json", "csv".
                Defaults to "json".
            output_file (str, optional): Custom name for the output file. If None,
                a timestamped filename will be generated automatically.
        """
        self.taxonomy = json.load(open(taxonomy_file, "r", encoding="utf-8"))
        self.brand_dictionary = json.load(open(brand_file, "r", encoding="utf-8"))
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
        self.llm_cache = {}  # Cache LLM responses
        self.db_path = db_path
        self.output_format = output_format.lower()
        self.output_file = output_file or (
            f"ingredient_analysis_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}."
            f"{self.output_format}"
        )
        self.output_handle = None
        self.csv_writer = None
        self.results_written = 0
        self.brand_patterns = _get_default_brand_patterns()

    def _contains_as_words(self, text: str, terms: List[str]) -> bool:
        """Check if any term exists as complete words in text.

        This helper method uses regex to check if any of the provided terms
        exist as complete words (not substrings) within the given text.

        Args:
            text (str): The text to search within.
            terms (List[str]): List of terms to search for as complete words.

        Returns:
            bool: True if any term is found as a complete word, False otherwise.
        """
        for term in terms:
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _extract_brand_from_pattern(
        self, ingredient_text: str
    ) -> Tuple[Optional[str], str]:
        """Extract brand name and return cleaned ingredient text.

        Searches for brand references in ingredient text using compiled patterns
        and removes them to get the clean ingredient name. Handles common
        brand reference patterns like "preferably X", "such as X", etc.

        Args:
            ingredient_text: The raw ingredient description text that may
                            contain brand references.
            brand_patterns: Optional list of compiled regex patterns for brand
                        extraction. If None, uses default patterns that handle
                        common brand reference formats.

        Returns:
            A tuple containing:
                - brand: Brand name if found, None otherwise
                - cleaned_text: Ingredient text with brand reference removed
        """

        for pattern in self.brand_patterns:
            match = pattern.search(ingredient_text)
            if match:
                brand = match.group(1).strip().lower()

                # Filter out common usage instructions that aren't brands
                if brand in USAGE_INSTRUCTIONS:
                    continue

                # Remove the brand reference from the text
                cleaned_text = pattern.sub("", ingredient_text).strip().rstrip(",")
                # Clean up any empty parentheses left behind
                cleaned_text = re.sub(r"\s*\(\s*\)", "", cleaned_text).strip()
                return brand, cleaned_text
        return None, ingredient_text

    def extract_brand(self, ingredient_text: str) -> Tuple[Optional[str], str]:
        """Extract brand name using brand dictionary and patterns.

        This method first checks for exact matches in the brand dictionary, then
        falls back to pattern-based brand extraction using regex patterns.

        Args:
            ingredient_text (str): The raw ingredient text to analyze.

        Returns:
            Tuple[Optional[str], str]: A tuple containing:
                - The extracted brand name (or None if no brand found)
                - The cleaned ingredient text with brand removed
        """
        # Check if there is an exact match in brand_dictionary
        brand_category = self.brand_dictionary.get(ingredient_text)
        if brand_category:
            return ingredient_text, brand_category

        # Fall back to pattern-based brand extraction
        return self._extract_brand_from_pattern(ingredient_text)

    def rationalize_ingredient(self, ingredient_text: str) -> Optional[IngredientMatch]:
        """Identify brand of ingredient and assign it to a taxonomy category if possible

        This method performs dictionary-based ingredient matching by normalizing
        the input text and searching for exact matches or partial matches within
        the ingredient taxonomy. More specific matches are prioritized over less
        specific ones.

        Args:
            ingredient_text (str): The ingredient text to match against the taxonomy.

        Returns:
            Optional[IngredientMatch]: An IngredientMatch object if a match is found,
                containing brand, specific_type, category, confidence, and source.
                Returns None if no match is found.
        """
        # Use library's normalization function
        normalized = normalize_ingredient_text(ingredient_text)
        brand, cleaned_text = self.extract_brand(normalized)

        # Collect all possible matches with their specificity
        matches = []

        # Try exact matches first - prioritize specific_type matches over
        # category matches to handle cases where the same term appears as both
        for category, types in self.taxonomy.items():
            for specific_type, variations in types.items():
                all_variations = [specific_type] + variations
                for variation in all_variations:
                    if cleaned_text == variation:
                        matches.append(
                            (
                                IngredientMatch(
                                    brand, specific_type, category, 1.0, "dictionary"
                                ),
                                len(variation),  # specificity score
                            )
                        )
                    elif self._contains_as_words(cleaned_text, [variation]):
                        matches.append(
                            (
                                IngredientMatch(
                                    brand, specific_type, category, 0.9, "dictionary"
                                ),
                                len(variation),  # specificity score
                            )
                        )

        # Only check category matches if no specific_type match was found
        if not matches:
            for category, types in self.taxonomy.items():
                if cleaned_text == category:
                    matches.append(
                        (
                            IngredientMatch(brand, None, category, 1.0, "dictionary"),
                            len(category),  # specificity score
                        )
                    )

        # Return the most specific match (longest matched term)
        if matches:
            # Sort by confidence first (1.0 before 0.9), then by specificity
            matches.sort(key=lambda x: (x[0].confidence, x[1]), reverse=True)
            return matches[0][0]

        return None

    def get_ingredients_from_db(
        self, min_recipe_count: int = 1
    ) -> List[IngredientUsage]:
        """Retrieve ingredients from the database with usage statistics.

        This method queries the SQLite database to get ingredients along with
        their usage statistics, including the number of recipes they appear in
        and sample recipe names.

        Args:
            min_recipe_count (int): Minimum number of recipes an ingredient must
                appear in to be included in the results. Defaults to 1.

        Returns:
            List[IngredientUsage]: A list of IngredientUsage objects containing
                ingredient names, recipe counts, and sample recipe names.

        Note:
            This method handles database errors gracefully and returns an empty
            list if any database errors occur.
        """
        ingredients = []

        try:
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
                sample_list = sample_recipes.split(",")[:3] if sample_recipes else []
                ingredients.append(IngredientUsage(name, count, sample_list))

        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()

        return ingredients

    def llm_lookup(
        self, ingredient_text: str, model_id: str
    ) -> Optional[IngredientMatch]:
        """Perform LLM-based ingredient analysis using AWS Bedrock.

        This method uses various LLM models available through AWS Bedrock to
        analyze ingredient names and extract structured information including
        brand, specific type, and category. Results are cached to avoid
        repeated API calls for the same input.

        Args:
            ingredient_text (str): The ingredient text to analyze.
            model_id (str): The AWS Bedrock model ID to use for analysis.
                Supported models include Claude, Titan, and Nova variants.

        Returns:
            Optional[IngredientMatch]: An IngredientMatch object containing the
                LLM's analysis results, or None if the analysis fails.

        Note:
            This method handles various LLM response formats and includes
            error handling for API failures. The confidence score is set to 0.8
            for all LLM-based matches.
        """
        if (ingredient_text, model_id) in self.llm_cache:
            return self.llm_cache[(ingredient_text, model_id)]

        # Load taxonomy for existing categories
        taxonomy_path = os.path.join(
            os.path.dirname(__file__), "data", "ingredient_taxonomy.json"
        )
        with open(taxonomy_path, "r") as f:
            taxonomy = json.load(f)

        categories = list(taxonomy.keys())
        categories_str = ", ".join(categories)

        prompt = f"""
You are an expert mixologist. Your task is to analyze an ingredient name from a
cocktail recipe and break it down into its constituent parts: brand, specific
type, and category.

Here is the ingredient name:
"{ingredient_text}"

For the category field, please use one of these existing categories when possible:
{categories_str}

Based on this, provide the following information in JSON format:
{{
  "brand": "The brand name, if present. Otherwise, null.",
  "specific_type": "The specific type of the ingredient (e.g., 'London Dry Gin',
    'Angostura Aromatic Bitters').",
  "category": "The general category of the ingredient. Use one of the existing categories above when appropriate."
}}
"""

        body = ""
        if "anthropic.claude-3" in model_id:
            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                }
            )
        elif "amazon.nova-lite" in model_id:
            body = json.dumps(
                {
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                }
            )
        else:  # Legacy Claude
            body = json.dumps(
                {
                    "prompt": f"\n\nHuman:{prompt}\n\nAssistant:",
                    "max_tokens_to_sample": 500,
                    "temperature": 0.1,
                    "top_p": 0.9,
                }
            )

        try:
            if model_id == "anthropic.claude-3-5-haiku-20241022-v1:0":
                response = self.bedrock_client.invoke_model(
                    body=body,
                    modelId="arn:aws:bedrock:us-east-1:732940910135:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0",
                    accept="application/json",
                    contentType="application/json",
                )
            else:
                response = self.bedrock_client.invoke_model(
                    body=body,
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json",
                )
            response_body = json.loads(response.get("body").read())

            completion = ""
            if "anthropic.claude-3" in model_id:
                completion = response_body.get("content", [{}])[0].get("text", "")
            elif "amazon.titan" in model_id:
                completion = response_body.get("results")[0].get("outputText")
            elif "amazon.nova-lite" in model_id:
                completion = (
                    response_body.get("output", {})
                    .get("message", {})
                    .get("content", [{}])[0]
                    .get("text", "")
                )
            else:  # Legacy Claude
                completion = response_body.get("completion", "")

            if "```json" in completion:
                completion = completion.split("```json")[1].split("```")[0]
            elif completion.strip().startswith("```json"):
                completion = completion.strip()[7:-3].strip()
            else:
                # Attempt to find the first and last curly brace to extract JSON
                start_index = completion.find("{")
                end_index = completion.rfind("}")
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    completion = completion[start_index : end_index + 1]
                else:
                    # If no JSON found, set completion to empty string to avoid parsing errors
                    completion = ""

            parsed = json.loads(completion.strip())

            # Ensure all fields are strings, not lists
            def safe_string_extract(value):
                if value is None:
                    return None
                elif isinstance(value, list):
                    # If it's a list, take the first non-empty item
                    return next(
                        (item for item in value if item and str(item).strip()), None
                    )
                else:
                    return str(value).strip() if str(value).strip() else None

            match = IngredientMatch(
                brand=safe_string_extract(parsed.get("brand")),
                specific_type=safe_string_extract(parsed.get("specific_type")),
                category=safe_string_extract(parsed.get("category")),
                confidence=0.8,
                source=f"llm:{model_id}",
            )
            self.llm_cache[(ingredient_text, model_id)] = match
            return match
        except Exception as e:
            print(
                f"Error during LLM lookup for '{ingredient_text}' with model {model_id}: {e}"
            )
            return None
