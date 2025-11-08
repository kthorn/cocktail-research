"""Recipe source utilities for handling multiple recipe sources."""

import glob
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup


def validate_url(url: str, recipe_name: str) -> bool:
    """Validate URL by making a HEAD request.

    Args:
        url: URL to validate
        recipe_name: Recipe name for error reporting

    Returns:
        True if URL is valid (status < 400), False otherwise
    """
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        if response.status_code >= 400:
            print(
                f"⚠️  INVALID URL for recipe '{recipe_name}': {url} (Status: {response.status_code})"
            )
            return False
        return True
    except requests.exceptions.RequestException as e:
        print(
            f"⚠️  URL VALIDATION ERROR for recipe '{recipe_name}': {url} (Error: {str(e)})"
        )
        return False


class RecipeSource(ABC):
    """Abstract base class for recipe sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this recipe source."""
        pass

    @property
    @abstractmethod
    def raw_recipes_dir(self) -> str:
        """Return the directory containing raw HTML files."""
        pass

    @property
    @abstractmethod
    def source_path_prefix(self) -> str:
        """Return the prefix used in database source_file paths."""
        pass

    @abstractmethod
    def find_html_files(self) -> List[str]:
        """Find all HTML recipe files for this source.

        Returns:
            List of absolute paths to HTML files
        """
        pass

    @abstractmethod
    def parse_recipe_from_html(self, html_content: str, html_file: str) -> Optional[Dict]:
        """Parse recipe from HTML content.

        Args:
            html_content: Raw HTML content
            html_file: Path to the HTML file (for error reporting)

        Returns:
            Dict with keys: name, ingredients, instructions, description
            or None if parsing fails
        """
        pass

    @abstractmethod
    def derive_source_url(self, recipe_name: str) -> str:
        """Derive the source URL from a recipe name.

        Args:
            recipe_name: Name of the recipe

        Returns:
            Source URL for the recipe
        """
        pass

    @abstractmethod
    def clean_html_content(self, html_content: str) -> str:
        """Clean HTML content for display.

        Args:
            html_content: Raw HTML content

        Returns:
            Cleaned HTML content for display in validator
        """
        pass

    def get_db_path_from_html_path(self, html_file: str) -> str:
        """Convert an absolute HTML file path to the database source_file format.

        Args:
            html_file: Absolute path to HTML file

        Returns:
            Database source_file path (e.g., "raw_recipes/punch_html/recipe.html")
        """
        path_parts = html_file.split(os.sep)
        if len(path_parts) >= 2:
            # Take last 2 parts: source_dir/filename.html
            db_path = "/".join(path_parts[-2:])
            return f"raw_recipes/{db_path}"
        return html_file


class PunchRecipeSource(RecipeSource):
    """Recipe source for Punch Drink recipes."""

    @property
    def name(self) -> str:
        return "punch"

    @property
    def raw_recipes_dir(self) -> str:
        return "/home/kurtt/cocktail-research/raw_recipes/punch_html"

    @property
    def source_path_prefix(self) -> str:
        return "raw_recipes/punch_html"

    def find_html_files(self) -> List[str]:
        """Find all Punch HTML recipe files."""
        return sorted(glob.glob(f"{self.raw_recipes_dir}/*.html"))

    def parse_recipe_from_html(self, html_content: str, html_file: str) -> Optional[Dict]:
        """Parse Punch recipe from HTML."""
        try:
            from cocktail_utils.recipes.parsing import parse_recipe_html

            recipe = parse_recipe_html(html_content)
            if not recipe:
                return None

            # Convert to our format
            ingredients = []
            for ing_str in recipe.ingredients:
                # Parse ingredient string (e.g., "2 oz gin")
                # For now, keep as raw strings - rationalization happens later
                ingredients.append({
                    "ingredient_name": ing_str,
                    "amount": "",
                    "unit_name": ""
                })

            return {
                "name": recipe.name,
                "ingredients": ingredients,
                "instructions": " ".join(recipe.directions) if recipe.directions else "",
                "description": recipe.description or "",
            }
        except Exception as e:
            print(f"Error parsing Punch recipe from {html_file}: {e}")
            return None

    def derive_source_url(self, recipe_name: str) -> str:
        """Derive Punch source URL from recipe name."""
        clean_name = recipe_name.lower()

        # Replace unicode characters
        unicode_replacements = {
            "\u2019": "'",  # right single quotation mark
            "\u2018": "'",  # left single quotation mark
            "\u201c": '"',  # left double quotation mark
            "\u201d": '"',  # right double quotation mark
            "\u00a0": " ",  # non-breaking space
        }
        for unicode_char, replacement in unicode_replacements.items():
            clean_name = clean_name.replace(unicode_char, replacement)

        # Remove special characters
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in " -&")
        clean_name = clean_name.replace(" ", "-").replace("&", "and")
        clean_name = "-".join(filter(None, clean_name.split("-")))

        return f"https://punchdrink.com/recipes/{clean_name}/"

    def clean_html_content(self, html_content: str) -> str:
        """Clean Punch HTML content for display."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Find the main recipe content area
            recipe_content = None
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
                recipe_content = soup.find("body") or soup

            # Remove unwanted sections
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

            for element in recipe_content.find_all(text=True):
                if any(pattern in element.strip() for pattern in sharing_text_patterns):
                    parent = element.parent
                    if parent and parent.get_text().strip() in sharing_text_patterns:
                        parent.decompose()
                    else:
                        element.replace_with("")

            # Create clean HTML structure
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
            return html_content


class DiffordsRecipeSource(RecipeSource):
    """Recipe source for Difford's Guide recipes."""

    @property
    def name(self) -> str:
        return "diffords"

    @property
    def raw_recipes_dir(self) -> str:
        return "/home/kurtt/cocktail-research/raw_recipes/diffords_html"

    @property
    def source_path_prefix(self) -> str:
        return "raw_recipes/diffords_html"

    def find_html_files(self) -> List[str]:
        """Find all Difford's HTML recipe files."""
        # Exclude download_state.json
        all_files = glob.glob(f"{self.raw_recipes_dir}/*.html")
        return sorted([f for f in all_files if "download_state" not in f])

    def parse_recipe_from_html(self, html_content: str, html_file: str) -> Optional[Dict]:
        """Parse Difford's recipe from HTML."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Find recipe name
            name_tag = soup.find("h1")
            if not name_tag:
                return None

            name = name_tag.get_text(strip=True)

            # Find ingredients in the legacy-ingredients-table
            ingredients = []
            ingredient_table = soup.find("table", class_="legacy-ingredients-table")
            if ingredient_table:
                rows = ingredient_table.find_all("tr")
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 2:
                        # First cell: amount + unit
                        amount_unit = cells[0].get_text(strip=True)
                        # Second cell: ingredient name
                        ingredient_name = cells[1].get_text(strip=True)

                        if ingredient_name:
                            # For now, keep amount+unit together as a string
                            # The rationalization step will handle parsing
                            ingredients.append({
                                "ingredient_name": f"{amount_unit} {ingredient_name}",
                                "amount": "",
                                "unit_name": ""
                            })

            # Find instructions/method
            instructions = ""
            # Look for h2 with "How to make"
            h2_method = soup.find("h2", string=lambda x: x and "how to make" in x.lower())
            if h2_method:
                # Get all paragraph text after the h2 until next h2
                instruction_parts = []
                for sibling in h2_method.find_next_siblings():
                    if sibling.name == "h2":
                        break
                    if sibling.name == "p":
                        instruction_parts.append(sibling.get_text(strip=True))
                instructions = " ".join(instruction_parts)

            # Find description
            description = ""
            desc_elem = soup.find("meta", attrs={"name": "description"})
            if desc_elem:
                description = desc_elem.get("content", "")

            if not ingredients:
                print(f"Warning: No ingredients found in {html_file}")
                return None

            return {
                "name": name,
                "ingredients": ingredients,
                "instructions": instructions,
                "description": description,
            }

        except Exception as e:
            print(f"Error parsing Difford's recipe from {html_file}: {e}")
            return None

    def derive_source_url(self, recipe_name: str) -> str:
        """Derive Difford's source URL from recipe name.

        Note: This is a best-guess approach. Actual URLs may differ
        since Difford's uses numeric IDs in URLs.
        """
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

        # Remove special characters and create URL-safe name
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in " -")
        clean_name = clean_name.replace(" ", "-")
        clean_name = "-".join(filter(None, clean_name.split("-")))

        # Note: Real Difford's URLs include numeric IDs like:
        # https://www.diffordsguide.com/cocktails/recipe/1/abacaxi-ricaco
        # We could extract the ID from the HTML if needed
        return f"https://www.diffordsguide.com/cocktails/recipe/{clean_name}"

    def clean_html_content(self, html_content: str) -> str:
        """Clean Difford's HTML content for display."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Find the main recipe content area
            recipe_content = None
            selectors_to_try = [
                ".recipe-container",
                ".recipe-detail",
                "article",
                ".main-content",
                '[itemtype*="Recipe"]',
                "main",
            ]

            for selector in selectors_to_try:
                recipe_content = soup.select_one(selector)
                if recipe_content:
                    break

            if not recipe_content:
                recipe_content = soup.find("body") or soup

            # Remove unwanted sections
            unwanted_selectors = [
                "nav",
                "header",
                "footer",
                ".navigation",
                ".sidebar",
                ".ads",
                ".advertisement",
                "script",
                "style",
                ".social-share",
                "iframe",
                ".comments",
                "img",
                "picture",
                "figure",
            ]

            for selector in unwanted_selectors:
                for element in recipe_content.select(selector):
                    element.decompose()

            # Create clean HTML structure
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
            return html_content


# Registry of available recipe sources
RECIPE_SOURCES: Dict[str, RecipeSource] = {
    "punch": PunchRecipeSource(),
    "diffords": DiffordsRecipeSource(),
}


def get_recipe_source(name: str) -> Optional[RecipeSource]:
    """Get a recipe source by name.

    Args:
        name: Name of the recipe source (e.g., "punch", "diffords")

    Returns:
        RecipeSource instance or None if not found
    """
    return RECIPE_SOURCES.get(name.lower())


def get_all_recipe_sources() -> List[RecipeSource]:
    """Get all registered recipe sources.

    Returns:
        List of all RecipeSource instances
    """
    return list(RECIPE_SOURCES.values())
