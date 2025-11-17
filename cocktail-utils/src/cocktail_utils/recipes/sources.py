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
            from cocktail_utils.ingredients import parse_quantity

            recipe = parse_recipe_html(html_content)
            if not recipe:
                return None

            # Parse ingredients to extract amount, unit, and name
            ingredients = []
            for ing_str in recipe.ingredients:
                # Parse ingredient string (e.g., "2 oz gin")
                amount, unit, ingredient_name = parse_quantity(ing_str)

                # If parsing failed, keep the original string as ingredient name
                if ingredient_name is None:
                    ingredient_name = ing_str

                ingredients.append({
                    "ingredient_name": ingredient_name,
                    "amount": amount if amount is not None else "",
                    "unit_name": unit if unit is not None else ""
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
            from cocktail_utils.ingredients import parse_quantity

            soup = BeautifulSoup(html_content, "html.parser")

            # Extract canonical URL from <link rel="canonical"> in <head>
            canonical_link = soup.find("link", rel="canonical")
            source_url = canonical_link.get("href") if canonical_link else None

            # Find recipe name - use the h1 closest to the ingredients table
            # (Diffords pages have multiple h1s, first is often a banner/promo)
            ingredient_table = soup.find("table", class_="legacy-ingredients-table")
            if ingredient_table:
                # Find the nearest h1 before the table
                name_tag = ingredient_table.find_previous("h1")
            else:
                # Fallback: try the last h1 on the page
                all_h1s = soup.find_all("h1")
                name_tag = all_h1s[-1] if all_h1s else None

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
                            # Normalize unicode fractions (⁄ -> /) and add spaces
                            amount_unit = amount_unit.replace('\u2044', '/')  # fraction slash
                            # Add space between whole number and fraction (e.g., 21/2 -> 2 1/2)
                            amount_unit = re.sub(r'(\d)(\d/\d)', r'\1 \2', amount_unit)
                            # Add space between number and unit if missing
                            amount_unit = re.sub(r'(\d)(oz|ml|cl|dash|drop|tsp|tbsp|cup)', r'\1 \2', amount_unit)

                            # Combine and parse to extract amount, unit, and name
                            # Preserve parentheses for Diffords to retain context like "(or substitute X)"
                            full_ingredient = f"{amount_unit} {ingredient_name}"
                            amount, unit, parsed_name = parse_quantity(full_ingredient, preserve_parentheses=True)

                            # If parsing failed, use the ingredient name from the second cell
                            if parsed_name is None:
                                parsed_name = ingredient_name

                            ingredients.append({
                                "ingredient_name": parsed_name,
                                "amount": amount if amount is not None else "",
                                "unit_name": unit if unit is not None else ""
                            })

            # Extract instructions from "How to make:" section
            instructions = ""
            how_to_make_header = soup.find('h2', string='How to make:')
            if how_to_make_header:
                content = how_to_make_header.find_next_sibling()
                if content:
                    instructions = content.get_text(strip=True)

            # If no instructions found, use generic fallback
            if not instructions:
                instructions = "Shake all ingredients with ice and strain into a cocktail or coupe glass"

            # Extract description from "Review:" and "History:" sections
            description_parts = []

            review_header = soup.find('h2', string='Review:')
            if review_header:
                content = review_header.find_next_sibling()
                if content:
                    review_text = content.get_text(strip=True)
                    if review_text:
                        description_parts.append(review_text)

            history_header = soup.find('h2', string='History:')
            if history_header:
                content = history_header.find_next_sibling()
                if content:
                    history_text = content.get_text(strip=True)
                    if history_text:
                        description_parts.append(history_text)

            description = " ".join(description_parts)

            if not ingredients:
                print(f"Warning: No ingredients found in {html_file}")
                return None

            result = {
                "name": name,
                "ingredients": ingredients,
                "instructions": instructions,
                "description": description,
                "source": "Difford's Guide",
            }

            # Include source_url if we found it
            if source_url:
                result["source_url"] = source_url

            return result

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

            # Find the ingredients table - this is the core recipe content
            ingredient_table = soup.find("table", class_="legacy-ingredients-table")

            if not ingredient_table:
                # Fallback if no table found
                return html_content

            # Get the recipe name (h1 before the table)
            recipe_name_tag = ingredient_table.find_previous("h1")

            # Create a minimal clean HTML with recipe name, ingredients, and directions/description
            recipe_content_parts = []

            if recipe_name_tag:
                recipe_content_parts.append(str(recipe_name_tag))

            recipe_content_parts.append(str(ingredient_table))

            # Add "How to make:" section
            how_to_make_header = soup.find('h2', string='How to make:')
            if how_to_make_header:
                recipe_content_parts.append(str(how_to_make_header))
                content = how_to_make_header.find_next_sibling()
                if content:
                    recipe_content_parts.append(str(content))

            # Add "Review:" section
            review_header = soup.find('h2', string='Review:')
            if review_header:
                recipe_content_parts.append(str(review_header))
                content = review_header.find_next_sibling()
                if content:
                    recipe_content_parts.append(str(content))

            # Add "History:" section
            history_header = soup.find('h2', string='History:')
            if history_header:
                recipe_content_parts.append(str(history_header))
                content = history_header.find_next_sibling()
                if content:
                    recipe_content_parts.append(str(content))

            recipe_content = "".join(recipe_content_parts)

            # Create clean HTML structure
            clean_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Recipe</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; margin-bottom: 10px; margin-top: 20px; }}
                    table {{ border-collapse: collapse; margin-top: 10px; margin-bottom: 20px; }}
                    td, th {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    p {{ margin: 10px 0; line-height: 1.5; }}
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
