"""Recipe parsing utilities."""

import dataclasses
from typing import List, Optional

from bs4 import BeautifulSoup


@dataclasses.dataclass
class Recipe:
    """Dataclass for holding parsed recipe data."""

    name: str
    ingredients: List[str]
    directions: List[str]
    description: Optional[str] = None
    garnish: Optional[str] = None
    editors_note: Optional[str] = None


def parse_recipe_html(html: str) -> Optional[Recipe]:
    """Parse recipe HTML to extract structured data.

    Args:
        html: Raw HTML content of a recipe page.

    Returns:
        A Recipe object or None if parsing fails.
    """
    soup = BeautifulSoup(html, "lxml")

    recipe_box = soup.find("div", class_="recipe-box")
    if not recipe_box:
        return None

    name_tag = recipe_box.find("h1")
    name = name_tag.get_text(strip=True) if name_tag else "Unnamed Recipe"

    # Description is in a div with class 'entry-content' before the recipe box
    description_tag = soup.find("div", class_="entry-content")
    description = description_tag.get_text(strip=True) if description_tag else None

    ingredients_list = recipe_box.find("ul", class_="ingredients-list")
    ingredients = []
    if ingredients_list:
        # Ignore hidden list items
        for li in ingredients_list.find_all("li", style=lambda x: x != "display:none"):
            ingredient_text = li.get_text(separator=" ", strip=True)
            if ingredient_text:  # Only add non-empty ingredients
                ingredients.append(ingredient_text)

    directions_list = recipe_box.find("ol", itemprop="recipeInstructions")
    directions = []
    if directions_list:
        for li in directions_list.find_all("li"):
            directions.append(li.get_text(strip=True))

    garnish_tag = recipe_box.find("p", class_="garn-glass")
    garnish = garnish_tag.get_text(strip=True) if garnish_tag else None

    editors_note_tag = recipe_box.find("h5", string="Editor's Note")
    editors_note = None
    if editors_note_tag:
        note_p = editors_note_tag.find_next_sibling("p")
        if note_p:
            editors_note = note_p.get_text(strip=True)

    return Recipe(
        name=name,
        description=description,
        ingredients=ingredients,
        directions=directions,
        garnish=garnish,
        editors_note=editors_note,
    )
