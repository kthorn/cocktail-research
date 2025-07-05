"""Ingredient parsing and normalization utilities."""

import re
from typing import Optional, Tuple

from cocktail_utils.ingredients.number_utils import (
    _is_fraction,
    _is_integer,
    _is_number,
    _parse_fraction,
)

# --- Constants ---

# Unicode fraction mappings
UNICODE_FRAC = {"¼": ".25", "½": ".5", "¾": ".75", "⅓": ".333", "⅔": ".667"}

# Unit normalization mapping
UNIT_MAP = {
    # Volume
    "ounce": ["ounce", "ounces", "oz", "oz."],
    "tablespoon": [
        "tablespoon",
        "tablespoons",
        "tbsp",
        "tbsp.",
        "tablespoonful",
        "tablespoonfuls",
    ],
    "teaspoon": ["teaspoon", "teaspoons", "tsp", "tsp.", "teaspoonful", "teaspoonfuls"],
    "cup": ["cup", "cups"],
    "pint": ["pint", "pints", "pt", "pt."],
    "quart": ["quart", "quarts", "qt", "qt."],
    "gallon": ["gallon", "gallons", "gal", "gal."],
    "ml": ["milliliter", "milliliters", "ml", "ml."],
    "cl": ["centiliter", "centiliters", "cl", "cl."],
    "l": ["liter", "liters", "l", "l."],
    # Count/measure
    "dash": ["dash", "dashes"],
    "drop": ["drop", "drops"],
    "part": ["part", "parts"],
    "splash": ["splash", "splashes"],
    "pinch": ["pinch", "pinches"],
    "piece": ["piece", "pieces", "shoulder"],
    "slice": ["slice", "slices"],
    "wedge": ["wedge", "wedges"],
    "whole": ["whole"],
    "cube": ["cube", "cubes"],
}

# Create reverse mapping for lookup
UNIT_LOOKUP = {v: k for k, vs in UNIT_MAP.items() for v in vs}


# --- Functions ---


def normalize_unit(unit: str) -> str:
    """Normalize unit names to their standard form.

    Takes a raw unit string and converts it to the standardized unit name
    used throughout the system. Handles various abbreviations and alternate
    forms of common measurement units.

    Args:
        unit: Raw unit string that may contain abbreviations or variations.

    Returns:
        Normalized unit name in standard form.

    Examples:
        >>> normalize_unit("oz")
        'ounce'
        >>> normalize_unit("tbsp.")
        'tablespoon'
        >>> normalize_unit("ML")
        'ml'
    """
    unit = unit.lower().strip(".")
    return UNIT_LOOKUP.get(unit, unit)


def parse_quantity(text: str) -> Tuple[Optional[float], Optional[str], str]:
    """Parse ingredient quantities, handling special cases.

    Extracts amount, unit, and ingredient name from ingredient text.
    Handles special cases like 'heavy', 'scant', 'to top', mixed numbers,
    fractions, and unicode fraction characters.

    Args:
        text: Raw ingredient text to parse (e.g., "2 oz gin" or "Ginger beer, to top").

    Returns:
        A tuple containing:
            - amount: Numeric quantity as float, or None if no quantity found
            - unit: Unit of measurement as normalized string, or None if no unit found
            - ingredient_name: Clean ingredient name with quantity/unit removed

    Examples:
        >>> parse_quantity("2 oz gin")
        (2.0, 'ounce', 'gin')
        >>> parse_quantity("1/2 cup fresh lemon juice")
        (0.5, 'cup', 'fresh lemon juice')
        >>> parse_quantity("Ginger beer, to top")
        (None, 'to top', 'Ginger beer')
        >>> parse_quantity("3 Cherries")
        (3.0, None, 'Cherries')
        >>> parse_quantity("1 sugar cube")
        (1.0, 'cube', 'sugar')
    """
    original_text = text.strip()
    t = original_text.lower()

    # Handle special cases like "to top" or "as needed"
    if " to top" in t or " as needed" in t:
        unit = "to top" if " to top" in t else "as needed"
        # Extract the ingredient name before the special phrase
        ingredient_part = re.split(f",? {unit}", original_text, flags=re.IGNORECASE)[0]
        return None, unit, clean_ingredient_name(ingredient_part)

    # Pre-process the string
    t = re.sub(r"^\([^)]*\)\s*", "", t)  # remove parenthetical quantities
    t = "".join(UNICODE_FRAC.get(c, c) for c in t)  # convert unicode fractions

    # Remove special quantity words like "heavy" or "scant"
    t = re.sub(r"^(heavy|scant|about)\s+", "", t, flags=re.IGNORECASE)

    # Attempt to parse amount and unit from the start of the string
    amount, rest = _parse_amount(t)
    unit, rest = _parse_unit(rest)
    ingredient_name = clean_ingredient_name(rest)

    # If cleaning results in an empty string, fall back to the original text
    if not ingredient_name:
        ingredient_name = original_text

    return amount, unit, ingredient_name


def _parse_amount(text: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse amount from the start of an ingredient string.

    Helper function that extracts numeric quantities
    from the beginning of ingredient text. Handles mixed numbers, fractions,
    and decimal values.

    Args:
        text: Ingredient text with potential quantity and unit at the start.

    Returns:
        A tuple containing:
            - amount: Parsed numeric value as float, or None if no valid number found
            - rest: Remaining text after removing amount and unit

    """
    words = text.split()
    if not words:
        return None, ""
    amount = None
    name_start_index = 0

    try:
        # Case 1: Number range (e.g., "2 to 3" or "2-3")
        if (
            len(words) >= 3
            and _is_number(words[0])
            and words[1].lower() == "to"
            and _is_number(words[2])
        ):
            amount = (float(words[0]) + float(words[2])) / 2
            name_start_index = 3
        elif (
            len(words) >= 1 and "-" in words[0] and len(words[0].split("-")) == 2
        ):  # e.g. 2-3
            parts = words[0].split("-")
            if _is_number(parts[0]) and _is_number(parts[1]):
                amount = (float(parts[0]) + float(parts[1])) / 2
                name_start_index = 1
        # Case 2: Mixed number (e.g., "1 1/2")
        elif len(words) >= 2 and _is_integer(words[0]) and _is_fraction(words[1]):
            whole_part = int(words[0])
            fraction_part = _parse_fraction(words[1])
            amount = float(whole_part + fraction_part)
            name_start_index = 2

        # Case 3: Simple fraction (e.g., "1/2")
        elif len(words) >= 1 and _is_fraction(words[0]):
            amount = float(_parse_fraction(words[0]))
            name_start_index = 1

        # Case 4: Decimal or integer (e.g., "2.5" or "3")
        elif len(words) >= 1 and _is_number(words[0]):
            amount = float(words[0])
            name_start_index = 1

    except (ValueError, ZeroDivisionError):
        # If parsing fails, no amount can be parsed
        amount = None
        name_start_index = 0

    # The rest of the string is the ingredient name
    rest = " ".join(words[name_start_index:])
    return amount, rest


def _parse_unit(text: str) -> tuple[Optional[str], str]:
    """Parse unit from the start of an ingredient string.

    Helper function that extracts the unit from the start of an ingredient string.
    """
    words = text.split()
    if not words:
        return None, text

    name_start_index = 0
    if name_start_index < len(words):
        potential_unit = words[name_start_index].lower().strip(".")
        if potential_unit in UNIT_LOOKUP:
            unit = normalize_unit(potential_unit)
            name_start_index += 1
            return unit, " ".join(words[name_start_index:])

    # No unit found - return None for unit and the original text
    return None, text


def clean_ingredient_name(name: str) -> str:
    """Clean up ingredient names by removing formatting and notes.

    Removes parenthetical notes, extra whitespace, and other formatting
    artifacts to produce clean ingredient names suitable for matching
    and categorization.

    Args:
        name: Raw ingredient name that may contain parenthetical notes
              and extra formatting.

    Returns:
        Cleaned ingredient name with formatting removed.

    Examples:
        >>> clean_ingredient_name("fresh lemon juice (about 1 lemon)")
        'fresh lemon juice'
        >>> clean_ingredient_name("vodka  , premium")
        'vodka, premium'
        >>> clean_ingredient_name("(optional) fresh mint")
        'fresh mint'
    """
    # Remove parenthetical quantities at the start (including "about", "heavy", etc)
    name = re.sub(r"^\([^)]*\)\s*", "", name)

    # Remove parenthetical notes in the middle/end
    name = re.sub(r"\s*\([^)]*\)", "", name)

    # Remove extra whitespace and commas
    name = re.sub(r"\s+", " ", name)
    name = name.strip().strip(",")

    return name


def normalize_ingredient_text(ingredient_text: str) -> str:
    """Normalize ingredient text for consistent matching.

    Removes quantities, measurements, and common prefixes that don't
    affect ingredient categorization. Converts to lowercase and
    normalizes whitespace for consistent taxonomy matching.

    Args:
        ingredient_text: The raw ingredient description text that may
                        contain quantities, measurements, and qualifiers.

    Returns:
        Normalized ingredient text suitable for taxonomy matching,
        with quantities and non-essential qualifiers removed.

    Examples:
        >>> normalize_ingredient_text("2 oz fresh lemon juice")
        'lemon juice'
        >>> normalize_ingredient_text("1 tsp premium vanilla extract")
        'vanilla extract'
        >>> normalize_ingredient_text("3 dashes good quality bitters")
        'bitters'
    """
    # Normalize Unicode apostrophes and quotes to ASCII
    text = (
        ingredient_text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    # Remove quantities and measurements
    text = re.sub(
        r"^\d+[\s\d/]*\s*(ounces?|oz|cups?|tsp|tbsp|ml|cl|dashes?|drops?)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove common prefixes that don't affect categorization
    text = re.sub(
        r"^(fresh|freshly|good|quality|premium|good\s+quality)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace and convert to lowercase
    text = " ".join(text.lower().split())

    return text
