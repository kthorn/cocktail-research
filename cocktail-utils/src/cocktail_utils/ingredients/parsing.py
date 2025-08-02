"""Ingredient parsing and normalization utilities."""

import re
from typing import Optional, Tuple

from cocktail_utils.database.units import UNIT_LOOKUP, normalize_unit
from cocktail_utils.ingredients.number_utils import (
    _is_fraction,
    _is_integer,
    _is_number,
    _parse_fraction,
)

# --- Constants ---

# Unicode fraction mappings
UNICODE_FRAC = {"¼": ".25", "½": ".5", "¾": ".75", "⅓": ".333", "⅔": ".667"}

# --- Functions ---


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
    """

    original_text = text.strip()
    t = original_text.lower()
    t = re.sub(r"^\([^)]*\)\s*", "", t)  # remove parenthetical quantities
    # Remove special quantity words like scant at the beginning of the string
    t = re.sub(
        r"^(generous|small|heaping|short|shy|heavy|scant|about)\b\s+",
        "",
        t,
        flags=re.IGNORECASE,
    )

    # Handle special cases like "to top" or "as needed"
    if "to top" in t or "as needed" in t:
        unit = "to top" if "to top" in t else "as needed"
        # Extract the ingredient name before the special phrase
        ingredient_part = re.split(f",? {unit}", original_text, flags=re.IGNORECASE)[0]
        return None, unit, clean_ingredient_name(ingredient_part)

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
    # Convert unicode fractions to decimal equivalents
    text = "".join(UNICODE_FRAC.get(c, c) for c in text)

    words = text.split()
    if not words:
        return None, ""

    try:
        # Try different parsing patterns in order of complexity
        for parser in [_parse_number_range, _parse_mixed_number, _parse_simple_number]:
            amount, consumed_words = parser(words)
            if amount is not None:
                rest = " ".join(words[consumed_words:])
                return amount, rest
    except (ValueError, ZeroDivisionError):
        # If any parsing fails, no amount can be parsed
        pass

    # No valid amount found
    return None, " ".join(words)


def _parse_number_range(words: list[str]) -> Tuple[Optional[float], int]:
    """Parse number ranges like '2 to 3' or '2-3'."""
    # Pattern: "2 to 3"
    if (
        len(words) >= 3
        and _is_number(words[0])
        and words[1].lower() == "to"
        and _is_number(words[2])
    ):
        amount = (float(words[0]) + float(words[2])) / 2
        return amount, 3

    # Pattern: "2-3"
    if len(words) >= 1 and "-" in words[0] and len(words[0].split("-")) == 2:
        parts = words[0].split("-")
        if _is_number(parts[0]) and _is_number(parts[1]):
            amount = (float(parts[0]) + float(parts[1])) / 2
            return amount, 1

    return None, 0


def _parse_mixed_number(words: list[str]) -> Tuple[Optional[float], int]:
    """Parse mixed numbers like '1 1/2' or '1 .5'."""
    if len(words) < 2 or not _is_integer(words[0]):
        return None, 0

    whole_part = int(words[0])

    # Pattern: "1 1/2" (whole number + fraction)
    if _is_fraction(words[1]):
        fraction_part = _parse_fraction(words[1])
        amount = float(whole_part + fraction_part)
        return amount, 2

    # Pattern: "1 .5" (whole number + decimal, after unicode conversion)
    if _is_number(words[1]):
        decimal_part = float(words[1])
        # Only treat as mixed number if the decimal part is less than 1
        if 0 < decimal_part < 1:
            amount = float(whole_part + decimal_part)
            return amount, 2

    return None, 0


def _parse_simple_number(words: list[str]) -> Tuple[Optional[float], int]:
    """Parse simple numbers like '1/2', '2.5', or '3'."""
    if len(words) < 1:
        return None, 0

    # Pattern: "1/2" (simple fraction)
    if _is_fraction(words[0]):
        amount = float(_parse_fraction(words[0]))
        return amount, 1

    # Pattern: "2.5" or "3" (decimal or integer)
    if _is_number(words[0]):
        amount = float(words[0])
        return amount, 1

    return None, 0


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
