"""Ingredient quantity parsing utilities."""

import re
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple

from .normalization import normalize_unit

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
}

# Create reverse mapping for lookup
UNIT_LOOKUP = {v: k for k, vs in UNIT_MAP.items() for v in vs}


def parse_quantity(text: str) -> Tuple[Optional[float], Optional[str], str]:
    """Parse ingredient quantities, handling special cases like 'heavy', 'scant', and 'to top'.
    
    Args:
        text: Raw ingredient text to parse
        
    Returns:
        Tuple of (amount, unit, ingredient_name)
        
    Examples:
        >>> parse_quantity("2 oz gin")
        (2.0, "ounce", "gin")
        >>> parse_quantity("1/2 cup fresh lemon juice")
        (0.5, "cup", "fresh lemon juice")
        >>> parse_quantity("Ginger beer, to top")
        (0, "to top", "Ginger beer")
    """
    original_text = text.strip()
    t = original_text.lower()

    # Handle "to top" ingredients (like ginger beer, tonic water, etc)
    if " to top" in t or " as needed" in t:
        # First extract the ingredient part before ", to top"
        parts = original_text.split(", to top")
        if len(parts) == 1:  # Try " to top" without comma
            parts = original_text.split(" to top")
        if len(parts) == 1:  # Try "as needed"
            parts = original_text.split(" as needed")

        ingredient_part = parts[0].strip()
        ingredient_part_lower = ingredient_part.lower()

        # Try to extract quantity information from the ingredient part
        # Check for ranges like "4 to 6 ounces Dr Pepper"
        range_match = re.match(
            r"^(\d+(?:\s+\d+/\d+)?)\s+to\s+(\d+(?:\s+\d+/\d+)?)\s+(\w+)\s+(.+)",
            ingredient_part_lower,
        )
        if range_match:
            # Extract the ingredient name (group 4)
            ingredient_name = re.match(
                r"^(\d+(?:\s+\d+/\d+)?)\s+to\s+(\d+(?:\s+\d+/\d+)?)\s+(\w+)\s+(.+)",
                ingredient_part,
                re.IGNORECASE,
            ).group(4)
            return 0, "to top", clean_ingredient_name(ingredient_name)

        # Check for simple quantities like "2 ounces Dr Pepper"
        simple_qty_match = re.match(
            r"^(\d+(?:\.\d+)?(?:\s+\d+/\d+)?)\s+(\w+)\s+(.+)", ingredient_part_lower
        )
        if simple_qty_match:
            # Extract the ingredient name (group 3)
            ingredient_name = re.match(
                r"^(\d+(?:\.\d+)?(?:\s+\d+/\d+)?)\s+(\w+)\s+(.+)",
                ingredient_part,
                re.IGNORECASE,
            ).group(3)
            return 0, "to top", clean_ingredient_name(ingredient_name)

        # Fallback to original behavior - clean the whole ingredient part
        return 0, "to top", clean_ingredient_name(ingredient_part)

    # Remove parenthetical quantities at the start
    t = re.sub(r"^\([^)]*\)\s*", "", t)
    original_text = re.sub(r"^\([^)]*\)\s*", "", original_text)

    # Remove special quantity words
    for word in ["heavy", "scant", "about"]:
        if t.startswith(word + " "):
            t = t[len(word) :].strip()
            # Also remove from original text, preserving case
            word_pattern = re.compile(r"^" + re.escape(word) + r"\s+", re.IGNORECASE)
            original_text = word_pattern.sub("", original_text).strip()

    # Handle ranges (e.g. "1 to 2 ounces")
    range_match = re.match(r"^(\d+(?:\s+\d+/\d+)?)\s+to\s+(\d+(?:\s+\d+/\d+)?)\s+", t)
    if range_match:
        # Convert both numbers to decimals and take average
        start = Decimal(str(eval(range_match.group(1))))
        end = Decimal(str(eval(range_match.group(2))))
        amt = float((start + end) / 2)
        # Remove the range part from both versions
        t = t[range_match.end() :]
        # Find the same pattern in original text
        original_range_match = re.match(
            r"^(\d+(?:\s+\d+/\d+)?)\s+to\s+(\d+(?:\s+\d+/\d+)?)\s+",
            original_text,
            re.IGNORECASE,
        )
        if original_range_match:
            original_text = original_text[original_range_match.end() :]
        parts = t.split()
        original_parts = original_text.split()
        unit = normalize_unit(parts[0])
        rest = " ".join(original_parts[1:]) if len(original_parts) > 1 else ""
        return amt, unit, clean_ingredient_name(rest)

    # Convert unicode fractions
    for k, v in UNICODE_FRAC.items():
        t = t.replace(k, v)

    parts = t.split()
    original_parts = original_text.split()
    try:
        # Check if first part is a unit (like "ounces vodka")
        if parts[0] in UNIT_LOOKUP:
            amt = Decimal(1)
            unit = normalize_unit(parts[0])
            rest = " ".join(original_parts[1:])
        else:
            # Try to parse as a mixed number (e.g., "2 3/4" or just "2" or "3/4")
            amt = None
            unit_idx = 1  # Default assumption: first part is number, second is unit

            # Check for mixed numbers like "2 3/4"
            if (
                len(parts) >= 2
                and re.match(r"^\d+$", parts[0])
                and re.match(r"^\d+/\d+$", parts[1])
            ):
                # Mixed number: "2 3/4"
                whole = int(parts[0])
                frac_parts = parts[1].split("/")
                fraction = Decimal(frac_parts[0]) / Decimal(frac_parts[1])
                amt = Decimal(whole) + fraction
                unit_idx = 2
            else:
                # Try to parse the first part as a number (could be "2", "3/4", "2.5", etc.)
                try:
                    amt = Decimal(str(eval(parts[0])))  # "3/4" → 0.75, "2" → 2
                except (ValueError, ZeroDivisionError, SyntaxError):
                    # If that fails, maybe it's a decimal
                    amt = Decimal(parts[0])

            if unit_idx < len(parts):
                unit = normalize_unit(parts[unit_idx])
                rest = (
                    " ".join(original_parts[unit_idx + 1 :])
                    if unit_idx + 1 < len(original_parts)
                    else ""
                )
            else:
                # No unit found
                unit = None
                rest = " ".join(original_parts[1:]) if len(original_parts) > 1 else ""
    except (InvalidOperation, IndexError, SyntaxError, NameError):
        # If we can't parse a quantity, treat the whole line as ingredient name
        amt = None
        unit = None
        rest = " ".join(original_parts)

    # Clean the ingredient name
    rest = clean_ingredient_name(rest)

    # If we ended up with an empty ingredient name, use the original text
    if not rest:
        rest = original_text

    return float(amt) if amt is not None else None, unit, rest


def clean_ingredient_name(name: str) -> str:
    """Clean up ingredient names by removing parenthetical notes and other formatting.
    
    Args:
        name: Raw ingredient name
        
    Returns:
        Cleaned ingredient name
        
    Examples:
        >>> clean_ingredient_name("fresh lemon juice (about 1 lemon)")
        "fresh lemon juice"
        >>> clean_ingredient_name("vodka  , premium")
        "vodka, premium"
    """
    # Remove parenthetical quantities at the start (including "about", "heavy", etc)
    name = re.sub(r"^\([^)]*\)\s*", "", name)

    # Remove parenthetical notes in the middle/end
    name = re.sub(r"\s*\([^)]*\)", "", name)

    # Remove extra whitespace and commas
    name = re.sub(r"\s+", " ", name)
    name = name.strip().strip(",")

    return name