"""Ingredient normalization utilities."""

import re
from typing import List, Optional, Tuple

# Unit normalization mapping (imported from parsing.py to avoid duplication)
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


def normalize_unit(unit: str) -> str:
    """Normalize unit names to their standard form.
    
    Args:
        unit: Raw unit string
        
    Returns:
        Normalized unit name
        
    Examples:
        >>> normalize_unit("oz")
        "ounce"
        >>> normalize_unit("tbsp.")
        "tablespoon"
    """
    unit = unit.lower().strip(".")
    return UNIT_LOOKUP.get(unit, unit)  # Return original if not found


def normalize_ingredient_text(ingredient_text: str) -> str:
    """Normalize ingredient text for consistent matching.

    Removes quantities, measurements, and common prefixes that don't
    affect ingredient categorization. Converts to lowercase and
    normalizes whitespace.

    Args:
        ingredient_text: The raw ingredient description text.

    Returns:
        Normalized ingredient text suitable for taxonomy matching.

    Examples:
        >>> normalize_ingredient_text("2 oz fresh lemon juice")
        "lemon juice"
        >>> normalize_ingredient_text("1 tsp premium vanilla extract")
        "vanilla extract"
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


def extract_brand(ingredient_text: str, brand_patterns: Optional[List[re.Pattern]] = None) -> Tuple[Optional[str], str]:
    """Extract brand name and return cleaned ingredient text.

    Searches for brand references in ingredient text using compiled patterns
    and removes them to get the clean ingredient name.

    Args:
        ingredient_text: The raw ingredient description text.
        brand_patterns: Optional list of compiled regex patterns for brand extraction.
                       If None, uses default patterns.

    Returns:
        A tuple containing:
            - Optional brand name if found, None otherwise
            - Cleaned ingredient text with brand reference removed
            
    Examples:
        >>> extract_brand("gin, preferably Hendrick's")
        ("Hendrick's", "gin")
        >>> extract_brand("dry vermouth such as Dolin")
        ("Dolin", "dry vermouth")
    """
    if brand_patterns is None:
        brand_patterns = _get_default_brand_patterns()
    
    for pattern in brand_patterns:
        match = pattern.search(ingredient_text)
        if match:
            brand = match.group(1).strip()
            # Remove the brand reference from the text
            cleaned_text = pattern.sub("", ingredient_text).strip().rstrip(",")
            return brand, cleaned_text
    return None, ingredient_text


def _get_default_brand_patterns() -> List[re.Pattern]:
    """Get default regex patterns for brand extraction from ingredient text.

    Creates regex patterns to identify and extract brand names from
    ingredient descriptions, looking for patterns like "preferably X",
    "such as X", "like X", and brands mentioned at the end.

    Returns:
        A list of compiled regex patterns for brand extraction.
    """
    return [
        re.compile(r"preferably\s+([A-Z][a-zA-Z\s&\'\\-\\d]+)", re.IGNORECASE),
        re.compile(r"such as\s+([A-Z][a-zA-Z\s&\'\\-\\d]+)", re.IGNORECASE),
        re.compile(r"like\s+([A-Z][a-zA-Z\s&\'\\-\\d]+)", re.IGNORECASE),
        # Pattern for brands at the end without "preferably"
        re.compile(r",\s+([A-Z][a-zA-Z\s&\'\\-\\d]+)$", re.IGNORECASE),
    ]