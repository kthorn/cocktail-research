"""Unit conversion utilities for cocktail recipe databases."""

import math
import pathlib
from typing import Optional, Union
import numpy as np


# Standard volumetric conversions to ml
UNIT_CONVERSIONS = {
    # Already in ml
    "ml": 1.0,
    "milliliter": 1.0,
    "millilitre": 1.0,
    # Liters
    "l": 1000.0,
    "liter": 1000.0,
    "litre": 1000.0,
    # US fluid ounces
    "ounce": 29.5735,
    "oz": 29.5735,
    "fl oz": 29.5735,
    "fluid ounce": 29.5735,
    # US cups
    "cup": 236.588,
    "cups": 236.588,
    # US tablespoons
    "tablespoon": 14.7868,
    "tbsp": 14.7868,
    "T": 14.7868,
    # US teaspoons
    "teaspoon": 4.92892,
    "tsp": 4.92892,
    "t": 4.92892,
    # US quarts
    "quart": 946.353,
    "qt": 946.353,
    # US gallons
    "gallon": 3785.41,
    "gal": 3785.41,
    # Bar measurements (approximate)
    "dash": 0.616,  # ~1/8 teaspoon
    "splash": 5.0,  # ~1 teaspoon
    "drop": 0.05,  # ~1/100 teaspoon
    "pinch": 0.31,  # ~1/16 teaspoon
    "part": 30.0,  # Standard bar jigger shot
}

# Non-volumetric units that should be set to 0
NON_VOLUMETRIC_UNITS = {
    "cube",
    "piece",
    "slice",
    "wedge",
    "whole",
    "each",
    "clove",
    "sprig",
    "leaf",
    "leaves",
    "twist",
    "peel",
    "garnish",
    "rim",
    "float",
}

# Special handling units
SPECIAL_UNITS = {
    "as needed": 0.0,
    "to top": 90.0,  # Variable amount, set to 90 ml/3 oz for analysis
    "to taste": 0.0,
}


def convert_to_ml(amount: Optional[Union[float, int]], unit: Optional[str]) -> float:
    """Convert a given amount and unit to milliliters.

    Args:
        amount: The numeric amount (can be None for missing values)
        unit: The unit string (can be None for missing values)

    Returns:
        The amount converted to milliliters, or NaN if the ingredient/amount is missing

    Examples:
        >>> convert_to_ml(1, "ounce")
        29.5735
        >>> convert_to_ml(1, "slice")
        0.0
        >>> convert_to_ml(None, "ml")
        nan
    """
    # Handle missing values
    if amount is None or unit is None:
        return np.nan

    # Convert amount to float if it's not already
    try:
        amount_float = float(amount)
    except (ValueError, TypeError):
        return np.nan

    # Handle zero amounts
    if amount_float == 0:
        return 0.0

    # Normalize unit string
    unit_normalized = unit.lower().strip()

    # Check for special units first
    if unit_normalized in SPECIAL_UNITS:
        return SPECIAL_UNITS[unit_normalized]

    # Check for non-volumetric units
    if unit_normalized in NON_VOLUMETRIC_UNITS:
        return 0.0

    # Check for volumetric conversions
    if unit_normalized in UNIT_CONVERSIONS:
        return amount_float * UNIT_CONVERSIONS[unit_normalized]

    # If unit is not recognized, log it and return 0
    print(f"Warning: Unknown unit '{unit}', treating as non-volumetric (0 ml)")
    return 0.0


def get_all_units_from_db(db_path: Union[str, pathlib.Path]) -> set:
    """Get all unique units from the database for validation.

    Args:
        db_path: Path to the SQLite database

    Returns:
        Set of all unique units found in the database
    """
    import sqlite3
    from .utils import get_connection

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT unit FROM recipe_ingredient WHERE unit IS NOT NULL"
        )
        units = {row[0] for row in cursor.fetchall()}
        return units
    finally:
        conn.close()


def validate_unit_coverage(db_path: Union[str, pathlib.Path]) -> None:
    """Validate that all units in the database have conversion coverage.

    Args:
        db_path: Path to the SQLite database

    Prints warnings for any units that don't have explicit conversion rules.
    """
    db_units = get_all_units_from_db(db_path)

    all_known_units = (
        set(UNIT_CONVERSIONS.keys()) | NON_VOLUMETRIC_UNITS | set(SPECIAL_UNITS.keys())
    )

    unknown_units = set()
    for unit in db_units:
        if unit.lower().strip() not in all_known_units:
            unknown_units.add(unit)

    if unknown_units:
        print(f"Warning: Found {len(unknown_units)} unknown units in database:")
        for unit in sorted(unknown_units):
            print(f"  - '{unit}'")
        print("These will be treated as non-volumetric (0 ml)")
    else:
        print(f"✓ All {len(db_units)} units in database have conversion coverage")
