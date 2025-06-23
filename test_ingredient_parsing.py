#!/usr/bin/env python3
"""
Unit tests for ingredient parsing functions from punch-dl.py
"""

import unittest
from decimal import Decimal
import sys
import os

# Add the scripts directory to the path so we can import punch_dl directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from punch_dl import parse_qty, clean_ingredient_name, normalize_unit


class TestIngredientParsing(unittest.TestCase):
    def test_parse_qty_simple_cases(self):
        """Test basic quantity parsing"""
        # Simple whole number
        amt, unit, name = parse_qty("2 ounces vodka")
        self.assertEqual(amt, 2.0)
        self.assertEqual(unit, "ounce")
        self.assertEqual(name, "vodka")

        # Simple fraction
        amt, unit, name = parse_qty("1/2 ounce lime juice")
        self.assertEqual(amt, 0.5)
        self.assertEqual(unit, "ounce")
        self.assertEqual(name, "lime juice")

    def test_parse_qty_mixed_numbers(self):
        """Test mixed number parsing - this is the failing case mentioned"""
        # This should parse "2 3/4" as 2.75, not just "2"
        amt, unit, name = parse_qty("2 3/4 ounces vodka, such as Truman")
        self.assertEqual(amt, 2.75)
        self.assertEqual(unit, "ounce")
        self.assertEqual(name, "vodka, such as Truman")

        # Another mixed number case
        amt, unit, name = parse_qty("1 1/2 cups sugar")
        self.assertEqual(amt, 1.5)
        self.assertEqual(unit, "cup")
        self.assertEqual(name, "sugar")

    def test_parse_qty_unicode_fractions(self):
        """Test unicode fraction parsing"""
        amt, unit, name = parse_qty("¼ ounce simple syrup")
        self.assertEqual(amt, 0.25)
        self.assertEqual(unit, "ounce")
        self.assertEqual(name, "simple syrup")

        amt, unit, name = parse_qty("½ cup water")
        self.assertEqual(amt, 0.5)
        self.assertEqual(unit, "cup")
        self.assertEqual(name, "water")

        amt, unit, name = parse_qty("¾ ounce lemon juice")
        self.assertEqual(amt, 0.75)
        self.assertEqual(unit, "ounce")
        self.assertEqual(name, "lemon juice")

    def test_parse_qty_ranges(self):
        """Test range parsing"""
        amt, unit, name = parse_qty("1 to 2 ounces gin")
        self.assertEqual(amt, 1.5)  # Average of 1 and 2
        self.assertEqual(unit, "ounce")
        self.assertEqual(name, "gin")

        amt, unit, name = parse_qty("2 to 4 dashes bitters")
        self.assertEqual(amt, 3.0)  # Average of 2 and 4
        self.assertEqual(unit, "dash")
        self.assertEqual(name, "bitters")

    def test_parse_qty_special_quantities(self):
        """Test special quantity words"""
        amt, unit, name = parse_qty("heavy 2 ounces rum")
        self.assertEqual(amt, 2.0)
        self.assertEqual(unit, "ounce")
        self.assertEqual(name, "rum")

        amt, unit, name = parse_qty("scant 1 cup flour")
        self.assertEqual(amt, 1.0)
        self.assertEqual(unit, "cup")
        self.assertEqual(name, "flour")

        amt, unit, name = parse_qty("about 3 tablespoons honey")
        self.assertEqual(amt, 3.0)
        self.assertEqual(unit, "tablespoon")
        self.assertEqual(name, "honey")

    def test_parse_qty_to_top(self):
        """Test 'to top' ingredients"""
        amt, unit, name = parse_qty("ginger beer to top")
        self.assertEqual(amt, 0)
        self.assertEqual(unit, "to top")
        self.assertEqual(name, "ginger beer")

        amt, unit, name = parse_qty("club soda as needed")
        self.assertEqual(amt, 0)
        self.assertEqual(unit, "to top")
        self.assertEqual(name, "club soda")

        # Complex case: range with "to top" - the quantity should be extracted
        amt, unit, name = parse_qty("4 to 6 ounces Dr Pepper, to top")
        # The "to top" instruction should take precedence, but the ingredient name should be clean
        self.assertEqual(amt, 0)  # "to top" means variable amount
        self.assertEqual(unit, "to top")
        self.assertEqual(name, "Dr Pepper")  # Should not include "4 to 6 ounces"

    def test_parse_qty_unit_first(self):
        """Test cases where unit comes first"""
        amt, unit, name = parse_qty("ounces vodka")
        self.assertEqual(amt, 1.0)  # Should default to 1
        self.assertEqual(unit, "ounce")
        self.assertEqual(name, "vodka")

    def test_parse_qty_no_quantity(self):
        """Test ingredients with no explicit quantity"""
        amt, unit, name = parse_qty("salt")
        self.assertIsNone(amt)
        self.assertIsNone(unit)
        self.assertEqual(name, "salt")

        amt, unit, name = parse_qty("garnish lemon twist")
        self.assertIsNone(amt)
        self.assertIsNone(unit)
        self.assertEqual(name, "garnish lemon twist")

    def test_clean_ingredient_name(self):
        """Test ingredient name cleaning"""
        # Remove parenthetical notes
        name = clean_ingredient_name("vodka (such as Tito's)")
        self.assertEqual(name, "vodka")

        # Remove parenthetical quantities at start
        name = clean_ingredient_name("(about 2 oz) gin")
        self.assertEqual(name, "gin")

        # Clean up whitespace and commas
        name = clean_ingredient_name("  rum,  dark  ")
        self.assertEqual(name, "rum, dark")

        # Complex case with multiple parentheticals
        name = clean_ingredient_name("whiskey (bourbon preferred) (100 proof)")
        self.assertEqual(name, "whiskey")

    def test_normalize_unit(self):
        """Test unit normalization"""
        # Plural to singular
        self.assertEqual(normalize_unit("ounces"), "ounce")
        self.assertEqual(normalize_unit("tablespoons"), "tablespoon")

        # Abbreviations to full form
        self.assertEqual(normalize_unit("oz"), "ounce")
        self.assertEqual(normalize_unit("tbsp"), "tablespoon")
        self.assertEqual(normalize_unit("tsp"), "teaspoon")

        # Case insensitive
        self.assertEqual(normalize_unit("OUNCE"), "ounce")
        self.assertEqual(normalize_unit("Oz."), "ounce")

        # Unknown units should pass through
        self.assertEqual(normalize_unit("jiggly"), "jiggly")

    def test_parse_qty_edge_cases(self):
        """Test edge cases and potential failures"""
        # Decimal numbers
        amt, unit, name = parse_qty("2.5 ounces gin")
        self.assertEqual(amt, 2.5)
        self.assertEqual(unit, "ounce")
        self.assertEqual(name, "gin")

        # Complex fractions
        amt, unit, name = parse_qty("3/8 ounce syrup")
        self.assertEqual(amt, 0.375)
        self.assertEqual(unit, "ounce")
        self.assertEqual(name, "syrup")

    def test_parse_qty_real_world_examples(self):
        """Test with real-world examples from Punch recipes"""
        # Examples that might be found on the site
        test_cases = [
            ("2 ounces bourbon whiskey", 2.0, "ounce", "bourbon whiskey"),
            ("¾ ounce fresh lemon juice", 0.75, "ounce", "fresh lemon juice"),
            ("½ ounce simple syrup", 0.5, "ounce", "simple syrup"),
            ("2 dashes Angostura bitters", 2.0, "dash", "Angostura bitters"),
            (
                "1 lemon twist, for garnish",
                1.0,
                "lemon",
                "twist, for garnish",
            ),  # "lemon" will be parsed as unit
            ("Club soda, to top", 0, "to top", "Club soda"),
        ]

        for ingredient_text, expected_amt, expected_unit, expected_name in test_cases:
            with self.subTest(ingredient=ingredient_text):
                try:
                    amt, unit, name = parse_qty(ingredient_text)
                    if expected_amt == 0:  # Special case for "to top"
                        self.assertEqual(amt, expected_amt)
                        self.assertEqual(unit, expected_unit)
                    else:
                        self.assertEqual(
                            amt,
                            expected_amt,
                            f"Amount mismatch for '{ingredient_text}'",
                        )
                        self.assertEqual(
                            unit,
                            expected_unit,
                            f"Unit mismatch for '{ingredient_text}'",
                        )
                    # Name comparison might be more flexible
                    self.assertIn(
                        expected_name.lower(),
                        name.lower(),
                        f"Name mismatch for '{ingredient_text}'",
                    )
                except Exception as e:
                    self.fail(f"Failed to parse '{ingredient_text}': {e}")


if __name__ == "__main__":
    unittest.main()
