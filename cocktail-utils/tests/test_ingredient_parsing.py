import pytest

from cocktail_utils.ingredients.parsing import (
    _parse_amount,
    _parse_unit,
    clean_ingredient_name,
    normalize_ingredient_text,
    normalize_unit,
    parse_quantity,
)


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("2 oz fresh lemon juice", "lemon juice"),
        ("1 tsp premium vanilla extract", "vanilla extract"),
        ("3 dashes Angostura bitters", "angostura bitters"),
        ("Freshly grated nutmeg", "grated nutmeg"),
        ("  lots   of   whitespace  ", "lots of whitespace"),
    ],
)
def test_normalize_ingredient_text(input_text, expected_text):
    """Test that normalize_ingredient_text removes quantities, units, and prefixes."""
    assert normalize_ingredient_text(input_text) == expected_text


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("vodka (such as Tito's)", "vodka"),
        ("(about 2 oz) gin", "gin"),
        ("  rum,  dark  ", "rum, dark"),
        ("whiskey (bourbon preferred) (100 proof)", "whiskey"),
    ],
)
def test_clean_ingredient_name(input_text, expected_text):
    """Test ingredient name cleaning."""
    assert clean_ingredient_name(input_text) == expected_text


@pytest.mark.parametrize(
    "input_unit, expected_unit",
    [
        ("ounces", "ounce"),
        ("oz.", "ounce"),
        ("TBSP.", "tablespoon"),
    ],
)
def test_normalize_unit(input_unit, expected_unit):
    """Test unit normalization."""
    assert normalize_unit(input_unit) == expected_unit


@pytest.mark.parametrize(
    "input_text, expected_amt, expected_rest",
    [
        ("2 ounces vodka", 2.0, "ounces vodka"),
        ("1/2 cup sugar", 0.5, "cup sugar"),
        ("1 1/2 tsp salt", 1.5, "tsp salt"),
        ("salt", None, "salt"),
        ("2.5 ml water", 2.5, "ml water"),
        ("", None, ""),
        ("2 to 3 dashes Herbstura", 2.5, "dashes Herbstura"),
        ("2-3 dashes Herbstura", 2.5, "dashes Herbstura"),
        (
            "4 lemons, peeled and peels reserved",
            4.0,
            "lemons, peeled and peels reserved",
        ),
        ("1 ½ ounces  Jamaican rum", 1.5, "ounces Jamaican rum"),
        ("½ ounce  Jamaican rum", 0.5, "ounce Jamaican rum"),
    ],
)
def test_parse_amount(input_text, expected_amt, expected_rest):
    """Test the private helper _parse_amount."""
    # Note: This test assumes a bug in _parse_amount is fixed where it returns
    # three values instead of two.
    result = _parse_amount(input_text)
    assert len(result) == 2, "_parse_amount should return a tuple of (amount, rest)"
    amt, rest = result
    assert amt == expected_amt
    assert rest == expected_rest


@pytest.mark.parametrize(
    "input_text, expected_unit, expected_rest",
    [
        ("ounces vodka", "ounce", "vodka"),
        ("cup sugar", "cup", "sugar"),
        ("cubes of ice", "cube", "of ice"),
        ("salt", None, "salt"),
        ("barspoon simple syrup", "barspoon", "simple syrup"),
        ("", None, ""),
    ],
)
def test_parse_unit(input_text, expected_unit, expected_rest):
    """Test the private helper _parse_unit."""
    unit, rest = _parse_unit(input_text)
    assert unit == expected_unit
    assert rest == expected_rest


@pytest.mark.parametrize(
    "input_text, expected_amt, expected_unit, expected_ingredient",
    [
        # Basic cases with amount, unit, and ingredient
        ("2 oz gin", 2.0, "ounce", "gin"),
        ("1.5 cups sugar", 1.5, "cup", "sugar"),
        ("1/2 tsp salt", 0.5, "teaspoon", "salt"),
        ("1 1/2 oz lime juice", 1.5, "ounce", "lime juice"),
        # Special cases like "to top" and "as needed"
        ("Ginger beer, to top", None, "to top", "Ginger beer"),
        ("Club soda to top", None, "to top", "Club soda"),
        ("Salt, as needed", None, "as needed", "Salt"),
        ("Fresh herbs as needed", None, "as needed", "Fresh herbs"),
        ("soda water (to top)", None, "to top", "soda water"),
        # Cases with parenthetical notes
        ("2 oz fresh lemon juice (about 1 lemon)", 2.0, "ounce", "fresh lemon juice"),
        # Cases with special quantity words
        ("Scant 1 tsp vanilla", 1.0, "teaspoon", "vanilla"),
        ("About 3 dashes bitters", 3.0, "dash", "bitters"),
        # Mixed numbers and fractions
        ("3/4 oz simple syrup", 0.75, "ounce", "simple syrup"),
        # Number ranges
        ("2 to 3 dashes Angostura bitters", 2.5, "dash", "angostura bitters"),
        ("2-3 oz whiskey", 2.5, "ounce", "whiskey"),
        # Unicode fractions
        ("1½ oz bourbon", 1.5, "ounce", "bourbon"),
        ("1 ½ oz bourbon", 1.5, "ounce", "bourbon"),
        ("¼ cup water", 0.25, "cup", "water"),
        ("⅔ cup milk", 0.667, "cup", "milk"),
        # Cases with no amount or unit
        ("Fresh mint leaves", None, None, "fresh mint leaves"),
        ("Ice", None, None, "ice"),
        # Cases with only amount, no unit
        ("2 lemons", 2.0, None, "lemons"),
        ("3 limes, juiced", 3.0, None, "limes, juiced"),
        # Complex ingredient names
        ("2 oz premium aged rum", 2.0, "ounce", "premium aged rum"),
        ("1 barspoon rich simple syrup", 1.0, "barspoon", "rich simple syrup"),
        ("4 dashes Peychaud's bitters", 4.0, "dash", "peychaud's bitters"),
        # Edge cases
        ("", None, None, ""),
        ("   ", None, None, ""),
        ("0.5 oz vodka", 0.5, "ounce", "vodka"),
        # Cases with commas and complex formatting
        ("2 oz gin, premium", 2.0, "ounce", "gin, premium"),
        ("1 cup ice, crushed", 1.0, "cup", "ice, crushed"),
    ],
)
def test_parse_quantity(input_text, expected_amt, expected_unit, expected_ingredient):
    """Test the main parse_quantity function with various input formats."""
    amount, unit, ingredient = parse_quantity(input_text)

    if expected_amt is None:
        assert amount is None
    else:
        assert amount == pytest.approx(expected_amt, abs=1e-3)

    assert unit == expected_unit
    assert ingredient == expected_ingredient


def test_parse_quantity_fallback_to_original():
    """Test that parse_quantity falls back to original text when cleaning results in empty string."""
    # This is a case where cleaning might result in an empty ingredient name
    result = parse_quantity("()")
    amount, unit, ingredient = result
    assert amount is None
    assert unit is None
    assert ingredient == "()"  # Should fallback to original text
