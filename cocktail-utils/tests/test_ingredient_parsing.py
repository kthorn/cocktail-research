import pytest
from cocktail_utils.ingredients.parsing import (
    _parse_amount,
    _parse_unit,
    normalize_ingredient_text,
    normalize_unit,
    parse_quantity,
    clean_ingredient_name,
    extract_brand,
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
    "input_text, expected_brand, expected_name",
    [
        ("gin, preferably Hendrick's", "Hendrick's", "gin"),
        ("dry vermouth such as Dolin", "Dolin", "dry vermouth"),
        ("whiskey", None, "whiskey"),
        (
            "Japanese whisky (preferably Suntory Toki)",
            "Suntory Toki",
            "Japanese whisky",
        ),
    ],
)
def test_extract_brand(input_text, expected_brand, expected_name):
    """Test brand extraction from ingredient text."""
    brand, name = extract_brand(input_text)
    assert brand == expected_brand
    assert name == expected_name


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
        ("", None, ""),
    ],
)
def test_parse_unit(input_text, expected_unit, expected_rest):
    """Test the private helper _parse_unit."""
    unit, rest = _parse_unit(input_text)
    assert unit == expected_unit
    assert rest == expected_rest
