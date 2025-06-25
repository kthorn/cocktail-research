import pytest
from cocktail_utils.ingredients.normalization import normalize_ingredient_text


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("2 oz fresh lemon juice", "lemon juice"),
        ("1 tsp premium vanilla extract", "vanilla extract"),
        ("1/2 cup good quality gin", "gin"),
        ("3 dashes Angostura bitters", "angostura bitters"),
        ("5 drops saline solution", "saline solution"),
        ("Freshly grated nutmeg", "grated nutmeg"),
        ("  lots   of   whitespace  ", "lots of whitespace"),
        ("1 1/2 oz. bourbon", "bourbon"),
        ("2oz lime juice", "lime juice"),
        ("1 lime shoulder", "lime shoulder"),
    ],
)
def test_normalize_ingredient_text(input_text, expected_text):
    """Test that normalize_ingredient_text removes quantities, units, and prefixes."""
    assert normalize_ingredient_text(input_text) == expected_text


def test_normalize_ingredient_text_no_changes():
    """Test that normalize_ingredient_text doesn't change already-normalized text."""
    assert normalize_ingredient_text("lemon juice") == "lemon juice"


def test_normalize_ingredient_text_empty_string():
    """Test that normalize_ingredient_text handles empty strings."""
    assert normalize_ingredient_text("") == ""
