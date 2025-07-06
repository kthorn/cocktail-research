import json
import pytest

from cocktail_utils.database import create_schema, get_connection
from cocktail_utils.ingredients.rationalization import IngredientParser


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_recipes.db"
    conn = get_connection(db_path)
    create_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def ingredient_parser(tmp_path):
    return IngredientParser(
        db_path=str(tmp_path / "test_recipes.db"),
    )


@pytest.mark.parametrize(
    "ingredient_text,expected_brand,expected_cleaned_text",
    [
        ("gin, preferably Hendricks", "hendricks", "gin"),
        ("dry vermouth such as Dolin", "dolin", "dry vermouth"),
        ("simple gin", None, "simple gin"),
        ("crème de cacao, preferably tempus fugit", "tempus fugit", "crème de cacao"),
        ("white rum, preferably caña brava", "caña brava", "white rum"),
        ("frangelico", "frangelico", "hazelnut liqueur"),
    ],
)
def test_extract_brand(
    ingredient_parser, ingredient_text, expected_brand, expected_cleaned_text
):
    brand, cleaned_text = ingredient_parser.extract_brand(ingredient_text)
    assert brand == expected_brand
    assert cleaned_text == expected_cleaned_text


@pytest.mark.parametrize(
    "ingredient_text,expected_match",
    [
        # Exact match test
        (
            "dry vermouth",
            {
                "specific_type": "dry vermouth",
                "category": "vermouth",
                "confidence": 1.0,
                "source": "dictionary",
            },
        ),
        (
            "gin, preferably st. george terroir",
            {
                "brand": "st. george terroir",
                "specific_type": None,
                "category": "gin",
                "confidence": 1.0,
                "source": "dictionary",
            },
        ),
        (
            "rum",
            {
                "specific_type": None,
                "category": "rum",
                "confidence": 1.0,
                "source": "dictionary",
            },
        ),
        # Word boundary match test
        (
            "some london dry gin",
            {
                "specific_type": "london dry gin",
                "category": "gin",
                "confidence": 0.9,
                "source": "dictionary",
            },
        ),
        # No match test
        ("unknown ingredient", None),
    ],
)
def test_rationalize(ingredient_parser, ingredient_text, expected_match):
    """Test dictionary lookup with various ingredient texts."""
    match = ingredient_parser.rationalize_ingredient(ingredient_text)

    if expected_match is None:
        assert match is None
    else:
        assert match is not None
        assert match.specific_type == expected_match.get("specific_type")
        assert match.category == expected_match["category"]
        assert match.confidence == expected_match["confidence"]
        assert match.source == expected_match["source"]
        if "brand" in expected_match:
            assert match.brand == expected_match["brand"]


def test_get_ingredients_from_db(temp_db, ingredient_parser):
    cur = temp_db.cursor()
    cur.execute("INSERT INTO ingredient (id, name) VALUES (?, ?)", (1, "gin"))
    cur.execute("INSERT INTO ingredient (id, name) VALUES (?, ?)", (2, "vermouth"))
    cur.execute("INSERT INTO recipe (id, name) VALUES (?, ?)", (101, "Martini"))
    cur.execute("INSERT INTO recipe (id, name) VALUES (?, ?)", (102, "Negroni"))
    cur.execute("INSERT INTO recipe (id, name) VALUES (?, ?)", (103, "Gin & Tonic"))
    cur.execute(
        "INSERT INTO recipe_ingredient (recipe_id, ingredient_id, note) "
        "VALUES (?, ?, ?)",
        (101, 1, ""),
    )
    cur.execute(
        "INSERT INTO recipe_ingredient (recipe_id, ingredient_id, note) "
        "VALUES (?, ?, ?)",
        (101, 2, ""),
    )
    cur.execute(
        "INSERT INTO recipe_ingredient (recipe_id, ingredient_id, note) "
        "VALUES (?, ?, ?)",
        (102, 1, ""),
    )
    cur.execute(
        "INSERT INTO recipe_ingredient (recipe_id, ingredient_id, note) "
        "VALUES (?, ?, ?)",
        (103, 1, ""),
    )
    temp_db.commit()

    ingredients = ingredient_parser.get_ingredients_from_db(min_recipe_count=2)
    assert len(ingredients) == 1
    assert ingredients[0].ingredient_name == "gin"
    assert ingredients[0].recipe_count == 3
    assert set(ingredients[0].sample_recipes) == {"Martini", "Negroni", "Gin & Tonic"}

    ingredients_single = ingredient_parser.get_ingredients_from_db(min_recipe_count=1)
    assert len(ingredients_single) == 2


def test_llm_lookup_caching(ingredient_parser, mocker):
    # Mock the boto3 client to prevent actual API calls
    mock_invoke_model = mocker.patch.object(
        ingredient_parser.bedrock_client, "invoke_model"
    )

    # Simulate a response for the first call
    mock_invoke_model.return_value = {
        "body": mocker.Mock(
            read=lambda: json.dumps(
                {
                    "completion": json.dumps(
                        {
                            "brand": None,
                            "specific_type": "London Dry Gin",
                            "category": "Gin",
                        }
                    )
                }
            )
        )
    }

    ingredient_text = "gin"
    model_id = "test_model"

    # First call - should hit the mocked API
    match1 = ingredient_parser.llm_lookup(ingredient_text, model_id)
    assert match1 is not None
    assert match1.specific_type == "London Dry Gin"
    mock_invoke_model.assert_called_once()

    # Second call - should hit the cache
    match2 = ingredient_parser.llm_lookup(ingredient_text, model_id)
    assert match2 is not None
    assert match2.specific_type == "London Dry Gin"
    mock_invoke_model.assert_called_once()  # Should not be called again


def test_llm_lookup_error_handling(ingredient_parser, mocker):
    mock_invoke_model = mocker.patch.object(
        ingredient_parser.bedrock_client, "invoke_model"
    )
    mock_invoke_model.side_effect = Exception("API Error")

    match = ingredient_parser.llm_lookup("gin", "test_model")
    assert match is None
