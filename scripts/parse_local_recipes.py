"""Parse local HTML recipe files and load them into a SQLite database."""

import pathlib

from tqdm import tqdm

from cocktail_utils.database import get_connection, transaction, upsert_ingredient
from cocktail_utils.database.schema import create_schema
from cocktail_utils.recipes import parse_recipe_html
from cocktail_utils.ingredients import parse_quantity


def word_set(text: str) -> set[str]:
    return set(word.lower().strip() for word in text.split())


def load_skip_files() -> list[str]:
    """Load the list of files to skip from punch_files_to_skip.txt, excluding comment lines."""
    skip_file_path = pathlib.Path(
        "cocktail-utils/src/cocktail_utils/ingredients/data/punch_files_to_skip.txt"
    )
    skip_files = []

    if skip_file_path.exists():
        with open(skip_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    skip_files.append(line)

    return skip_files


def main():
    """Main function to parse recipes and load them into the database."""
    db_path = "data/recipes.db"
    conn = get_connection(db_path)
    create_schema(conn)

    # Find all HTML files in the raw_recipes directory
    recipe_files = list(pathlib.Path("raw_recipes").rglob("*.html"))
    skip_files = load_skip_files()

    with transaction(conn) as cur:
        for file_path in tqdm(recipe_files, desc="Parsing recipes"):
            if str(file_path) in skip_files:
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()

            recipe = parse_recipe_html(html)

            if recipe:
                # Insert the recipe
                cur.execute(
                    """INSERT INTO recipe (name, source_file, description, garnish, directions, editors_note)
                       VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(source_file) DO NOTHING""",
                    (
                        recipe.name,
                        str(file_path),
                        recipe.description,
                        recipe.garnish,
                        "\n".join(recipe.directions),
                        recipe.editors_note,
                    ),
                )
                cur.execute("SELECT id FROM recipe WHERE source_file = ?", (str(file_path),))
                recipe_id = cur.fetchone()[0]

                # Insert the ingredients
                for ingredient_text in recipe.ingredients:
                    skip_words = ["optional", "rim", "to taste"]
                    # skip optional ingredients and ingredients for rimming the glass
                    if any(word in ingredient_text.lower() for word in skip_words):
                        continue
                    ingredient_text = ingredient_text.replace("bar spoon", "barspoon")
                    amount, unit, ingredient_name = parse_quantity(ingredient_text)

                    # some ingredients no amount or unit and in the directions it specifies to top with them
                    # try to find these
                    # We check that at least one word is in common between the ingredient and the direction line
                    # e.g. if the ingredient is "soda water" and the direction line is "top with soda"
                    if amount is None and unit is None:
                        # Check each direction line for "top with" and see if ingredient has word in common
                        for direction_line in recipe.directions:
                            if "top" in direction_line.lower():
                                ingredient_words = word_set(ingredient_name)
                                direction_words = word_set(direction_line)
                                if ingredient_words & direction_words:
                                    amount = 0.0
                                    unit = "to top"
                                    break
                        # even stranger, some recipes call for "soda water" without specifying anything else
                        # we assume these are "to top"
                        if "soda water" in ingredient_name.lower():
                            amount = 0.0
                            unit = "to top"

                    ingredient_id = upsert_ingredient(cur, ingredient_name)

                    cur.execute(
                        """INSERT INTO recipe_ingredient (recipe_id, ingredient_id, amount, unit, note)
                           VALUES (?, ?, ?, ?, ?) ON CONFLICT(recipe_id, ingredient_id, note) DO NOTHING""",
                        (recipe_id, ingredient_id, amount, unit, ""),
                    )


if __name__ == "__main__":
    main()
