"""Parse local HTML recipe files and load them into a SQLite database."""

import pathlib
import sqlite3

from tqdm import tqdm

from cocktail_utils.database import get_connection, transaction, upsert_ingredient
from cocktail_utils.database.schema import create_schema
from cocktail_utils.recipes import parse_recipe_html
from cocktail_utils.ingredients import parse_quantity


def main():
    """Main function to parse recipes and load them into the database."""
    db_path = "data/recipes.db"
    conn = get_connection(db_path)
    create_schema(conn)

    # Find all HTML files in the raw_recipes directory
    recipe_files = list(pathlib.Path("raw_recipes").rglob("*.html"))

    with transaction(conn) as cur:
        for file_path in tqdm(recipe_files, desc="Parsing recipes"):
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
                cur.execute("SELECT id FROM recipe WHERE name = ?", (recipe.name,))
                recipe_id = cur.fetchone()[0]

                # Insert the ingredients
                for ingredient_text in recipe.ingredients:
                    amount, unit, ingredient_name = parse_quantity(ingredient_text)
                    ingredient_id = upsert_ingredient(cur, ingredient_name)

                    cur.execute(
                        """INSERT INTO recipe_ingredient (recipe_id, ingredient_id, amount, unit, note)
                           VALUES (?, ?, ?, ?, ?) ON CONFLICT(recipe_id, ingredient_id, note) DO NOTHING""",
                        (recipe_id, ingredient_id, amount, unit, ""),
                    )


if __name__ == "__main__":
    main()
