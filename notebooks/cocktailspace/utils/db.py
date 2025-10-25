"""Utility functions for working with the cocktail database."""

import sqlite3
import pandas as pd


# Configuration
DB_PATH = "backup-2025-10-21_08-00-45.db"


def load_recipes_from_db(db_path: str = DB_PATH) -> pd.DataFrame:
    """Load recipes and their ingredients from SQLite database.

    Args:
        db_path: Path to SQLite database
        recipe_names: Optional list of recipe names to filter by

    Returns:
        DataFrame with recipe and ingredient information including:
        - recipe_id, recipe_name
        - ingredient_id, ingredient_name, ingredient_path
        - amount, unit_id, conversion_to_ml
        - volume_ml: calculated volume (amount * conversion_to_ml)
        - volume_fraction: fraction of ingredient volume relative to total recipe volume
    """
    conn = sqlite3.connect(db_path)
    # Load all recipes
    query = """
    SELECT
        r.id as recipe_id,
        r.name as recipe_name,
        i.id as ingredient_id,
        i.name as ingredient_name,
        i.path as ingredient_path,
        i.substitution_level as substitution_level,
        ri.amount,
        u.conversion_to_ml * ri.amount as volume_ml,
        (u.conversion_to_ml * ri.amount) / 
            SUM(u.conversion_to_ml * ri.amount) OVER (PARTITION BY r.id) as volume_fraction
    FROM recipes r
    JOIN recipe_ingredients ri ON r.id = ri.recipe_id
    JOIN ingredients i ON ri.ingredient_id = i.id
    LEFT JOIN units u ON ri.unit_id = u.id
    ORDER BY r.id, i.id
    """
    df = pd.read_sql_query(query, conn)

    conn.close()

    if df.empty:
        raise ValueError("No recipes found matching the specified names")

    return df


def load_ingredients_from_db(db_path: str = DB_PATH) -> pd.DataFrame:
    """Load ingredient substitution levels from the database.

    Args:
        db_path: Path to SQLite database

    Returns:
        DataFrame with columns: id, name, path, substitution_level
        substitution_level is 0 for NULL/NaN values
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT id, name, path, COALESCE(substitution_level, 0) as substitution_level FROM ingredients"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
