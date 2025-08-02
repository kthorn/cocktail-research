"""Database utility functions for cocktail recipe databases."""

import contextlib
import pathlib
import sqlite3
from typing import Generator, Union

import pandas as pd


def get_connection(db_path: Union[str, pathlib.Path]) -> sqlite3.Connection:
    """Get a SQLite database connection with foreign keys enabled.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        SQLite connection with foreign keys enabled
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Cursor, None, None]:
    """Context manager for database transactions.

    Args:
        conn: SQLite database connection

    Yields:
        Database cursor for executing queries

    Example:
        with transaction(conn) as cur:
            cur.execute("INSERT INTO ingredient(name) VALUES (?)", ("gin",))
    """
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def upsert_ingredient(cur: sqlite3.Cursor, name: str) -> int:
    """Insert ingredient if it doesn't exist, return its ID.

    Args:
        cur: Database cursor
        name: Ingredient name

    Returns:
        Integer ID of the ingredient
    """
    cur.execute(
        "INSERT INTO ingredient(name) VALUES (?) ON CONFLICT(name) DO NOTHING", (name,)
    )
    cur.execute("SELECT id FROM ingredient WHERE name = ?", (name,))
    return cur.fetchone()[0]


def get_recipe_ingredient_data(db_path: Union[str, pathlib.Path]):
    """Get all recipe-ingredient relationships from the database with standardized units.

    Args:
        db_path: Path to the SQLite database

    Returns:
        DataFrame with columns: recipe_name, ingredient_name, amount_ml, original_amount, original_unit

    Raises:
        ImportError: If pandas is not installed
        Exception: If database query fails
    """
    if pd is None:
        raise ImportError("pandas is required for get_recipe_ingredient_data")

    print("Fetching recipe-ingredient relationships from database...")

    conn = get_connection(db_path)

    query = """
    SELECT 
        r.name as recipe_name,
        r.id as recipe_id,
        i.name as ingredient_name,
        ri.amount,
        ri.unit
    FROM recipe r
    JOIN recipe_ingredient ri ON r.id = ri.recipe_id
    JOIN ingredient i ON ri.ingredient_id = i.id
    ORDER BY r.name, i.name
    """

    try:
        df = pd.read_sql_query(query, conn)
        print(f"Found {len(df)} recipe-ingredient relationships")
        print(
            f"Recipes: {df['recipe_name'].nunique()}, Ingredients: {df['ingredient_name'].nunique()}"
        )

        # Convert units to ml
        from .units import convert_to_ml

        df["amount_ml"] = df.apply(
            lambda row: convert_to_ml(row["amount"], row["unit"]), axis=1
        )

        # Keep original values for reference
        df["original_amount"] = df["amount"]
        df["original_unit"] = df["unit"]

        # Reorder columns
        df = df[
            [
                "recipe_name",
                "recipe_id",
                "ingredient_name",
                "amount_ml",
                "original_amount",
                "original_unit",
            ]
        ]

        print(f"Converted units to ml for analysis")

        return df
    except Exception as e:
        print(f"Error querying database: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
