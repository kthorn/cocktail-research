"""Database schema definitions for cocktail recipe databases."""

import sqlite3

DDL = """
CREATE TABLE IF NOT EXISTS ingredient(
    id   INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS recipe(
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    source_url   TEXT UNIQUE,
    source_file  TEXT UNIQUE,
    description  TEXT,
    garnish      TEXT,
    directions   TEXT,
    editors_note TEXT
);

CREATE TABLE IF NOT EXISTS recipe_ingredient(
    recipe_id     INTEGER,
    ingredient_id INTEGER,
    amount        REAL,
    unit          TEXT,
    note          TEXT,
    PRIMARY KEY(recipe_id, ingredient_id, note),
    FOREIGN KEY(recipe_id)     REFERENCES recipe(id)     ON DELETE CASCADE,
    FOREIGN KEY(ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
);

"""


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the database schema for cocktail recipes.

    Args:
        conn: SQLite database connection
    """
    conn.executescript(DDL)
    conn.execute("PRAGMA foreign_keys = ON")
