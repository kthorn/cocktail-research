"""Database utility functions for cocktail recipe databases."""

import contextlib
import pathlib
import sqlite3
from typing import Generator, Union


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