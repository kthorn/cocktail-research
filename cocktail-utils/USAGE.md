# Using cocktail-utils

This document shows how to migrate from the original scripts to use the `cocktail-utils` library.

## Installation

From the cocktail-utils directory:

```bash
pip install -e .
```

Or to install from a git repository:

```bash
pip install git+https://github.com/yourusername/cocktail-utils.git
```

## Key Changes

### Database Operations

**Before:**
```python
import sqlite3
import contextlib

conn = sqlite3.connect("recipes.db")
conn.execute("PRAGMA foreign_keys = ON")

DDL = """CREATE TABLE IF NOT EXISTS ingredient..."""
conn.executescript(DDL)

@contextlib.contextmanager
def tx():
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise

def upsert_ingredient(cur, name: str) -> int:
    cur.execute("INSERT INTO ingredient(name) VALUES (?) ON CONFLICT(name) DO NOTHING", (name,))
    cur.execute("SELECT id FROM ingredient WHERE name = ?", (name,))
    return cur.fetchone()[0]
```

**After:**
```python
from cocktail_utils.database import get_connection, create_schema, transaction, upsert_ingredient

conn = get_connection("recipes.db")
create_schema(conn)

with transaction(conn) as cur:
    ingredient_id = upsert_ingredient(cur, "gin")
```

### Ingredient Parsing

**Before:**
```python
def parse_qty(text: str):
    # Complex parsing logic...
    return amount, unit, ingredient

def normalize_unit(unit: str) -> str:
    # Unit normalization logic...
    return normalized_unit

def clean_ingredient_name(name: str) -> str:
    # Name cleaning logic...
    return cleaned_name
```

**After:**
```python
from cocktail_utils.ingredients import parse_quantity, normalize_unit, clean_ingredient_name

amount, unit, ingredient = parse_quantity("2 oz gin")
normalized = normalize_unit("oz")  # Returns "ounce"
clean_name = clean_ingredient_name("fresh lemon juice (organic)")
```

### Web Scraping

**Before:**
```python
import requests
import time
import random
from urllib import robotparser
from functools import wraps

def retry_on_connection_error(max_retries=3):
    # Retry decorator logic...

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT

def polite_get(url: str):
    # Robots.txt checking and delay logic...
    return session.get(url)
```

**After:**
```python
from cocktail_utils.scraping import PoliteSession, retry_on_connection_error

session = PoliteSession("https://example.com", "mybot/1.0")

@retry_on_connection_error(max_retries=3)
def fetch_data(url):
    return session.get(url)  # Automatically respects robots.txt and delays
```

### Ingredient Normalization

**Before:**
```python
def normalize_ingredient(self, ingredient_text: str) -> str:
    # Remove quantities and measurements
    text = re.sub(r"^\d+[\s\d/]*\s*(ounces?|oz|cups?|tsp|tbsp|ml|cl|dashes?|drops?)\s+", "", ingredient_text, flags=re.IGNORECASE)
    # Remove common prefixes
    text = re.sub(r"^(fresh|freshly|good|quality|premium)\s+", "", text, flags=re.IGNORECASE)
    # Normalize whitespace
    return " ".join(text.lower().split())

def extract_brand(self, ingredient_text: str):
    # Brand extraction logic...
    return brand, cleaned_text
```

**After:**
```python
from cocktail_utils.ingredients import normalize_ingredient_text, extract_brand

normalized = normalize_ingredient_text("2 oz fresh lemon juice")
brand, cleaned = extract_brand("gin, preferably Hendrick's")
```

## Migration Checklist

1. **Install the library**: `pip install -e .` from the cocktail-utils directory
2. **Update imports**: Replace local function definitions with library imports
3. **Update database code**: Use `get_connection()`, `create_schema()`, and `transaction()` context manager
4. **Update parsing code**: Use `parse_quantity()` instead of `parse_qty()`
5. **Update scraping code**: Use `PoliteSession` instead of manual session management
6. **Test functionality**: Ensure all features work as expected with the new library

## Benefits

- **Reusable code**: Common functionality is now available across multiple projects
- **Better error handling**: Improved exception handling and retry logic
- **Cleaner imports**: Single import statements instead of copying code
- **Maintainability**: Bug fixes and improvements benefit all users
- **Documentation**: Better documented with type hints and docstrings
- **Testing**: Library can have comprehensive tests