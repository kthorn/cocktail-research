# Cocktail Utils

A Python library for cocktail recipe processing and analysis, extracted from common functionality used in cocktail research scripts.

## Features

- **Database utilities**: SQLite helpers for cocktail recipe databases
- **Ingredient parsing**: Quantity parsing, unit normalization, and ingredient name cleaning
- **Web scraping utilities**: Polite crawling with robots.txt compliance and retry logic

## Installation

```bash
pip install cocktail-utils
```

## Usage

### Database Utilities

```python
from cocktail_utils.database import get_connection, create_schema, transaction

# Create database with schema
conn = get_connection("recipes.db")
create_schema(conn)

# Use transaction context manager
with transaction(conn) as cur:
    cur.execute("INSERT INTO ingredient(name) VALUES (?)", ("gin",))
```

### Ingredient Parsing

```python
from cocktail_utils.ingredients import parse_quantity, normalize_unit, clean_ingredient_name

# Parse ingredient quantities
amount, unit, ingredient = parse_quantity("2 oz gin")
# Returns: (2.0, "ounce", "gin")

# Normalize units
unit = normalize_unit("oz")  # Returns "ounce"

# Clean ingredient names
name = clean_ingredient_name("fresh lemon juice (about 1 lemon)")
# Returns: "fresh lemon juice"
```

### Web Scraping

```python
from cocktail_utils.scraping import polite_get, retry_on_connection_error

# Polite web scraping with robots.txt compliance
response = polite_get("https://example.com/recipe")

# Retry decorator for unreliable connections
@retry_on_connection_error(max_retries=3)
def fetch_data(url):
    return requests.get(url)
```

## Development

```bash
git clone https://github.com/kurtthorn/cocktail-utils
cd cocktail-utils
pip install -e .[dev]
```

## License

MIT