#!/usr/bin/env python3
"""
Punch → SQLite recipe scraper
--------------------------------
• Discovers every recipe URL on https://punchdrink.com/recipes/
• Parses title, ingredients, instructions, tags
• Inserts into a local SQLite DB that follows your cocktaildb schema
   (tables are created if they don't exist)
• Crawls politely: honours robots.txt Disallow rules + crawl-delay and
  waits 1-3 s (delay + jitter) between all HTTP requests
"""

import contextlib
import pathlib
import random
import re
import sqlite3
import time
from decimal import Decimal, InvalidOperation
from urllib import robotparser
from urllib.parse import urljoin
from functools import wraps
import logging

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

# ── Config ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://punchdrink.com"
DB_PATH = pathlib.Path("punch_recipes.db")
USER_AGENT = "cocktail-bot/1.0 (+mailto:kurt.thorn@gmail.com)"

# Algolia API config - values from https://punchdrink.com/recipes/ network requests
ALGOLIA_APP_ID = "H0IEE3ERGC"
ALGOLIA_API_KEY = "9a128c4989675ec375c59a2de9ef3fc1"  # Public search-only key
ALGOLIA_INDEX = "wp_posts_recipe"  # This is the index name used in their API calls

# ── Polite crawling helpers ─────────────────────────────────────────
rp = robotparser.RobotFileParser()
rp.set_url(urljoin(BASE_URL, "/robots.txt"))
rp.read()


def allowed(url: str, ua: str = USER_AGENT) -> bool:
    return rp.can_fetch(ua, url)


CRAWL_DELAY = rp.crawl_delay(USER_AGENT) or 1.0  # seconds

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def retry_on_connection_error(max_retries=3, initial_delay=1):
    """Decorator that retries a function on connection errors with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ReadTimeout,
                    ConnectionResetError,
                ) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Connection error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise last_exception
            return None

        return wrapper

    return decorator


@retry_on_connection_error()
def polite_get(url: str) -> requests.Response:
    if not allowed(url):
        raise RuntimeError(f"robots.txt disallows {url}")
    time.sleep(CRAWL_DELAY + random.uniform(0.5, 1.5))
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r


# ── SQLite helpers ──────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA foreign_keys = ON")

DDL = """
CREATE TABLE IF NOT EXISTS ingredient(
    id   INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS recipe(
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    method      TEXT,
    source_url  TEXT UNIQUE
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

CREATE TABLE IF NOT EXISTS tag(
    id   INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS recipe_tag(
    recipe_id INTEGER,
    tag_id    INTEGER,
    PRIMARY KEY(recipe_id, tag_id),
    FOREIGN KEY(recipe_id) REFERENCES recipe(id) ON DELETE CASCADE,
    FOREIGN KEY(tag_id)    REFERENCES tag(id)    ON DELETE CASCADE
);
"""
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
    cur.execute(
        "INSERT INTO ingredient(name) VALUES (?) ON CONFLICT(name) DO NOTHING", (name,)
    )
    cur.execute("SELECT id FROM ingredient WHERE name = ?", (name,))
    return cur.fetchone()[0]


# ── Quantity parsing ────────────────────────────────────────────────
UNICODE_FRAC = {"¼": ".25", "½": ".5", "¾": ".75", "⅓": ".333", "⅔": ".667"}

# Unit normalization mapping
UNIT_MAP = {
    # Volume
    "ounce": ["ounce", "ounces", "oz", "oz."],
    "tablespoon": [
        "tablespoon",
        "tablespoons",
        "tbsp",
        "tbsp.",
        "tablespoonful",
        "tablespoonfuls",
    ],
    "teaspoon": ["teaspoon", "teaspoons", "tsp", "tsp.", "teaspoonful", "teaspoonfuls"],
    "cup": ["cup", "cups"],
    "pint": ["pint", "pints", "pt", "pt."],
    "quart": ["quart", "quarts", "qt", "qt."],
    "gallon": ["gallon", "gallons", "gal", "gal."],
    "ml": ["milliliter", "milliliters", "ml", "ml."],
    "cl": ["centiliter", "centiliters", "cl", "cl."],
    "l": ["liter", "liters", "l", "l."],
    # Count/measure
    "dash": ["dash", "dashes"],
    "drop": ["drop", "drops"],
    "part": ["part", "parts"],
    "splash": ["splash", "splashes"],
    "pinch": ["pinch", "pinches"],
    "piece": ["piece", "pieces", "shoulder"],
    "slice": ["slice", "slices"],
    "wedge": ["wedge", "wedges"],
    "whole": ["whole"],
}

# Create reverse mapping for lookup
UNIT_LOOKUP = {v: k for k, vs in UNIT_MAP.items() for v in vs}


def normalize_unit(unit: str) -> str:
    """Normalize unit names to their standard form."""
    unit = unit.lower().strip(".")
    return UNIT_LOOKUP.get(unit, unit)  # Return original if not found


def parse_qty(text: str):
    """Parse ingredient quantities, handling special cases like 'heavy', 'scant', and 'to top'."""
    original_text = text.strip()
    t = original_text.lower()

    # Handle "to top" ingredients (like ginger beer, tonic water, etc)
    if " to top" in t or " as needed" in t:
        # First extract the ingredient part before ", to top"
        parts = original_text.split(", to top")
        if len(parts) == 1:  # Try " to top" without comma
            parts = original_text.split(" to top")
        if len(parts) == 1:  # Try "as needed"
            parts = original_text.split(" as needed")

        ingredient_part = parts[0].strip()
        ingredient_part_lower = ingredient_part.lower()

        # Try to extract quantity information from the ingredient part
        # Check for ranges like "4 to 6 ounces Dr Pepper"
        range_match = re.match(
            r"^(\d+(?:\s+\d+/\d+)?)\s+to\s+(\d+(?:\s+\d+/\d+)?)\s+(\w+)\s+(.+)",
            ingredient_part_lower,
        )
        if range_match:
            # Extract the ingredient name (group 4)
            ingredient_name = re.match(
                r"^(\d+(?:\s+\d+/\d+)?)\s+to\s+(\d+(?:\s+\d+/\d+)?)\s+(\w+)\s+(.+)",
                ingredient_part,
                re.IGNORECASE,
            ).group(4)
            return 0, "to top", clean_ingredient_name(ingredient_name)

        # Check for simple quantities like "2 ounces Dr Pepper"
        simple_qty_match = re.match(
            r"^(\d+(?:\.\d+)?(?:\s+\d+/\d+)?)\s+(\w+)\s+(.+)", ingredient_part_lower
        )
        if simple_qty_match:
            # Extract the ingredient name (group 3)
            ingredient_name = re.match(
                r"^(\d+(?:\.\d+)?(?:\s+\d+/\d+)?)\s+(\w+)\s+(.+)",
                ingredient_part,
                re.IGNORECASE,
            ).group(3)
            return 0, "to top", clean_ingredient_name(ingredient_name)

        # Fallback to original behavior - clean the whole ingredient part
        return 0, "to top", clean_ingredient_name(ingredient_part)

    # Remove parenthetical quantities at the start
    t = re.sub(r"^\([^)]*\)\s*", "", t)
    original_text = re.sub(r"^\([^)]*\)\s*", "", original_text)

    # Remove special quantity words
    for word in ["heavy", "scant", "about"]:
        if t.startswith(word + " "):
            t = t[len(word) :].strip()
            # Also remove from original text, preserving case
            word_pattern = re.compile(r"^" + re.escape(word) + r"\s+", re.IGNORECASE)
            original_text = word_pattern.sub("", original_text).strip()

    # Handle ranges (e.g. "1 to 2 ounces")
    range_match = re.match(r"^(\d+(?:\s+\d+/\d+)?)\s+to\s+(\d+(?:\s+\d+/\d+)?)\s+", t)
    if range_match:
        # Convert both numbers to decimals and take average
        start = Decimal(str(eval(range_match.group(1))))
        end = Decimal(str(eval(range_match.group(2))))
        amt = float((start + end) / 2)
        # Remove the range part from both versions
        t = t[range_match.end() :]
        # Find the same pattern in original text
        original_range_match = re.match(
            r"^(\d+(?:\s+\d+/\d+)?)\s+to\s+(\d+(?:\s+\d+/\d+)?)\s+",
            original_text,
            re.IGNORECASE,
        )
        if original_range_match:
            original_text = original_text[original_range_match.end() :]
        parts = t.split()
        original_parts = original_text.split()
        unit = normalize_unit(parts[0])
        rest = " ".join(original_parts[1:]) if len(original_parts) > 1 else ""
        return amt, unit, clean_ingredient_name(rest)

    # Convert unicode fractions
    for k, v in UNICODE_FRAC.items():
        t = t.replace(k, v)

    parts = t.split()
    original_parts = original_text.split()
    try:
        # Check if first part is a unit (like "ounces vodka")
        if parts[0] in UNIT_LOOKUP:
            amt = Decimal(1)
            unit = normalize_unit(parts[0])
            rest = " ".join(original_parts[1:])
        else:
            # Try to parse as a mixed number (e.g., "2 3/4" or just "2" or "3/4")
            amt = None
            unit_idx = 1  # Default assumption: first part is number, second is unit

            # Check for mixed numbers like "2 3/4"
            if (
                len(parts) >= 2
                and re.match(r"^\d+$", parts[0])
                and re.match(r"^\d+/\d+$", parts[1])
            ):
                # Mixed number: "2 3/4"
                whole = int(parts[0])
                frac_parts = parts[1].split("/")
                fraction = Decimal(frac_parts[0]) / Decimal(frac_parts[1])
                amt = Decimal(whole) + fraction
                unit_idx = 2
            else:
                # Try to parse the first part as a number (could be "2", "3/4", "2.5", etc.)
                try:
                    amt = Decimal(str(eval(parts[0])))  # "3/4" → 0.75, "2" → 2
                except:
                    # If that fails, maybe it's a decimal
                    amt = Decimal(parts[0])

            if unit_idx < len(parts):
                unit = normalize_unit(parts[unit_idx])
                rest = (
                    " ".join(original_parts[unit_idx + 1 :])
                    if unit_idx + 1 < len(original_parts)
                    else ""
                )
            else:
                # No unit found
                unit = None
                rest = " ".join(original_parts[1:]) if len(original_parts) > 1 else ""
    except (InvalidOperation, IndexError, SyntaxError, NameError):
        # If we can't parse a quantity, treat the whole line as ingredient name
        amt = None
        unit = None
        rest = " ".join(original_parts)

    # Clean the ingredient name
    rest = clean_ingredient_name(rest)

    # If we ended up with an empty ingredient name, use the original text
    if not rest:
        rest = original_text

    return float(amt) if amt is not None else None, unit, rest


def clean_ingredient_name(name: str) -> str:
    """Clean up ingredient names by removing parenthetical notes and other formatting."""
    # Remove parenthetical quantities at the start (including "about", "heavy", etc)
    name = re.sub(r"^\([^)]*\)\s*", "", name)

    # Remove parenthetical notes in the middle/end
    name = re.sub(r"\s*\([^)]*\)", "", name)

    # Remove extra whitespace and commas
    name = re.sub(r"\s+", " ", name)
    name = name.strip().strip(",")

    return name


# ── Algolia API helpers ────────────────────────────────────────────
@retry_on_connection_error()
def search_recipes(page: int = 0, hits_per_page: int = 100) -> list:
    """Search recipes using Algolia API"""
    url = f"https://{ALGOLIA_APP_ID}-dsn.algolia.net/1/indexes/*/queries"
    headers = {
        "X-Algolia-API-Key": ALGOLIA_API_KEY,
        "X-Algolia-Application-Id": ALGOLIA_APP_ID,
        "X-Algolia-Agent": "Algolia for JavaScript (4.5.1); Browser (lite); instantsearch.js (4.8.3); JS Helper (3.2.2)",
    }
    data = {
        "requests": [
            {
                "indexName": ALGOLIA_INDEX,
                "params": f"hitsPerPage={hits_per_page}&facetingAfterDistinct=true&query=&maxValuesPerFacet=20&page={page}&distinct=false&filters=record_index%3D0&facets=%5B%22spirit%22%2C%22style%22%2C%22season%22%2C%22flavor_profile%22%2C%22family%22%5D&tagFilters=",
            }
        ]
    }

    response = session.post(url, json=data, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()["results"][
        0
    ]  # The API returns a list of results, we want the first one


def list_recipe_urls():
    """Get all recipe URLs using Algolia API"""
    page = 0
    while True:
        try:
            results = search_recipes(page=page)
            hits = results.get("hits", [])
            if not hits:
                break

            for hit in hits:
                url = hit.get(
                    "permalink"
                )  # The API returns full URLs in the permalink field
                if url:
                    yield url

            # Check if we've reached the end
            if page >= results.get("nbPages", 0) - 1:
                break

            page += 1
            time.sleep(CRAWL_DELAY)  # Be nice to Algolia's API

        except Exception as e:
            logger.error(f"Error fetching page {page}: {e}")
            # Add a longer delay before retrying the next page
            time.sleep(CRAWL_DELAY * 2)
            continue  # Try the next page instead of breaking


def scrape_recipe(url: str):
    soup = BeautifulSoup(polite_get(url).text, "lxml")

    title = soup.find("h1").get_text(" ", strip=True)

    ing_ul = soup.find("h5", string=re.compile("Ingredients", re.I)).find_next("ul")
    raw_ings = [li.get_text(" ", strip=True) for li in ing_ul.select("li")][::2]

    ingredients = [parse_qty(t) for t in raw_ings]

    steps = "\n".join(
        li.get_text(" ", strip=True)
        for li in soup.find("h5", string=re.compile("Directions", re.I))
        .find_next("ol")
        .select("li")
    )

    tags = [
        a.get_text(strip=True)
        for a in soup.select("a[href*='/recipes/']:not(:has(img))")
        if a.get_text(strip=True)
    ]

    return {
        "title": title,
        "instructions": steps,
        "tags": list(dict.fromkeys(tags)),
        "ingredients": [
            {"amount": amt, "unit": unit, "name": name}
            for amt, unit, name in ingredients
        ],
        "source_url": url,
    }


# ── DB loader ───────────────────────────────────────────────────────
def load_recipe(rec):
    with tx() as cur:
        cur.execute(
            "INSERT INTO recipe(name, method, source_url) VALUES (?,?,?) "
            "ON CONFLICT(source_url) DO NOTHING",
            (rec["title"], rec["instructions"], rec["source_url"]),
        )
        cur.execute("SELECT id FROM recipe WHERE source_url = ?", (rec["source_url"],))
        rid = cur.fetchone()[0]

        for it in rec["ingredients"]:
            iid = upsert_ingredient(cur, it["name"])
            cur.execute(
                "INSERT INTO recipe_ingredient(recipe_id, ingredient_id, amount, unit, note) "
                "VALUES (?,?,?,?,?)",
                (rid, iid, it["amount"], it["unit"], None),
            )

        for tag in rec["tags"]:
            cur.execute(
                "INSERT INTO tag(name) VALUES (?) ON CONFLICT(name) DO NOTHING", (tag,)
            )
            cur.execute("SELECT id FROM tag WHERE name = ?", (tag,))
            tid = cur.fetchone()[0]
            cur.execute(
                "INSERT OR IGNORE INTO recipe_tag(recipe_id, tag_id) VALUES (?,?)",
                (rid, tid),
            )


# ── Driver ──────────────────────────────────────────────────────────
def main():
    # Get all existing source_urls from the DB
    with conn:
        existing_urls = set(
            row[0] for row in conn.execute("SELECT source_url FROM recipe")
        )

    for url in list_recipe_urls():
        if url in existing_urls:
            logger.info(f"Skipping already downloaded recipe: {url}")
            continue
        try:
            data = scrape_recipe(url)
            load_recipe(data)
            logger.info(f"✓ {data['title']}")
        except Exception as e:
            logger.error(f"⚠ {url} ({str(e)})")
            continue  # Continue with next URL even if one fails


if __name__ == "__main__":
    main()
