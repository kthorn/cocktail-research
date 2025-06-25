#!/usr/bin/env python3
"""
Parses downloaded HTML recipe files and loads them into a SQLite database.
"""

import json
import logging
import pathlib
from typing import Dict, Any

from bs4 import BeautifulSoup
from tqdm.auto import tqdm

# Import from our new library
from cocktail_utils.database import get_connection, create_schema, transaction, upsert_ingredient
from cocktail_utils.ingredients import parse_quantity

# Config
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DB_PATH = pathlib.Path("punch_recipes.db")
HTML_DIR = pathlib.Path("punch_html")

def setup_database():
    """Set up database connection and schema."""
    conn = get_connection(DB_PATH)
    create_schema(conn)
    return conn

def parse_recipe_html(html_content: str, source_url: str) -> Dict[str, Any]:
    """Parse a single recipe from its HTML content."""
    soup = BeautifulSoup(html_content, "lxml")

    title = soup.find("h1").get_text(" ", strip=True)

    ing_ul = soup.find("h5", string=lambda text: text and "Ingredients" in text).find_next("ul")
    raw_ings = [li.get_text(" ", strip=True) for li in ing_ul.select("li")][::2]

    # Use our library's parsing function
    ingredients = [parse_quantity(t) for t in raw_ings]

    steps = "\n".join(
        li.get_text(" ", strip=True)
        for li in soup.find("h5", string=lambda text: text and "Directions" in text)
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
        "source_url": source_url,
    }


def load_recipe(conn, rec: Dict[str, Any]):
    """Load recipe into database using library utilities."""
    with transaction(conn) as cur:
        cur.execute(
            "INSERT INTO recipe(name, method, source_url) VALUES (?,?,?) "
            "ON CONFLICT(source_url) DO NOTHING",
            (rec["title"], rec["instructions"], rec["source_url"]),
        )
        cur.execute("SELECT id FROM recipe WHERE source_url = ?", (rec["source_url"],))
        result = cur.fetchone()
        if not result:
            # Recipe was already present, so we skip
            return
        rid = result[0]

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


def main():
    """Main function to parse downloaded recipes and load them into the database."""
    if not HTML_DIR.exists():
        logger.error(f"HTML directory not found: {HTML_DIR}")
        logger.info("Please run examples/download_punch_recipes.py first.")
        return

    conn = setup_database()

    # Get all existing source_urls from the DB to avoid re-processing
    with conn:
        existing_urls = set(
            row[0] for row in conn.execute("SELECT source_url FROM recipe")
        )

    html_files = list(HTML_DIR.glob("*.html"))

    for html_file in tqdm(html_files, desc="Parsing Recipes"):
        slug = html_file.stem
        # Reconstruct the source URL from the filename slug
        # This relies on the structure: https://punchdrink.com/recipes/{slug}/
        url = f"https://punchdrink.com/recipes/{slug}/"

        if url in existing_urls:
            # No need to log here, as it's expected for most files on re-runs
            continue

        try:
            html_content = html_file.read_text(encoding='utf-8')
            data = parse_recipe_html(html_content, url)
            load_recipe(conn, data)
            logger.info(f"✓ Parsed and loaded {data['title']}")
        except Exception as e:
            logger.error(f"⚠ Error parsing {html_file.name}: {e}")
            continue

if __name__ == "__main__":
    main()
