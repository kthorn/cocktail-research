#!/usr/bin/env python3
"""
Updated punch_dl.py using cocktail-utils library
"""

import json
import logging
import pathlib
import time
from typing import List, Dict, Any

from bs4 import BeautifulSoup
from tqdm.auto import tqdm

# Import from our new library
from cocktail_utils.database import get_connection, create_schema, transaction, upsert_ingredient
from cocktail_utils.ingredients import parse_quantity
from cocktail_utils.scraping import PoliteSession, retry_on_connection_error

# Config
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://punchdrink.com"
DB_PATH = pathlib.Path("punch_recipes.db")
USER_AGENT = "cocktail-bot/1.0 (+mailto:kurt.thorn@gmail.com)"

# Algolia API config
ALGOLIA_APP_ID = "H0IEE3ERGC"
ALGOLIA_API_KEY = "9a128c4989675ec375c59a2de9ef3fc1"
ALGOLIA_INDEX = "wp_posts_recipe"


def setup_database():
    """Set up database connection and schema."""
    conn = get_connection(DB_PATH)
    create_schema(conn)
    return conn


@retry_on_connection_error()
def search_recipes(session: PoliteSession, page: int = 0, hits_per_page: int = 100) -> list:
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

    response = session.session.post(url, json=data, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()["results"][0]


def list_recipe_urls(session: PoliteSession):
    """Get all recipe URLs using Algolia API"""
    page = 0
    while True:
        try:
            results = search_recipes(session, page=page)
            hits = results.get("hits", [])
            if not hits:
                break

            for hit in hits:
                url = hit.get("permalink")
                if url:
                    yield url

            if page >= results.get("nbPages", 0) - 1:
                break

            page += 1
            time.sleep(1)  # Be nice to Algolia's API

        except Exception as e:
            logger.error(f"Error fetching page {page}: {e}")
            time.sleep(2)
            continue


def scrape_recipe(session: PoliteSession, url: str) -> Dict[str, Any]:
    """Scrape a single recipe from URL."""
    response = session.get(url)
    soup = BeautifulSoup(response.text, "lxml")

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
        "source_url": url,
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


def main():
    """Main function using the cocktail-utils library."""
    conn = setup_database()
    session = PoliteSession(BASE_URL, USER_AGENT)

    # Get all existing source_urls from the DB
    with conn:
        existing_urls = set(
            row[0] for row in conn.execute("SELECT source_url FROM recipe")
        )

    for url in list_recipe_urls(session):
        if url in existing_urls:
            logger.info(f"Skipping already downloaded recipe: {url}")
            continue
        try:
            data = scrape_recipe(session, url)
            load_recipe(conn, data)
            logger.info(f"✓ {data['title']}")
        except Exception as e:
            logger.error(f"⚠ {url} ({str(e)})")
            continue


if __name__ == "__main__":
    main()