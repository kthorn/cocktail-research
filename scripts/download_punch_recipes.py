#!/usr/bin/env python3
"""
Downloads recipe pages from punchdrink.com to a local directory.
"""

import logging
import pathlib
import time
import re
from urllib.parse import urlparse

from tqdm.auto import tqdm

from cocktail_utils.scraping import PoliteSession, retry_on_connection_error

# Config
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://punchdrink.com"
OUTPUT_DIR = pathlib.Path("/home/kurtt/cocktail-research/raw_recipes/punch_html")
USER_AGENT = "cocktail-bot/1.0 (+mailto:kurt.thorn@gmail.com)"

# Algolia API config
ALGOLIA_APP_ID = "H0IEE3ERGC"
ALGOLIA_API_KEY = "9a128c4989675ec375c59a2de9ef3fc1"
ALGOLIA_INDEX = "wp_posts_recipe"


@retry_on_connection_error()
def search_recipes(
    session: PoliteSession,
    page: int = 0,
    hits_per_page: int = 1000,
    facet_filters: list[str] | None = None,
) -> list:
    """Search recipes using Algolia API"""
    url = f"https://{ALGOLIA_APP_ID}-dsn.algolia.net/1/indexes/*/queries"
    headers = {
        "X-Algolia-API-Key": ALGOLIA_API_KEY,
        "X-Algolia-Application-Id": ALGOLIA_APP_ID,
        "X-Algolia-Agent": "Algolia for JavaScript (4.5.1); Browser (lite); instantsearch.js (4.8.3); JS Helper (3.2.2)",
    }

    # Build the request parameters properly
    request_params = {
        "indexName": ALGOLIA_INDEX,
        "hitsPerPage": hits_per_page,
        "facetingAfterDistinct": True,
        "query": "",
        "maxValuesPerFacet": 50,
        "page": page,
        "distinct": False,
        "filters": "record_index=0",
        "facets": ["spirit", "style", "season", "flavor_profile", "family"],
        "tagFilters": [],
    }

    # Add facetFilters if provided
    if facet_filters:
        request_params["facetFilters"] = facet_filters

    data = {"requests": [request_params]}

    response = session.session.post(url, json=data, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()["results"][0]


def get_facet_values(session: PoliteSession, facet_name: str) -> list[str]:
    """Get all possible values for a given facet from Algolia."""
    results = search_recipes(session, hits_per_page=0)
    return list(results.get("facets", {}).get(facet_name, {}).keys())


def list_recipe_urls(session: PoliteSession):
    """Get all recipe URLs using Algolia API, paginating by spirit."""
    spirits = get_facet_values(session, "spirit")
    logger.info(f"Found {len(spirits)} spirits: {spirits}")
    all_urls = set()

    for spirit in tqdm(spirits, desc="Fetching recipe URLs by spirit"):
        page = 0
        while True:
            try:
                facet_filter = f"spirit:{spirit}"
                results = search_recipes(
                    session, page=page, facet_filters=[facet_filter]
                )
                hits = results.get("hits", [])
                if not hits:
                    break

                for hit in hits:
                    url = hit.get("permalink")
                    if url:
                        all_urls.add(url)

                if page >= results.get("nbPages", 0) - 1:
                    break

                page += 1
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching page {page} for spirit {spirit}: {e}")
                time.sleep(2)
                continue
    return list(all_urls)


def url_to_filename(url: str) -> str:
    """Generate a safe filename from a URL."""
    path = urlparse(url).path
    slug = path.strip("/").split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", slug) + ".html"


def download_recipe(session: PoliteSession, url: str, output_path: pathlib.Path):
    """Download a single recipe and save it to a file."""
    try:
        response = session.get(url)
        response.raise_for_status()
        output_path.write_text(response.text, encoding="utf-8")
        logger.info(f"✓ Downloaded {url}")
    except Exception as e:
        logger.error(f"⚠ Failed to download {url}: {e}")


def main():
    """Main function to download recipes."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    session = PoliteSession(BASE_URL, USER_AGENT)

    recipe_urls = list_recipe_urls(session)

    for url in tqdm(recipe_urls, desc="Downloading Recipes"):
        filename = url_to_filename(url)
        output_path = OUTPUT_DIR / filename

        if output_path.exists():
            continue

        download_recipe(session, url, output_path)


if __name__ == "__main__":
    main()
