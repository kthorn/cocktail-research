#!/usr/bin/env python3
"""
Downloads recipe pages from diffordsguide.com to a local directory.
"""

import logging
import pathlib
import time
import re
from urllib.parse import urlparse
import requests
from tqdm.auto import tqdm

from cocktail_utils.scraping import PoliteSession, retry_on_connection_error

# Config
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.diffordsguide.com"
OUTPUT_DIR = pathlib.Path("/home/kurtt/cocktail-research/raw_recipes/diffords_html")
USER_AGENT = "cocktail-bot/1.0 (+mailto:kurt.thorn@gmail.com)"

# Recipe ID range - we'll need to determine the upper bound
MIN_RECIPE_ID = 1
MAX_RECIPE_ID = 10000  # Conservative estimate, will stop when we hit 404s


@retry_on_connection_error()
def check_recipe_exists(session: PoliteSession, recipe_id: int) -> bool:
    """Check if a recipe exists by ID."""
    url = f"{BASE_URL}/cocktails/recipe/{recipe_id}/"
    try:
        response = session.get(url, allow_redirects=False)
        # A valid recipe should return 200
        # 404 means recipe doesn't exist
        # 301/302 might be redirects to valid recipes
        return response.status_code == 200 or response.status_code in [301, 302]
    except Exception as e:
        logger.error(f"Error checking recipe {recipe_id}: {e}")
        return False


def find_max_recipe_id(session: PoliteSession, start_max: int = 10000) -> int:
    """Binary search to find the maximum recipe ID."""
    logger.info("Finding maximum recipe ID...")
    
    # First, check if our initial max is too low
    if check_recipe_exists(session, start_max):
        # Double until we find a non-existent recipe
        while check_recipe_exists(session, start_max):
            start_max *= 2
            time.sleep(0.5)
    
    # Binary search between MIN and our found max
    low, high = MIN_RECIPE_ID, start_max
    max_found = MIN_RECIPE_ID
    
    while low <= high:
        mid = (low + high) // 2
        if check_recipe_exists(session, mid):
            max_found = mid
            low = mid + 1
        else:
            high = mid - 1
        time.sleep(0.5)
    
    logger.info(f"Maximum recipe ID found: {max_found}")
    return max_found


@retry_on_connection_error()
def download_recipe(session: PoliteSession, recipe_id: int, output_path: pathlib.Path):
    """Download a single recipe and save it to a file."""
    url = f"{BASE_URL}/cocktails/recipe/{recipe_id}/"
    try:
        response = session.get(url)
        if response.status_code == 404:
            logger.debug(f"Recipe {recipe_id} not found (404)")
            return False
        
        response.raise_for_status()
        output_path.write_text(response.text, encoding="utf-8")
        logger.info(f"✓ Downloaded recipe {recipe_id}")
        return True
    except Exception as e:
        logger.error(f"⚠ Failed to download recipe {recipe_id}: {e}")
        return False


def main():
    """Main function to download recipes."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    session = PoliteSession(BASE_URL, USER_AGENT)
    
    # Find the maximum recipe ID
    max_id = find_max_recipe_id(session)
    
    # Track statistics
    downloaded = 0
    skipped = 0
    not_found = 0
    
    # Download all recipes
    for recipe_id in tqdm(range(MIN_RECIPE_ID, max_id + 1), desc="Downloading Recipes"):
        filename = f"recipe_{recipe_id:05d}.html"
        output_path = OUTPUT_DIR / filename
        
        if output_path.exists():
            skipped += 1
            continue
        
        if download_recipe(session, recipe_id, output_path):
            downloaded += 1
        else:
            not_found += 1
        
        # Be polite - add delay between requests
        time.sleep(0.5)
    
    # Print summary
    print(f"\nDownload complete!")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Not found: {not_found}")
    print(f"  Total processed: {downloaded + skipped + not_found}")


if __name__ == "__main__":
    main()