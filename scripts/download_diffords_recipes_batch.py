#!/usr/bin/env python3
"""
Downloads recipe pages from diffordsguide.com with batch processing and resume capability.
"""

import json
import logging
import pathlib
import time
from datetime import datetime
from typing import Optional

from tqdm.auto import tqdm

from cocktail_utils.scraping import PoliteSession, retry_on_connection_error

# Config
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.diffordsguide.com"
OUTPUT_DIR = pathlib.Path("/home/kurtt/cocktail-research/raw_recipes/diffords_html")
USER_AGENT = "cocktail-bot/1.0 (+mailto:kurt.thorn@gmail.com)"
STATE_FILE = OUTPUT_DIR / "download_state.json"

# Download config
BATCH_SIZE = 100  # Process in batches for progress tracking
MAX_WORKERS = 1  # Number of parallel downloads (keep low to be polite)
DELAY_BETWEEN_REQUESTS = 1.0  # Seconds between requests
MAX_CONSECUTIVE_404S = 50  # Stop after this many consecutive 404s


class DownloadState:
    """Manages download state for resume capability."""

    def __init__(self, state_file: pathlib.Path):
        self.state_file = state_file
        self.state = self.load_state()

    def load_state(self) -> dict:
        """Load state from file or create new."""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {
            "last_checked_id": 0,
            "max_id_found": None,
            "downloaded_count": 0,
            "not_found_ids": [],
            "last_updated": None,
        }

    def save_state(self):
        """Save current state to file."""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def update_progress(self, recipe_id: int, found: bool):
        """Update progress for a recipe."""
        self.state["last_checked_id"] = max(self.state["last_checked_id"], recipe_id)
        if found:
            self.state["downloaded_count"] += 1
            if (
                self.state["max_id_found"] is None
                or recipe_id > self.state["max_id_found"]
            ):
                self.state["max_id_found"] = recipe_id
        else:
            self.state["not_found_ids"].append(recipe_id)


@retry_on_connection_error(max_retries=3)
def download_recipe(
    session: PoliteSession, recipe_id: int
) -> tuple[int, bool, Optional[str]]:
    """
    Download a single recipe.
    Returns: (recipe_id, success, html_content)
    """
    url = f"{BASE_URL}/cocktails/recipe/{recipe_id}/"

    try:
        response = session.get(url, timeout=30)

        if response.status_code == 404:
            return (recipe_id, False, None)

        response.raise_for_status()
        return (recipe_id, True, response.text)

    except Exception as e:
        logger.error(f"Error downloading recipe {recipe_id}: {e}")
        return (recipe_id, False, None)


def scan_for_max_id(session: PoliteSession, start: int = 1, end: int = 100000) -> int:
    """
    Scan to find approximate maximum recipe ID using exponential search then binary search.
    """
    logger.info("Scanning for maximum recipe ID...")

    # First, use exponential search to find the upper bound
    # Start with known checkpoints and double until we find a non-existent recipe
    test_points = [
        1000,
        5000,
        10000,
        15000,
        20000,
        25000,
        30000,
        35000,
        40000,
        50000,
        60000,
        80000,
        100000,
    ]

    logger.info("Finding approximate upper bound...")
    upper_bound = 0
    lower_bound = 0

    for test_id in test_points:
        _, found, _ = download_recipe(session, test_id)
        if found:
            lower_bound = test_id
            logger.info(f"Recipe {test_id} exists - continuing search")
        else:
            upper_bound = test_id
            logger.info(f"Recipe {test_id} not found - found upper bound")
            break
        time.sleep(0.5)

    # If we didn't find an upper bound, keep doubling
    if upper_bound == 0:
        current = test_points[-1]
        while current < 1000000:  # Safety limit at 1 million
            current = current * 2
            _, found, _ = download_recipe(session, current)
            if not found:
                upper_bound = current
                logger.info(f"Recipe {current} not found - found upper bound")
                break
            else:
                lower_bound = current
                logger.info(f"Recipe {current} exists - continuing search")
            time.sleep(0.5)

    if upper_bound == 0:
        logger.error("Could not find upper bound even at 1 million!")
        upper_bound = 1000000

    logger.info(f"Search range: {lower_bound} to {upper_bound}")

    # Now binary search to find the actual maximum
    # We'll check clusters of IDs to handle gaps
    low = lower_bound
    high = upper_bound
    max_found = 0

    logger.info("Binary searching for exact maximum...")

    while high - low > 1000:  # Stop when range is narrow enough
        mid = (low + high) // 2

        # Check a range around mid to handle gaps (recipes might not be continuous)
        found_in_range = False
        check_range = 200  # Check +/- 200 IDs around midpoint

        for check_id in range(
            max(1, mid - check_range), min(upper_bound, mid + check_range + 1), 50
        ):
            _, found, _ = download_recipe(session, check_id)
            if found:
                found_in_range = True
                max_found = max(max_found, check_id)
                logger.info(f"Recipe {check_id} exists")
                break
            time.sleep(0.2)

        if found_in_range:
            low = mid  # Recipes exist around mid, search higher
        else:
            high = mid  # No recipes around mid, search lower

        logger.info(f"Narrowing range: {low} to {high}")

    # Final sweep of the narrow range to find the absolute maximum
    logger.info(f"Final sweep of range {low} to {high}")

    for check_id in range(high, low - 1, -100):  # Check every 100 IDs going backwards
        _, found, _ = download_recipe(session, check_id)
        if found:
            max_found = max(max_found, check_id)
            logger.info(f"Recipe {check_id} exists - potential maximum")
            # Check a bit higher to be sure
            for extra_id in range(check_id + 1, min(check_id + 500, high + 1), 20):
                _, found2, _ = download_recipe(session, extra_id)
                if found2:
                    max_found = max(max_found, extra_id)
                    logger.info(f"Recipe {extra_id} also exists")
                time.sleep(0.2)
            break
        time.sleep(0.3)

    logger.info(f"Maximum recipe ID found: {max_found}")
    return max_found + 100  # Small buffer for very recent additions


def download_batch(
    session: PoliteSession, recipe_ids: list[int], output_dir: pathlib.Path
) -> dict:
    """Download a batch of recipes."""
    results = {"downloaded": 0, "skipped": 0, "not_found": 0, "errors": 0}

    for recipe_id in recipe_ids:
        filename = f"recipe_{recipe_id:05d}.html"
        output_path = output_dir / filename

        # Skip if already downloaded
        if output_path.exists():
            results["skipped"] += 1
            continue

        # Download recipe
        _, found, content = download_recipe(session, recipe_id)

        if found and content:
            output_path.write_text(content, encoding="utf-8")
            results["downloaded"] += 1
        elif not found:
            results["not_found"] += 1
        else:
            results["errors"] += 1

        # Be polite
        time.sleep(DELAY_BETWEEN_REQUESTS)

    return results


def main():
    """Main function to download recipes."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    session = PoliteSession(BASE_URL, USER_AGENT)
    state = DownloadState(STATE_FILE)

    # Determine range to download
    start_id = (
        state.state["last_checked_id"] + 1 if state.state["last_checked_id"] > 0 else 1
    )

    if state.state["max_id_found"]:
        max_id = state.state["max_id_found"] + 100  # Add buffer
        logger.info(
            f"Resuming from recipe {start_id}, max known ID: {state.state['max_id_found']}"
        )
    else:
        max_id = scan_for_max_id(session, start=1, end=100000)
        state.state["max_id_found"] = max_id
        state.save_state()

    # Download in batches
    total_stats = {"downloaded": 0, "skipped": 0, "not_found": 0, "errors": 0}

    consecutive_not_found = 0

    for batch_start in tqdm(
        range(start_id, max_id + 1, BATCH_SIZE), desc="Downloading batches"
    ):
        batch_end = min(batch_start + BATCH_SIZE, max_id + 1)
        batch_ids = list(range(batch_start, batch_end))

        logger.info(f"Processing batch {batch_start}-{batch_end - 1}")

        results = download_batch(session, batch_ids, OUTPUT_DIR)

        # Update totals
        for key in total_stats:
            total_stats[key] += results[key]

        # Update state
        state.state["last_checked_id"] = batch_end - 1
        state.state["downloaded_count"] = total_stats["downloaded"]
        state.save_state()

        # Check for consecutive not found
        if results["not_found"] == len(batch_ids):
            consecutive_not_found += results["not_found"]
            if consecutive_not_found >= MAX_CONSECUTIVE_404S:
                logger.info(
                    f"Stopping - {consecutive_not_found} consecutive recipes not found"
                )
                break
        else:
            consecutive_not_found = 0

        # Print batch summary
        logger.info(
            f"Batch complete - Downloaded: {results['downloaded']}, "
            f"Skipped: {results['skipped']}, Not found: {results['not_found']}"
        )

    # Print final summary
    print(f"\n{'=' * 50}")
    print(f"Download complete!")
    print(f"  Downloaded: {total_stats['downloaded']}")
    print(f"  Skipped (already exists): {total_stats['skipped']}")
    print(f"  Not found: {total_stats['not_found']}")
    print(f"  Errors: {total_stats['errors']}")
    print(f"  Total processed: {sum(total_stats.values())}")
    print(f"  Last checked ID: {state.state['last_checked_id']}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
