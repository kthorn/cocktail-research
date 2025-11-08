#!/usr/bin/env python3
"""
One-time migration script to convert validation_log.json to source-aware format.

Old format:
{
  "current_index": 0,
  "reviewed": [],
  "accepted": ["Recipe 1", "Recipe 2"],
  "rejected": ["Recipe 3"],
  "auto_ingested": ["Recipe 4"],
  "needs_review": []
}

New format:
{
  "punch": {
    "current_index": 0,
    "accepted": ["Recipe 1", "Recipe 2"],
    "rejected": ["Recipe 3"],
    "auto_ingested": ["Recipe 4"],
    "needs_review": []
  },
  "diffords": {
    "current_index": 0,
    "accepted": [],
    "rejected": [],
    "auto_ingested": [],
    "needs_review": []
  }
}
"""

import json
import os
import shutil
from datetime import datetime


def migrate_validation_log(log_file: str) -> None:
    """Migrate validation_log.json to source-aware format.

    Args:
        log_file: Path to validation_log.json
    """
    if not os.path.exists(log_file):
        print(f"❌ Validation log not found: {log_file}")
        return

    # Load existing log
    with open(log_file, "r") as f:
        old_log = json.load(f)

    # Check if already migrated
    if "punch" in old_log or "diffords" in old_log:
        print("✅ Validation log already in new format - no migration needed")
        return

    # Create backup
    backup_file = f"{log_file}.backup-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(log_file, backup_file)
    print(f"📦 Created backup: {backup_file}")

    # Migrate to new format
    new_log = {
        "punch": {
            "current_index": old_log.get("current_index", 0),
            "accepted": old_log.get("accepted", []),
            "rejected": old_log.get("rejected", []),
            "auto_ingested": old_log.get("auto_ingested", []),
            "needs_review": old_log.get("needs_review", []),
        },
        "diffords": {
            "current_index": 0,
            "accepted": [],
            "rejected": [],
            "auto_ingested": [],
            "needs_review": [],
        }
    }

    # Write new format
    with open(log_file, "w") as f:
        json.dump(new_log, f, indent=2)

    print("✅ Migration complete!")
    print(f"   Punch recipes migrated:")
    print(f"     - Accepted: {len(new_log['punch']['accepted'])}")
    print(f"     - Rejected: {len(new_log['punch']['rejected'])}")
    print(f"     - Auto-ingested: {len(new_log['punch']['auto_ingested'])}")
    print(f"     - Needs review: {len(new_log['punch']['needs_review'])}")
    print(f"   Diffords initialized with empty progress")


def main():
    """Run migration."""
    log_file = "/home/kurtt/cocktail-research/data/validation_log.json"

    print("=" * 70)
    print("Validation Log Migration Script")
    print("=" * 70)
    print()

    migrate_validation_log(log_file)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
