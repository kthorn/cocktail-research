from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Dict, Any, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        # Keep non-ascii escaped to match existing style
        json.dump(data, f, indent=2, ensure_ascii=True, sort_keys=True)
        f.write("\n")


def merge_dicts(
    primary: Dict[str, Any],
    secondary: Dict[str, Any],
    prefer_primary: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Merge two mapping dicts of ingredient_name -> metadata.

    Returns merged dict and a dict of conflicts (key -> {primary, secondary}).
    """
    merged: Dict[str, Any] = {}
    conflicts: Dict[str, Any] = {}

    # Start from the secondary and overlay primary if prefer_primary
    if prefer_primary:
        merged.update(secondary)
        for k, v in primary.items():
            if k in merged and merged[k] != v:
                conflicts[k] = {"primary": v, "secondary": merged[k]}
            merged[k] = v
    else:
        merged.update(primary)
        for k, v in secondary.items():
            if k in merged and merged[k] != v:
                conflicts[k] = {"primary": merged[k], "secondary": v}
            merged[k] = v

    return merged, conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ingredient mapping JSON files")
    parser.add_argument(
        "--primary", type=Path, required=True, help="Primary/root JSON path"
    )
    parser.add_argument(
        "--secondary", type=Path, required=True, help="Secondary JSON path to merge in"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (defaults to --primary in-place)",
    )
    parser.add_argument(
        "--prefer",
        choices=["primary", "secondary"],
        default="primary",
        help="Which file to prefer on key conflicts",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="If set and writing in-place, write a timestamped backup of original output",
    )

    args = parser.parse_args()

    out_path: Path = args.output or args.primary
    prefer_primary: bool = args.prefer == "primary"

    primary_data = load_json(args.primary)
    secondary_data = load_json(args.secondary)

    merged, conflicts = merge_dicts(
        primary_data, secondary_data, prefer_primary=prefer_primary
    )

    # Backup if requested and overwriting the primary
    if args.backup and out_path.exists():
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = out_path.with_suffix("")
        backup_path = Path(f"{backup_path}.{ts}.bak.json")
        backup_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

    write_json(out_path, merged)

    # Summary
    print(
        json.dumps(
            {
                "primary_path": str(args.primary),
                "secondary_path": str(args.secondary),
                "output_path": str(out_path),
                "prefer": "primary" if prefer_primary else "secondary",
                "counts": {
                    "primary_keys": len(primary_data),
                    "secondary_keys": len(secondary_data),
                    "merged_keys": len(merged),
                    "conflicts": len(conflicts),
                    "added_from_secondary": len(
                        [k for k in secondary_data.keys() if k not in primary_data]
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
