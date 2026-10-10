#!/usr/bin/env python3
"""
End-to-end recipe ingest pipeline.

One entry point that replaces the old validation and ingredient-mapping web
apps: for each unprocessed recipe it shows the original HTML alongside the
rationalized preview, records accept/reject, and - when a recipe has unmapped
ingredients - drops you straight into the ingredient-mapping screen for those.
New ingredients can be created in the mixology.tools database directly, so
future recipes map to exactly what is in the database. When the queue is empty
(or on demand) it runs the batch auto-ingest pass, producing upload-ready batch
files for every recipe that is now fully mapped.

Usage:
    export MIXOLOGY_USERNAME=...   # optional: enables "create new ingredient"
    export MIXOLOGY_PASSWORD=...
    python ingest_pipeline.py --source punch
    python ingest_pipeline.py --source diffords [--port 5003]
"""

import json
import os
import sqlite3
import sys
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, "/home/kurtt/cocktail-research/cocktail-utils/src")
sys.path.insert(0, "/home/kurtt/cocktail-research/recipe_ingest")

import cognito_token
from batch_rationalize_recipes import BatchRecipeRationalizer

REPO = "/home/kurtt/cocktail-research"
MAPPINGS_FILE = f"{REPO}/data/ingredient_mappings.json"
VALIDATION_LOG_FILE = f"{REPO}/data/validation_log.json"
OUTPUT_DIR = f"{REPO}/output"
DATABASE_FILE = f"{REPO}/data/recipes.db"
API_BASE = "https://mixology.tools"


def normalize_string(text: str) -> str:
    """Normalize quotes/apostrophes and strip 'fresh' so name variants collide."""
    for char in ("\u2019", "\u2018", "\u02bc", "`"):
        text = text.replace(char, "'")
    for char in ("\u201c", "\u201d", "\u201e"):
        text = text.replace(char, '"')
    for pattern in (", fresh", " (fresh)", "(fresh)"):
        text = text.replace(pattern, "")
    if text.startswith("fresh "):
        text = text[6:]
    return " ".join(text.split())


class APIError(Exception):
    """Raised when the ingredient API call fails."""


class NoCredentials(APIError):
    """Raised when ingredient creation is requested without credentials."""


class IngredientAPI:
    """Read access to mixology.tools ingredients (public) plus editor creation."""

    def __init__(self, base: str = API_BASE, get_token=None):
        self.base = base
        self._get_token = get_token or cognito_token.get_token_from_env
        self._cache: Optional[list] = None

    def ingredients(self) -> list:
        if self._cache is None:
            import requests

            response = requests.get(f"{self.base}/api/ingredients", timeout=30)
            response.raise_for_status()
            self._cache = response.json()
        return self._cache

    def by_id(self, ingredient_id: int) -> Optional[dict]:
        return next((i for i in self.ingredients() if i["id"] == ingredient_id), None)

    def search(self, query: str, limit: int = 20) -> list:
        query = normalize_string(query.lower())
        return [
            {"id": i["id"], "name": i["name"], "path": i.get("path", "")}
            for i in self.ingredients()
            if query in normalize_string(i["name"].lower())
        ][:limit]

    def suggestions(self, name: str, threshold: float = 0.6, limit: int = 10) -> list:
        lookup = normalize_string(name.lower())
        matches = []
        for ing in self.ingredients():
            api_name = normalize_string(ing["name"].lower())
            similarity = SequenceMatcher(None, lookup, api_name).ratio()
            contains = lookup in api_name or api_name in lookup
            if api_name == lookup:
                matches.append({**ing, "similarity": 1.0, "exact_match": True})
            elif similarity >= threshold or contains:
                matches.append(
                    {**ing, "similarity": similarity, "exact_match": False, "contains_match": contains}
                )
        matches.sort(key=lambda x: (x.get("exact_match", False), x["similarity"]), reverse=True)
        return [
            {"id": i["id"], "name": i["name"], "path": i.get("path", ""), "similarity": round(i["similarity"], 2)}
            for i in matches[:limit]
        ]

    def create(self, payload: dict) -> dict:
        """Create an ingredient via the editor API. Returns the created record."""
        import requests

        try:
            token = self._get_token()
        except cognito_token.AuthError as exc:
            raise NoCredentials(str(exc)) from exc

        body = {
            key: value
            for key, value in payload.items()
            if key in ("name", "description", "parent_id", "percent_abv", "sugar_g_per_l",
                       "titratable_acidity_g_per_l", "allow_substitution")
            and value not in (None, "")
        }
        response = requests.post(
            f"{self.base}/api/v1/ingredients",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        if response.status_code == 401:
            # Token may have expired mid-session: authenticate once more and retry.
            response = requests.post(
                f"{self.base}/api/v1/ingredients",
                json=body,
                headers={"Authorization": f"Bearer {self._get_token()}"},
                timeout=30,
            )
        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise APIError(f"Ingredient creation failed ({response.status_code}): {detail}")
        return response.json()


class RecipePipeline:
    def __init__(
        self,
        engine: BatchRecipeRationalizer,
        api: IngredientAPI = None,
        mappings_file: str = MAPPINGS_FILE,
    ):
        self.engine = engine
        self.api = api or IngredientAPI()
        self.mappings_file = mappings_file
        self.batch_factory = None  # injectable for tests; default: BatchRecipeRationalizer

        with open(mappings_file, encoding="utf-8") as f:
            content = f.read().strip()
        self.engine.mappings = json.loads(content) if content else {}

        # DB names, for noting when a recipe already lives in the database
        self.db_names = self._load_database_names()
        self.staged: Optional[dict] = None
        print("Parsing pending recipes to find which still need review...")
        self._candidates = self._load_candidates()
        self._variation_index = self._build_variation_index()
        print(f"{len(self._candidates)} recipe(s) still need review")

    # -- state ------------------------------------------------------------

    def _load_candidates(self):
        candidates = {}
        for html_file in self.engine.html_files:
            parsed = self.engine.parse_recipe_from_html(html_file)
            if parsed and not self._is_processed(parsed["name"]) and not self.engine.should_skip_recipe(parsed):
                candidates[html_file] = parsed
        return candidates

    def _load_database_names(self):
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            rows = conn.execute("SELECT name FROM recipe").fetchall()
            conn.close()
            return {name for (name,) in rows}
        except Exception as exc:
            print(f"Warning: could not read recipe database ({exc})")
            return []

    def _is_processed(self, recipe_name: str) -> bool:
        log = self.engine.validation_log
        return (
            recipe_name in log.get("accepted", [])
            or recipe_name in log.get("rejected", [])
            or recipe_name in log.get("auto_ingested", [])
        )

    def _unmapped_ingredients(self, recipe_data: dict) -> List[str]:
        unmapped = []
        for ing in recipe_data.get("ingredients", []):
            name = ing.get("ingredient_name", "").strip()
            if name and name not in self.engine.mappings:
                unmapped.append(name)
        return unmapped

    def _rationalize(self, parsed_recipe: dict) -> dict:
        return self.engine.rationalize_recipe(parsed_recipe)

    def state(self) -> dict:
        log = self.engine.validation_log
        if self.staged is not None:
            name = self.staged["remaining"][0]
            return {
                "phase": "mapping",
                "ingredient": {
                    "name": name,
                    "variations": self._variations_for(name),
                    "suggestions": self.api.suggestions(name),
                    "recipes_using": [self.staged["name"]],
                    "can_create": self._can_create(),
                },
                "staged_recipe": self.staged["name"],
                "stats": self._stats(),
            }

        html_file = next(iter(self._candidates), None)
        if html_file is None:
            return {"phase": "done", "stats": self._stats(), "source": self.engine.source_name}

        parsed = self._candidates[html_file]
        rationalized = self._rationalize(parsed)
        return {
            "phase": "recipe",
            "source": self.engine.source_name,
            "recipe": {
                "name": parsed["name"],
                "html_file": html_file,
                "html_content": self.engine.recipe_source.clean_html_content(
                    self.engine.load_html(html_file)
                ),
                "rationalized_data": rationalized,
                "unmapped": self._unmapped_ingredients(rationalized),
                "already_in_db": parsed["name"] in self.db_names,
            },
            "stats": self._stats(),
        }

    def _can_create(self) -> bool:
        return bool(
            os.environ.get("MIXOLOGY_USERNAME") and os.environ.get("MIXOLOGY_PASSWORD")
        )

    def _stats(self) -> dict:
        log = self.engine.validation_log
        return {
            "source": self.engine.source_name,
            "remaining": len(self._candidates),
            "accepted": len(log.get("accepted", [])),
            "rejected": len(log.get("rejected", [])),
            "needs_review": len(log.get("needs_review", [])),
            "mappings": len(self.engine.mappings),
            "edited": len(log.get("edited_recipes", {})),
        }

    # -- actions -----------------------------------------------------------

    def reject(self) -> dict:
        html_file = next(iter(self._candidates), None)
        if html_file is None:
            return {"success": False, "error": "No recipe pending"}
        name = self._candidates.pop(html_file)["name"]
        self.engine.validation_log["rejected"].append(name)
        self.engine.save_validation_log()
        return {"success": True}

    def accept(self, recipe_data: Optional[dict] = None, edited: bool = False) -> dict:
        html_file = next(iter(self._candidates), None)
        if html_file is None:
            return {"success": False, "error": "No recipe pending"}
        parsed = self._candidates.pop(html_file)

        data = recipe_data if recipe_data else self._rationalize(parsed)
        if not isinstance(data, dict) or "name" not in data or not isinstance(data.get("ingredients"), list):
            return {"success": False, "error": "Recipe must be a JSON object with name and ingredients"}

        unmapped = self._unmapped_ingredients(data)
        if unmapped:
            self.staged = {"name": data["name"], "data": data, "remaining": unmapped}
            return {"success": True, "phase": "mapping"}

        self._record_accepted(data["name"], data if edited else None)
        return {"success": True, "phase": "recipe"}

    def _record_accepted(self, name: str, edited_data: Optional[dict]):
        log = self.engine.validation_log
        log.setdefault("accepted", []).append(name)
        if edited_data is not None:
            log.setdefault("edited_recipes", {})[name] = edited_data
        self.engine.save_validation_log()

    def map_ingredient(self, action: dict) -> dict:
        if self.staged is None:
            return {"success": False, "error": "No ingredient pending"}

        name = self.staged["remaining"][0]
        map_type = action.get("type")

        if map_type == "existing":
            api_ing = self.api.by_id(int(action["ingredient_id"]))
            if not api_ing:
                return {"success": False, "error": f"Unknown ingredient id {action['ingredient_id']}"}
            self._write_mapping(name, {"id": api_ing["id"], "name": api_ing["name"]})
        elif map_type == "new":
            try:
                created = self.api.create(action)
            except APIError as exc:
                return {"success": False, "error": str(exc)}
            self._write_mapping(name, {"id": created["id"], "name": created["name"]})
        elif map_type == "skip":
            self.engine.validation_log.setdefault("needs_review", []).append(
                {"recipe_name": self.staged["name"], "unmapped_ingredients": list(self.staged["remaining"])}
            )
            self.engine.save_validation_log()
            self.staged = None
            return {"success": True, "phase": "recipe"}
        else:
            return {"success": False, "error": f"Unknown mapping type: {map_type}"}

        self.staged["remaining"] = self._unmapped_ingredients(self.staged["data"])
        if not self.staged["remaining"]:
            self._record_accepted(self.staged["name"], None)
            self.staged = None
        return {"success": True, "phase": "mapping" if self.staged else "recipe"}

    @staticmethod
    def _creation_fields(action: dict) -> dict:
        allowed = ("name", "description", "parent_id", "percent_abv", "sugar_g_per_l",
                   "titratable_acidity_g_per_l", "allow_substitution")
        return {key: action[key] for key in allowed if key in action and action[key] not in (None, "")}

    def _write_mapping(self, raw_name: str, entry: dict):
        entry = dict(entry, mapped_at=datetime.now().isoformat())
        for variant in self._variations_for(raw_name):
            self.engine.mappings[variant] = dict(entry)
        with open(self.mappings_file, "w", encoding="utf-8") as f:
            json.dump(self.engine.mappings, f, indent=2)

    def _build_variation_index(self) -> Dict[str, List[str]]:
        """Map normalized ingredient keys to every raw variant seen in pending recipes."""
        index: Dict[str, List[str]] = {}
        for parsed in self._candidates.values():
            for ing in parsed.get("ingredients", []):
                raw = ing.get("ingredient_name", "").strip()
                if not raw:
                    continue
                key = normalize_string(raw.lower())
                variants = index.setdefault(key, [])
                if raw not in variants:
                    variants.append(raw)
        return index

    def _variations_for(self, name: str) -> List[str]:
        """All raw names in the pending corpus that normalize to the same key."""
        return self._variation_index.get(normalize_string(name.lower()), [name])

    def run_batch(self) -> dict:
        """Run the batch auto-ingest pass over every fully-mapped unprocessed recipe."""
        factory = self.batch_factory or BatchRecipeRationalizer
        batcher = factory(
            source_name=self.engine.source_name,
            mappings_file=self.mappings_file,
            validation_log_file=VALIDATION_LOG_FILE,
            output_dir=OUTPUT_DIR,
        )
        summary = batcher.run()
        return {"success": True, "summary": summary}


# -- Flask app ---------------------------------------------------------------


def create_app(pipeline: RecipePipeline) -> Flask:
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    app = Flask(__name__, template_folder=template_dir)

    @app.route("/")
    def index():
        return render_template("pipeline.html", source=pipeline.engine.source_name)

    @app.route("/api/state")
    def state():
        return jsonify(pipeline.state())

    @app.route("/api/accept", methods=["POST"])
    def accept():
        data = request.get_json(silent=True) or {}
        return jsonify(
            pipeline.accept(recipe_data=data.get("recipe_data"), edited=bool(data.get("edited")))
        )

    @app.route("/api/reject", methods=["POST"])
    def reject():
        return jsonify(pipeline.reject())

    @app.route("/api/map", methods=["POST"])
    def map_ingredient():
        return jsonify(pipeline.map_ingredient(request.get_json(silent=True) or {}))

    @app.route("/api/search")
    def search():
        return jsonify(pipeline.api.search(request.args.get("q", "")))

    @app.route("/api/batch", methods=["POST"])
    def batch():
        return jsonify(pipeline.run_batch())

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description="End-to-end recipe ingest pipeline")
    parser.add_argument("--source", required=True, choices=["punch", "diffords"])
    parser.add_argument("--port", type=int, default=5003)
    args = parser.parse_args()

    engine = BatchRecipeRationalizer(
        source_name=args.source,
        mappings_file=MAPPINGS_FILE,
        validation_log_file=VALIDATION_LOG_FILE,
        output_dir=OUTPUT_DIR,
    )
    pipeline = RecipePipeline(engine=engine, mappings_file=MAPPINGS_FILE)

    print(f"Ingest pipeline - {args.source.upper()}")
    print(f"HTML files: {len(engine.html_files)}, mappings: {len(engine.mappings)}")
    stats = pipeline._stats()
    print(f"Remaining to review: {stats['remaining']} (accepted {stats['accepted']}, rejected {stats['rejected']})")

    create_app(pipeline).run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
