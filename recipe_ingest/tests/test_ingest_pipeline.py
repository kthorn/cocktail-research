"""Tests for the combined ingest pipeline app (validation + mapping + batch tail)."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[2] / "recipe_ingest" / "ingest_pipeline.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ingest_pipeline_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load ingest_pipeline module")
    return importlib.util.module_from_spec(spec), spec


# --- fakes ---------------------------------------------------------------


class FakeAPI:
    """Stand-in for IngredientAPI (no network)."""

    def __init__(self):
        self.items = [
            {"id": 7, "name": "Sweet Vermouth (API)"},
            {"id": 8, "name": "Gin"},
        ]

    def ingredients(self):
        return self.items

    def by_id(self, ingredient_id):
        return next((i for i in self.items if i["id"] == ingredient_id), None)

    def create(self, payload):  # tests patch this
        raise AssertionError("unexpected create() call")

    def search(self, query, limit=20):
        query = query.lower()
        return [
            {"id": i["id"], "name": i["name"], "path": ""}
            for i in self.items
            if query in i["name"].lower()
        ][:limit]

    def suggestions(self, name, **kwargs):
        return [
            {"id": i["id"], "name": i["name"], "path": "", "similarity": 0.9}
            for i in self.items
            if name.lower().split()[0] in i["name"].lower()
        ][:10]


class FakeSource:
    @staticmethod
    def clean_html_content(html):
        return html

    @staticmethod
    def derive_source_url(name):
        return f"http://example.com/{name}"


class FakeEngine:
    """Stand-in for BatchRecipeRationalizer."""

    def __init__(self, recipes, mappings=None):
        self.source_name = "punch"
        self.mappings = dict(mappings or {})
        self.mappings_file = None
        # one html file per recipe
        self.html_files = [f"html/{r['name']}.html" for r in recipes]
        self._recipes = {r["name"]: r for r in recipes}
        self.recipe_source = FakeSource()
        self.validation_log = self._fresh_log()

    @staticmethod
    def _fresh_log():
        return {
            "current_index": 0,
            "accepted": [],
            "rejected": [],
            "auto_ingested": [],
            "needs_review": [],
        }

    def save_validation_log(self):
        pass

    def load_validation_log(self):
        pass

    def load_html(self, html_file):
        return f"<html>{Path(html_file).stem}</html>"

    def parse_recipe_from_html(self, html_file, *args):
        name = Path(html_file).stem
        return self._recipes.get(name)

    def should_skip_recipe(self, parsed):
        if len(parsed.get("ingredients", [])) <= 2:
            return "has 2 or fewer ingredients"
        return None

    def can_rationalize_recipe(self, parsed):
        unmapped = []
        for ing in parsed["ingredients"]:
            name = ing["ingredient_name"].strip()
            if name and name not in self.mappings:
                unmapped.append(name)
        return len(unmapped) == 0, unmapped

    def rationalize_recipe(self, parsed):
        return {
            "name": parsed["name"],
            "description": "",
            "ingredients": [
                {"ingredient_name": i["ingredient_name"], "amount": "1 oz", "unit_name": "oz"}
                for i in parsed["ingredients"]
            ],
            "source_url": "http://example.com",
        }


def two_ingredient_recipe():
    return {"name": "Skipped", "ingredients": [{"ingredient_name": "gin"}, {"ingredient_name": "tonic"}]}


def normal_recipe():
    return {
        "name": "Manhattan",
        "ingredients": [
            {"ingredient_name": "Rye Whiskey"},
            {"ingredient_name": "Sweet Vermouth"},
            {"ingredient_name": "Bitters"},
        ],
    }


def make_pipeline(tmp_path, recipes=None, mappings=None):
    mappings = {
        "Rye Whiskey": {"id": 1, "name": "Rye Whiskey"},
        "Bitters": {"id": 3, "name": "Bitters"},
    } if mappings is None else mappings
    module, spec = load_module()
    spec.loader.exec_module(module)
    engine = FakeEngine(recipes or [normal_recipe(), two_ingredient_recipe()], mappings)
    mappings_file = tmp_path / "ingredient_mappings.json"
    mappings_file.write_text(json.dumps(mappings))
    pipeline = module.RecipePipeline(
        engine=engine,
        api=FakeAPI(),
        mappings_file=str(mappings_file),
    )
    app = module.create_app(pipeline)
    return module, pipeline, engine, app.test_client()


# --- tests ---------------------------------------------------------------


def test_pipeline_surfaces_recipe_with_unmapped_highlight(tmp_path):
    module, pipeline, engine, client = make_pipeline(tmp_path)
    state = client.get("/api/state").get_json()
    assert state["phase"] == "recipe"
    assert state["recipe"]["name"] == "Manhattan"
    assert state["recipe"]["unmapped"] == ["Sweet Vermouth"]


def test_accept_with_unmapped_moves_to_mapping(tmp_path):
    module, pipeline, engine, client = make_pipeline(tmp_path)
    state = client.get("/api/state").get_json()
    client.post("/api/accept", json={})
    state = client.get("/api/state").get_json()
    assert state["phase"] == "mapping"
    assert state["ingredient"]["name"] == "Sweet Vermouth"


def test_map_existing_saves_mapping_and_completes_recipe(tmp_path):
    module, pipeline, engine, client = make_pipeline(tmp_path)
    client.get("/api/state")
    client.post("/api/accept", json={})
    resp = client.post(
        "/api/map",
        json={"type": "existing", "ingredient_id": 7},
    ).get_json()
    assert resp["success"]

    saved = json.loads(Path(pipeline.mappings_file).read_text())
    assert saved["Sweet Vermouth"]["id"] == 7
    assert saved["Sweet Vermouth"]["name"] == "Sweet Vermouth (API)"

    state = client.get("/api/state").get_json()
    assert state["phase"] == "done"  # Manhattan accepted, small recipe was skipped
    assert "Manhattan" in engine.validation_log["accepted"]
    assert state["stats"]["accepted"] == 1


def test_map_new_creates_api_ingredient_and_maps_it(tmp_path):
    module, pipeline, engine, client = make_pipeline(tmp_path)
    client.get("/api/state")
    client.post("/api/accept", json={})

    created = {}
    with patch.object(pipeline.api, "create", side_effect=lambda payload: created.update(payload) or {"id": 42, "name": "Pineapple Gum"}):
        resp = client.post(
            "/api/map",
            json={"type": "new", "name": "Pineapple Gum", "percent_abv": 0},
        ).get_json()

    assert resp["success"]
    assert created["name"] == "Pineapple Gum"
    saved = json.loads(Path(pipeline.mappings_file).read_text())
    assert saved["Sweet Vermouth"]["id"] == 42
    assert saved["Sweet Vermouth"]["name"] == "Pineapple Gum"


def test_map_new_api_failure_keeps_ingredient_current(tmp_path):
    module, pipeline, engine, client = make_pipeline(tmp_path)
    client.get("/api/state")
    client.post("/api/accept", json={})

    with patch.object(pipeline.api, "create", side_effect=module.APIError("401 Unauthorized")):
        resp = client.post("/api/map", json={"type": "new", "name": "X"}).get_json()

    assert resp["success"] is False
    assert "401" in resp["error"]
    state = client.get("/api/state").get_json()
    assert state["phase"] == "mapping"
    assert state["ingredient"]["name"] == "Sweet Vermouth"


def test_map_new_without_credentials_is_rejected(tmp_path):
    module, pipeline, engine, client = make_pipeline(tmp_path)
    client.get("/api/state")
    client.post("/api/accept", json={})

    with patch.object(pipeline.api, "create", side_effect=module.NoCredentials("set MIXOLOGY_USERNAME")):
        resp = client.post("/api/map", json={"type": "new", "name": "X"}).get_json()

    assert resp["success"] is False
    assert "MIXOLOGY_USERNAME" in resp["error"]


def test_map_existing_unknown_id_rejected(tmp_path):
    module, pipeline, engine, client = make_pipeline(tmp_path)
    client.get("/api/state")
    client.post("/api/accept", json={})
    resp = client.post("/api/map", json={"type": "existing", "ingredient_id": 999}).get_json()
    assert resp["success"] is False
    state = client.get("/api/state").get_json()
    assert state["phase"] == "mapping"


def test_skip_ingredient_defers_recipe_to_needs_review(tmp_path):
    module, pipeline, engine, client = make_pipeline(tmp_path)
    client.get("/api/state")
    client.post("/api/accept", json={})
    client.post("/api/map", json={"type": "skip"}).get_json()

    review = engine.validation_log["needs_review"]
    assert review == [{"recipe_name": "Manhattan", "unmapped_ingredients": ["Sweet Vermouth"]}]
    state = client.get("/api/state").get_json()
    assert state["phase"] == "done"


def test_accept_edited_recipe_is_recorded(tmp_path):
    module, pipeline, engine, client = make_pipeline(
        tmp_path,
        mappings={
            "Rye Whiskey": {"id": 1, "name": "Rye"},
            "Sweet Vermouth": {"id": 2, "name": "Sweet Vermouth"},
            "Bitters": {"id": 3, "name": "Bitters (Angostura)"},
        },
    )
    edited = {
        "name": "Manhattan",
        "ingredients": [{"ingredient_name": "Rye Whiskey", "amount": "2 oz", "unit_name": "oz"}],
    }
    resp = client.post("/api/accept", json={"recipe_data": edited, "edited": True}).get_json()
    assert resp["success"]
    assert "Manhattan" in engine.validation_log["accepted"]
    assert engine.validation_log["edited_recipes"]["Manhattan"] == edited


def test_skipped_small_recipes_do_not_block(tmp_path):
    module, pipeline, engine, client = make_pipeline(
        tmp_path, recipes=[two_ingredient_recipe(), normal_recipe()]
    )
    # small recipe never surfaces; the normal one is the current candidate
    state = client.get("/api/state").get_json()
    assert state["recipe"]["name"] == "Manhattan"

    client.post("/api/reject", json={})
    assert "Manhattan" in engine.validation_log["rejected"]
    state = client.get("/api/state").get_json()
    assert state["phase"] == "done"


def test_batch_tail_uses_the_batch_script(tmp_path):
    module, pipeline, engine, client = make_pipeline(tmp_path)

    calls = {}

    class FakeBatcher:
        def __init__(self, **kwargs):
            calls.update(kwargs)
            calls["source"] = kwargs["source_name"]

        def run(self):
            return {"auto_ingested": 3, "needs_review": 1, "batch_files": ["out.json"]}

    pipeline.batch_factory = FakeBatcher
    resp = client.post("/api/batch", json={}).get_json()
    assert resp["success"] is True
    assert resp["summary"] == {"auto_ingested": 3, "needs_review": 1, "batch_files": ["out.json"]}
    assert calls["mappings_file"] == pipeline.mappings_file
    assert calls["validation_log_file"] is not None
    assert calls["source_name"] == "punch"


def test_suggestions_endpoint(tmp_path):
    module, pipeline, engine, client = make_pipeline(tmp_path)
    resp = client.get("/api/search?q=sweet").get_json()
    assert any("Sweet" in r["name"] for r in resp)


def test_page_renders(tmp_path):
    module, pipeline, engine, client = make_pipeline(tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200
