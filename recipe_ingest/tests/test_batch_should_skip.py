"""Tests for shared recipe skip rules used by the pipeline and the batch script."""

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "batch_rationalize_recipes.py"


def load_module():
    spec = importlib.util.spec_from_file_location("batch_rationalize_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load batch_rationalize_recipes module")
    return importlib.util.module_from_spec(spec), spec


def make_rationalizer():
    module, spec = load_module()
    spec.loader.exec_module(module)
    rationalizer = module.BatchRecipeRationalizer.__new__(module.BatchRecipeRationalizer)
    rationalizer.source_name = "test"
    return module, rationalizer


def parsed(name="Daiquiri", ingredients=3, instructions="Shake. Serve."):
    return {
        "name": name,
        "ingredients": [{"ingredient_name": f"ing{i}"} for i in range(ingredients)],
        "instructions": instructions,
    }


def test_skip_shot_glass_in_instructions():
    _, rationalizer = make_rationalizer()
    assert rationalizer.should_skip_recipe(
        parsed(instructions="Pour into a SHOT GLASS")
    ) is not None


def test_skip_shot_in_name():
    _, rationalizer = make_rationalizer()
    assert rationalizer.should_skip_recipe(parsed(name="Kamikaze Shot")) is not None


def test_skip_two_or_fewer_ingredients():
    _, rationalizer = make_rationalizer()
    assert rationalizer.should_skip_recipe(parsed(ingredients=2)) is not None


def test_normal_recipe_not_skipped():
    _, rationalizer = make_rationalizer()
    assert rationalizer.should_skip_recipe(parsed()) is None
