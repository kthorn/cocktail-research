# Multi-Source Recipe Pipeline Refactor

## Summary

Refactored the recipe ingestion pipeline to support multiple recipe sources (Punch, Diffords, and future sources) with a clean, extensible architecture that eliminates unnecessary complexity.

## Key Changes

### 1. Created RecipeSource Abstraction (`cocktail-utils/src/cocktail_utils/recipes/sources.py`)

**Base Class: `RecipeSource`**
- Abstract interface for all recipe sources
- Methods: `find_html_files()`, `parse_recipe_from_html()`, `derive_source_url()`, `clean_html_content()`

**Implementations:**
- `PunchRecipeSource` - Handles Punch Drink recipes
- `DiffordsRecipeSource` - Handles Difford's Guide recipes

**Registry:**
- `get_recipe_source(name)` - Get source by name
- `RECIPE_SOURCES` - Dict of all available sources

### 2. Eliminated Parquet File Dependency

**Before:**
- Required pre-processing step to create `raw_recipe_ingredients_*.parquet` files
- Complex pipeline with intermediate data storage

**After:**
- Parse HTML directly on-demand when validating/rationalizing
- Simpler, more maintainable pipeline
- No preprocessing required

### 3. Source-Aware Progress Tracking

**Migration Script:** `recipe_ingest/migrate_validation_log.py`
- One-time migration from old to new format
- Creates backup before migration
- Successfully migrated 775 accepted, 1074 rejected, 85 auto-ingested Punch recipes

**Old Format:**
```json
{
  "current_index": 0,
  "accepted": ["Recipe 1"],
  "rejected": [],
  "auto_ingested": [],
  "needs_review": []
}
```

**New Format:**
```json
{
  "punch": {
    "current_index": 0,
    "accepted": ["Recipe 1"],
    "rejected": [],
    "auto_ingested": [],
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
```

### 4. Refactored recipe_validator.py

**New Features:**
- `--source` argument required (punch|diffords)
- `--port` argument for Flask server (default: 5000)
- Parse HTML on-demand (no parquet files)
- Source-specific progress tracking

**Usage:**
```bash
python recipe_validator.py --source punch
python recipe_validator.py --source diffords --port 5001
```

### 5. Refactored batch_rationalize_recipes.py

**New Features:**
- `--source` argument required (punch|diffords)
- Parse HTML directly from source
- Source-specific batch file naming: `rationalized-recipes-{source}-batch-{num}.json`
- Source-aware progress tracking

**Usage:**
```bash
python batch_rationalize_recipes.py --source punch
python batch_rationalize_recipes.py --source diffords --batch-size 50
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Raw HTML Files                                              │
│   - raw_recipes/punch_html/                                 │
│   - raw_recipes/diffords_html/                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ RecipeSource Abstraction (cocktail-utils)                  │
│   - PunchRecipeSource                                       │
│   - DiffordsRecipeSource                                    │
│   - Methods: parse_html, derive_url, clean_html            │
└─────────────────────────────────────────────────────────────┘
                          ↓
           ┌──────────────┴──────────────┐
           │                             │
┌──────────▼─────────┐        ┌──────────▼─────────┐
│ recipe_validator   │        │ batch_rationalize  │
│ --source punch     │        │ --source punch     │
│ --source diffords  │        │ --source diffords  │
└────────────────────┘        └────────────────────┘
           │                             │
           ├─────────────────────────────┤
           │  ingredient_mappings.json   │
           │  validation_log.json        │
           └─────────────┬───────────────┘
                         ↓
           ┌──────────────────────────┐
           │ validated-recipes.json   │
           │ rationalized-recipes-*.json │
           └──────────────────────────┘
```

## Benefits

1. **Simpler Pipeline** - No parquet files or preprocessing steps
2. **Extensible** - Easy to add new recipe sources (just implement `RecipeSource`)
3. **Clean Separation** - Each source has its own progress tracking
4. **Preserved Progress** - All existing Punch recipe progress migrated safely
5. **Better Maintainability** - Shared code in cocktail-utils, cleaner abstractions

## Adding a New Recipe Source

To add a new recipe source (e.g., "liquor.com"):

1. Create a new class in `cocktail-utils/src/cocktail_utils/recipes/sources.py`:
   ```python
   class LiquorComRecipeSource(RecipeSource):
       @property
       def name(self) -> str:
           return "liquorcom"

       # Implement other abstract methods...
   ```

2. Register it in `RECIPE_SOURCES`:
   ```python
   RECIPE_SOURCES = {
       "punch": PunchRecipeSource(),
       "diffords": DiffordsRecipeSource(),
       "liquorcom": LiquorComRecipeSource(),  # Add here
   }
   ```

3. Update CLI argument choices in both tools:
   ```python
   choices=["punch", "diffords", "liquorcom"]
   ```

4. Done! The new source will work with existing tools.

## Git Information

- **Tag:** `pre-multi-source-refactor` (rollback point)
- **Branch:** `refactor/multi-source`
- **Commit:** `e0918b8`

## Next Steps

1. Test validator with Punch recipes (should preserve existing progress)
2. Test validator with Diffords recipes (new source)
3. Test batch rationalizer with both sources
4. Merge to main when validated
