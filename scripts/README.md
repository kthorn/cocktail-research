# Scripts Documentation

This directory contains various scripts for processing, analyzing, and managing cocktail recipe data. Each script serves a specific purpose in the cocktail research pipeline.

## Script Overview

### Data Collection Scripts

#### `download_punch_recipes.py`
**Purpose**: Downloads recipe pages from punchdrink.com to a local directory for offline processing.

**Features**:
- Uses Algolia API to search and discover recipe URLs
- Downloads recipes by spirit category to ensure comprehensive coverage
- Implements polite scraping with retry logic and rate limiting
- Saves HTML files with sanitized filenames

**Output**: HTML files saved to `raw_recipes/punch_html/` directory

---

### Data Processing Scripts

#### `parse_local_recipes.py`
**Purpose**: Parses local HTML recipe files and loads them into a SQLite database.

**Features**:
- Processes all HTML files in the `raw_recipes` directory
- Extracts recipe metadata (name, description, garnish, directions)
- Parses ingredient quantities and names
- Creates database schema and populates tables

**Output**: Populated SQLite database at `data/recipes.db`

---

#### `sample_ingredients_to_csv.py`
**Purpose**: Randomly samples recipe files and extracts ingredients to CSV format for analysis.

**Features**:
- Interactive script that prompts for number of recipes to sample
- Randomly selects recipe files from the `raw_recipes` directory
- Parses ingredients and extracts quantity, unit, and ingredient name
- Outputs results to CSV with recipe context

**Output**: `random_ingredients.csv` with columns: recipe_name, amount, unit, ingredient, original_text

---

### Ingredient Analysis Scripts

#### `dictionary_match_ingredients.py`
**Purpose**: Performs dictionary-based ingredient matching and generates comprehensive reports.

**Features**:
- Matches ingredients against the ingredient taxonomy dictionary
- Separates matched and unmatched ingredients
- Generates detailed CSV reports with confidence scores
- Provides statistical summaries of match rates

**Usage**:
```bash
python dictionary_match_ingredients.py [--min-recipe-count N] [--db-path PATH] [--output-dir DIR]
```

**Arguments**:
- `--min-recipe-count`: Minimum recipes an ingredient must appear in (default: 1)
- `--db-path`: Path to SQLite database (default: data/punch_recipes.db)
- `--output-dir`: Directory for output files (default: current directory)

**Output**: 
- `matched_ingredients_TIMESTAMP.csv` - Successfully matched ingredients
- `unmatched_ingredients_TIMESTAMP.csv` - Ingredients that couldn't be matched
- Console summary with match statistics

---

#### `rationalize_ingredients.py`
**Purpose**: Main script for ingredient rationalization using the cocktail-utils library.

**Features**:
- Loads ingredients from database with configurable minimum recipe count
- Supports multiple output formats (JSON, CSV)
- Uses the IngredientParser for comprehensive ingredient analysis
- Generates timestamped output files

**Usage**:
```bash
python rationalize_ingredients.py [--format json|csv] [--output FILENAME] [--min-recipes N] [--db-path PATH]
```

**Arguments**:
- `--format`: Output format - json or csv (default: json)
- `--output`: Custom output filename (default: auto-generated with timestamp)
- `--min-recipes`: Minimum recipes for ingredient inclusion (default: 2)
- `--db-path`: Path to database file (default: data/punch_recipes.db)

---

### Model Evaluation Scripts

#### `evaluate_models.py`
**Purpose**: Evaluates different LLM models for ingredient rationalization tasks.

**Features**:
- Finds ingredients that failed dictionary matching
- Tests multiple AWS Bedrock models on the same ingredients
- Generates comparison CSV with model outputs
- Creates a text file listing unmatched ingredients for inspection

**Usage**:
```bash
python evaluate_models.py [--num-ingredients N] [--models MODEL1 MODEL2...] [--db-path PATH]
```

**Arguments**:
- `--num-ingredients`: Number of ingredients to test (default: 20)
- `--models`: List of Bedrock model IDs to evaluate (default: claude-3-5-haiku, nova-lite)
- `--db-path`: Path to SQLite database (default: data/punch_recipes.db)

**Output**:
- `unmatched_ingredients_TIMESTAMP.txt` - List of ingredients for testing
- `model_evaluation_TIMESTAMP.csv` - Comparison of model outputs

**Dependencies**:
- `cocktail_utils.ingredients.IngredientParser`
- AWS Bedrock access

---

## Common Dependencies

All scripts depend on the `cocktail-utils` library, which provides:
- Database utilities and schema management
- Ingredient parsing and normalization
- Recipe HTML parsing
- Web scraping utilities
- LLM integration for ingredient analysis

## Database Schema

The scripts work with a SQLite database containing:
- `recipe` table: Recipe metadata (name, description, directions, etc.)
- `ingredient` table: Unique ingredients
- `recipe_ingredient` table: Many-to-many relationship with quantities

## Output Files

Most scripts generate timestamped output files to avoid overwriting previous results:
- Format: `{script_name}_{YYYYMMDD_HHMMSS}.{extension}`
- Common extensions: `.csv`, `.json`, `.txt`

## Error Handling

Scripts include error handling for:
- Database connection issues
- File I/O errors
- Network timeouts during scraping
- LLM API failures
- Invalid input data

## Usage Workflow

A typical workflow might be:
1. `download_punch_recipes.py` - Collect recipe data
2. `parse_local_recipes.py` - Load into database
3. `dictionary_match_ingredients.py` - Analyze ingredient matching
4. `evaluate_models.py` - Test LLM performance on unmatched ingredients
5. `rationalize_ingredients.py` - Generate final analysis reports 