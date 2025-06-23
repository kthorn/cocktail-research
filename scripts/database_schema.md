# Punch Recipes Database Schema

## Overview
The `punch_recipes.db` is a SQLite database containing cocktail and punch recipes with their ingredients, preparation methods, and categorization tags. The database uses a normalized relational structure with proper foreign key relationships.

## Database Statistics
- **Tables**: 5
- **Indexes**: 0
- **Views**: 0  
- **Triggers**: 0

## Table Structure

### 1. `recipe` Table
Stores the main recipe information including name, preparation method, and source.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Unique recipe identifier |
| `name` | TEXT | NOT NULL | Recipe name |
| `method` | TEXT | | Preparation instructions |
| `source_url` | TEXT | UNIQUE | Source URL for the recipe |

**CREATE Statement:**
```sql
CREATE TABLE recipe(
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    method      TEXT,
    source_url  TEXT UNIQUE
)
```

### 2. `ingredient` Table
Stores unique ingredients used across all recipes.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Unique ingredient identifier |
| `name` | TEXT | NOT NULL, UNIQUE | Ingredient name |

**CREATE Statement:**
```sql
CREATE TABLE ingredient(
    id   INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
)
```

### 3. `tag` Table
Stores categorization tags for recipes (e.g., "rum-based", "tropical", "classic").

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Unique tag identifier |
| `name` | TEXT | NOT NULL, UNIQUE | Tag name |

**CREATE Statement:**
```sql
CREATE TABLE tag(
    id   INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
)
```

**Row Count:** 1,018 tags

### 4. `recipe_ingredient` Table
Junction table linking recipes to their ingredients with quantity and measurement details.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `recipe_id` | INTEGER | PRIMARY KEY (composite) | Reference to recipe |
| `ingredient_id` | INTEGER | PRIMARY KEY (composite) | Reference to ingredient |
| `amount` | REAL | | Quantity of ingredient |
| `unit` | TEXT | | Unit of measurement |
| `note` | TEXT | PRIMARY KEY (composite) | Additional notes about ingredient usage |

**Foreign Keys:**
- `recipe_id` → `recipe(id)` ON DELETE CASCADE
- `ingredient_id` → `ingredient(id)` ON DELETE CASCADE

**CREATE Statement:**
```sql
CREATE TABLE recipe_ingredient(
    recipe_id     INTEGER,
    ingredient_id INTEGER,
    amount        REAL,
    unit          TEXT,
    note          TEXT,
    PRIMARY KEY(recipe_id, ingredient_id, note),
    FOREIGN KEY(recipe_id)     REFERENCES recipe(id)     ON DELETE CASCADE,
    FOREIGN KEY(ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
)
```

### 5. `recipe_tag` Table
Junction table linking recipes to their categorization tags.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `recipe_id` | INTEGER | PRIMARY KEY (composite) | Reference to recipe |
| `tag_id` | INTEGER | PRIMARY KEY (composite) | Reference to tag |

**Foreign Keys:**
- `recipe_id` → `recipe(id)` ON DELETE CASCADE  
- `tag_id` → `tag(id)` ON DELETE CASCADE

**CREATE Statement:**
```sql
CREATE TABLE recipe_tag(
    recipe_id INTEGER,
    tag_id    INTEGER,
    PRIMARY KEY(recipe_id, tag_id),
    FOREIGN KEY(recipe_id) REFERENCES recipe(id) ON DELETE CASCADE,
    FOREIGN KEY(tag_id)    REFERENCES tag(id)    ON DELETE CASCADE
)
```

## Relationships

### Entity Relationship Diagram (Text Format)
```
recipe (1) ←→ (M) recipe_ingredient (M) ←→ (1) ingredient
   ↓
   (1)
   ↓
   (M) recipe_tag (M) ←→ (1) tag
```

### Key Relationships
1. **Recipe ↔ Ingredient**: Many-to-many relationship through `recipe_ingredient`
   - A recipe can have multiple ingredients
   - An ingredient can be used in multiple recipes
   - Includes quantity, unit, and notes for each ingredient in each recipe

2. **Recipe ↔ Tag**: Many-to-many relationship through `recipe_tag`
   - A recipe can have multiple tags
   - A tag can be applied to multiple recipes

## Data Integrity Features
- **Referential Integrity**: All foreign keys have CASCADE DELETE to maintain consistency
- **Unique Constraints**: Ingredient names, tag names, and recipe source URLs are unique
- **Composite Primary Keys**: Junction tables use composite keys to prevent duplicate relationships
- **NOT NULL Constraints**: Essential fields like recipe names and ingredient names cannot be null

## Usage Notes
- The `note` field in `recipe_ingredient` is part of the composite primary key, allowing multiple entries for the same recipe-ingredient combination with different notes
- CASCADE DELETE ensures that removing a recipe automatically removes all its ingredient associations and tag associations
- The database structure supports complex recipe analysis and ingredient-based searches 