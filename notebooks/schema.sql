-- Database Schema for CocktailDB
-- This file contains all table definitions and functions for the ingredient hierarchy
-- It is kept to to date with the latest migrations and should represent the current state of the database


-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Table Definitions
CREATE TABLE ingredients (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT COLLATE NOCASE NOT NULL UNIQUE,
  description TEXT,
  parent_id INTEGER,
  path TEXT,
  substitution_level INTEGER DEFAULT 0,
  created_by TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (parent_id) REFERENCES ingredients(id)
);

CREATE TABLE units (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT COLLATE NOCASE NOT NULL UNIQUE,
  abbreviation TEXT,
  conversion_to_ml REAL
);

CREATE TABLE recipes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT COLLATE NOCASE NOT NULL UNIQUE,
  instructions TEXT,
  description TEXT,
  image_url TEXT,
  source TEXT,
  source_url TEXT,
  avg_rating REAL DEFAULT 0,
  rating_count INTEGER DEFAULT 0,
  created_by TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ratings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cognito_user_id TEXT NOT NULL,
  cognito_username TEXT NOT NULL,
  recipe_id INTEGER NOT NULL,
  rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
  FOREIGN KEY (recipe_id) REFERENCES recipes(id) ON DELETE CASCADE,
  UNIQUE(cognito_user_id, recipe_id)
);

CREATE TABLE tags (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT COLLATE NOCASE NOT NULL,
  created_by TEXT NULL, -- NULL for public tags, user_id for private tags
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE recipe_tags (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  recipe_id INTEGER NOT NULL,
  tag_id INTEGER NOT NULL,
  FOREIGN KEY (recipe_id) REFERENCES recipes(id) ON DELETE CASCADE,
  FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
  UNIQUE(recipe_id, tag_id)
);

CREATE TABLE recipe_ingredients (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  recipe_id INTEGER NOT NULL,
  ingredient_id INTEGER NOT NULL,
  unit_id INTEGER,
  amount REAL,
  FOREIGN KEY (recipe_id) REFERENCES recipes(id) ON DELETE CASCADE,
  FOREIGN KEY (ingredient_id) REFERENCES ingredients(id) ON DELETE RESTRICT,
  FOREIGN KEY (unit_id) REFERENCES units(id) ON DELETE SET NULL
);

CREATE TABLE user_ingredients (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cognito_user_id TEXT NOT NULL,
  ingredient_id INTEGER NOT NULL,
  added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (ingredient_id) REFERENCES ingredients(id) ON DELETE CASCADE,
  UNIQUE(cognito_user_id, ingredient_id)
);

-- Create indexes for better performance
CREATE INDEX idx_ingredients_parent_id ON ingredients(parent_id);
CREATE INDEX idx_ingredients_path ON ingredients(path);
CREATE INDEX idx_recipe_ingredients_recipe_id ON recipe_ingredients(recipe_id);
CREATE INDEX idx_recipe_ingredients_ingredient_id ON recipe_ingredients(ingredient_id);
CREATE INDEX idx_recipe_tags_recipe_id ON recipe_tags(recipe_id);
CREATE INDEX idx_recipe_tags_tag_id ON recipe_tags(tag_id);
CREATE INDEX idx_ratings_cognito_user_id ON ratings(cognito_user_id);
CREATE INDEX idx_ratings_recipe_id ON ratings(recipe_id);
CREATE INDEX idx_user_ingredients_cognito_user_id ON user_ingredients(cognito_user_id);
CREATE INDEX idx_user_ingredients_ingredient_id ON user_ingredients(ingredient_id);
CREATE INDEX idx_recipes_created_by ON recipes(created_by);
CREATE INDEX idx_ingredients_created_by ON ingredients(created_by);

CREATE UNIQUE INDEX idx_public_tags ON tags(name) WHERE created_by IS NULL;
CREATE UNIQUE INDEX idx_private_tags ON tags(name, created_by) WHERE created_by IS NOT NULL;
CREATE INDEX idx_tags_created_by ON tags(created_by);

-- Create trigger to update average rating when a new rating is added
CREATE TRIGGER update_avg_rating_insert 
AFTER INSERT ON ratings
BEGIN
  UPDATE recipes
  SET 
    avg_rating = (SELECT AVG(rating) FROM ratings WHERE recipe_id = NEW.recipe_id),
    rating_count = (SELECT COUNT(*) FROM ratings WHERE recipe_id = NEW.recipe_id)
  WHERE id = NEW.recipe_id;
END;

-- Create trigger to update average rating when a rating is updated
CREATE TRIGGER update_avg_rating_update 
AFTER UPDATE ON ratings
WHEN OLD.rating <> NEW.rating
BEGIN
  UPDATE recipes
  SET 
    avg_rating = (SELECT AVG(rating) FROM ratings WHERE recipe_id = NEW.recipe_id),
    rating_count = (SELECT COUNT(*) FROM ratings WHERE recipe_id = NEW.recipe_id)
  WHERE id = NEW.recipe_id;
END;

-- Create trigger to update average rating when a rating is deleted
CREATE TRIGGER update_avg_rating_delete 
AFTER DELETE ON ratings
BEGIN
  UPDATE recipes
  SET 
    avg_rating = COALESCE((SELECT AVG(rating) FROM ratings WHERE recipe_id = OLD.recipe_id), 0),
    rating_count = (SELECT COUNT(*) FROM ratings WHERE recipe_id = OLD.recipe_id)
  WHERE id = OLD.recipe_id;
END;

-- Create trigger to automatically update updated_at for recipes
CREATE TRIGGER update_recipes_updated_at
AFTER UPDATE ON recipes
BEGIN
  UPDATE recipes SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Create trigger to automatically update updated_at for ingredients
CREATE TRIGGER update_ingredients_updated_at
AFTER UPDATE ON ingredients
BEGIN
  UPDATE ingredients SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;


-- Insert common measurement units
INSERT INTO units (name, abbreviation, conversion_to_ml) VALUES
  ('Ounce', 'oz', 29.5735),
  ('Tablespoon', 'tbsp', 14.7868),
  ('Teaspoon', 'tsp', 4.92892),
  ('Barspoon', 'bsp', 2.46446),
  ('Dash', 'dash', 0.616115),
  ('Each', 'each', NULL),
  ('Drop', 'drop', 0.05),
  ('to top', 'top', NULL),
  ('to rinse', 'rinse', NULL);

-- Insert base ingredients
INSERT INTO ingredients (name, description, path) VALUES
  ('Whiskey', 'A spirit distilled from fermented grain mash', '/1/'),
  ('Rum', 'A spirit distilled from sugarcane byproducts', '/2/'),
  ('Vodka', 'A spirit distilled from fermented grains or potatoes', '/3/'),
  ('Gin', 'A spirit distilled from juniper berries', '/4/'),
  ('Brandy', 'A spirit distilled from wine or fruit', '/5/'),
  ('Tequila', 'A spirit distilled from agave', '/6/'),
  ('Juice', '', '/7/');

-- Insert common tags
INSERT INTO tags (name, created_by) VALUES
  ('Tiki', NULL),
  ('Classic', NULL);