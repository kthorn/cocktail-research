"""
This script prompts the user for a number, then randomly selects that many
recipe files from the 'raw_recipes' directory. It parses each recipe,
extracts the ingredients, and writes them to a CSV file.

Usage:
    python sample_ingredients_to_csv.py

"""

import csv
import pathlib
import random
from cocktail_utils.recipes import parse_recipe_html
from cocktail_utils.ingredients import parse_quantity


def main():
    """Main function to sample recipes and write ingredients to CSV."""

    # --- 1. Get all recipe files ---
    try:
        recipe_files = list(pathlib.Path("raw_recipes").rglob("*.html"))
        if not recipe_files:
            print("Error: No recipe files found in the 'raw_recipes' directory.")
            return
    except FileNotFoundError:
        print("Error: The 'raw_recipes' directory does not exist.")
        return

    # --- 2. Get user input for the number of recipes ---
    while True:
        try:
            num_to_sample = int(
                input(
                    f"How many recipes would you like to sample? (max: {len(recipe_files)}): "
                )
            )
            if 0 < num_to_sample <= len(recipe_files):
                break
            else:
                print(f"Please enter a number between 1 and {len(recipe_files)}.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    # --- 3. Randomly select recipe files ---
    selected_files = random.sample(recipe_files, num_to_sample)
    print(f"\nRandomly selected {len(selected_files)} recipes. Parsing...")

    # --- 4. Parse recipes and collect ingredients ---
    parsed_ingredients = []
    for file_path in selected_files:
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()

        recipe = parse_recipe_html(html)

        if recipe and recipe.ingredients:
            for ingredient_text in recipe.ingredients:
                amount, unit, ingredient_name = parse_quantity(ingredient_text)
                # Add a tuple of the parsed data
                parsed_ingredients.append(
                    (
                        recipe.name,
                        amount,
                        unit,
                        ingredient_name,
                        ingredient_text,
                    )
                )

    if not parsed_ingredients:
        print("Could not extract any ingredients from the selected files.")
        return

    # --- 5. Write ingredients to a CSV file ---
    output_filename = "random_ingredients.csv"
    try:
        with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(
                ["recipe_name", "amount", "unit", "ingredient", "original_text"]
            )
            # Write data
            writer.writerows(parsed_ingredients)

        print(
            f"\nSuccessfully wrote {len(parsed_ingredients)} ingredients to '{output_filename}'."
        )

    except IOError as e:
        print(f"\nError writing to file: {e}")


if __name__ == "__main__":
    main()
