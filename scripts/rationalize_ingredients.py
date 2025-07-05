import argparse
import csv
import datetime
from typing import List, Tuple

# Import from our new library
from cocktail_utils.ingredients import IngredientParser


def process_ingredients(
    parser: IngredientParser, min_recipe_count: int = 1
) -> Tuple[List[Tuple[str, dict, int]], List[Tuple[str, int]]]:
    """
    Process all ingredients and separate into matched and unmatched categories.

    Args:
        parser: IngredientParser instance
        min_recipe_count: Minimum number of recipes an ingredient must appear in

    Returns:
        Tuple of (matched_ingredients, unmatched_ingredients)
        matched_ingredients: List of tuples (original_text, match_data, recipe_count)
        unmatched_ingredients: List of tuples (original_text, recipe_count)
    """
    print("Processing ingredients for rationalization...")

    matched_ingredients = []
    unmatched_ingredients = []

    # Get all ingredients from database
    all_ingredients = parser.get_ingredients_from_db(min_recipe_count=min_recipe_count)

    for ing_usage in all_ingredients:
        original_text = ing_usage.ingredient_name
        recipe_count = ing_usage.recipe_count
        match = parser.rationalize_ingredient(original_text)

        if match:
            match_data = {
                "brand": match.brand,
                "specific_type": match.specific_type,
                "category": match.category,
                "confidence": match.confidence,
                "source": match.source,
            }
            matched_ingredients.append((original_text, match_data, recipe_count))
        else:
            unmatched_ingredients.append((original_text, recipe_count))

    # Sort both lists by recipe_count in descending order
    matched_ingredients.sort(key=lambda x: x[2], reverse=True)
    unmatched_ingredients.sort(key=lambda x: x[1], reverse=True)

    print(f"Found {len(matched_ingredients)} matched ingredients")
    print(f"Found {len(unmatched_ingredients)} unmatched ingredients")

    return matched_ingredients, unmatched_ingredients


def write_matched_ingredients_csv(
    matched_ingredients: List[Tuple[str, dict, int]], output_file: str
):
    """
    Write matched ingredients to CSV file with brand, specific_type, category, recipe_count, and original text.

    Args:
        matched_ingredients: List of tuples (original_text, match_data, recipe_count)
        output_file: Path to output CSV file
    """
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(
            [
                "original_text",
                "brand",
                "specific_type",
                "category",
                "confidence",
                "source",
                "recipe_count",
            ]
        )

        # Write data
        for original_text, match_data, recipe_count in matched_ingredients:
            writer.writerow(
                [
                    original_text,
                    match_data["brand"] or "",
                    match_data["specific_type"] or "",
                    match_data["category"],
                    match_data["confidence"],
                    match_data["source"],
                    recipe_count,
                ]
            )

    print(f"Wrote {len(matched_ingredients)} matched ingredients to {output_file}")


def write_unmatched_ingredients_csv(
    unmatched_ingredients: List[Tuple[str, int]], output_file: str
):
    """
    Write unmatched ingredients to CSV file with recipe count.

    Args:
        unmatched_ingredients: List of tuples (original_text, recipe_count)
        output_file: Path to output CSV file
    """
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["original_text", "recipe_count"])

        # Write data
        for original_text, recipe_count in unmatched_ingredients:
            writer.writerow([original_text, recipe_count])

    print(f"Wrote {len(unmatched_ingredients)} unmatched ingredients to {output_file}")


def main():
    """Main function using the cocktail-utils library for database operations."""
    parser_args = argparse.ArgumentParser(
        description="Rationalize cocktail ingredients from database"
    )
    parser_args.add_argument(
        "--min-recipes",
        type=int,
        default=1,
        help="Minimum number of recipes an ingredient must appear in (default: 1)",
    )
    parser_args.add_argument(
        "--db-path",
        type=str,
        default="data/punch_recipes.db",
        help="Path to the database file (default: data/punch_recipes.db)",
    )
    parser_args.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to write output CSV files (default: current directory)",
    )
    args = parser_args.parse_args()

    # Initialize parser with database configuration
    parser = IngredientParser(db_path=args.db_path)

    try:
        # Process ingredients using rationalize_ingredient method
        matched_ingredients, unmatched_ingredients = process_ingredients(
            parser, args.min_recipes
        )

        if not matched_ingredients and not unmatched_ingredients:
            print(
                "No ingredients found in database. Make sure punch_recipes.db exists and has data."
            )
            exit(1)

        # Generate output filenames with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        matched_file = f"{args.output_dir}/rationalized_matched_{timestamp}.csv"
        unmatched_file = f"{args.output_dir}/rationalized_unmatched_{timestamp}.csv"

        # Write results to CSV files
        if matched_ingredients:
            write_matched_ingredients_csv(matched_ingredients, matched_file)
        else:
            print("No matched ingredients found.")

        if unmatched_ingredients:
            write_unmatched_ingredients_csv(unmatched_ingredients, unmatched_file)
        else:
            print("No unmatched ingredients found.")

        # Calculate total recipe count for statistics
        total_matched_recipes = sum(count for _, _, count in matched_ingredients)
        total_unmatched_recipes = sum(count for _, count in unmatched_ingredients)
        total_recipes = total_matched_recipes + total_unmatched_recipes

        print(f"\nSummary:")
        print(
            f"  Total ingredients processed: {len(matched_ingredients) + len(unmatched_ingredients)}"
        )
        print(f"  Matched ingredients: {len(matched_ingredients)}")
        print(f"  Unmatched ingredients: {len(unmatched_ingredients)}")
        if len(matched_ingredients) + len(unmatched_ingredients) > 0:
            print(
                f"  Match rate: {len(matched_ingredients) / (len(matched_ingredients) + len(unmatched_ingredients)) * 100:.1f}%"
            )
        print(f"  Total recipe occurrences: {total_recipes}")
        print(f"  Matched recipe occurrences: {total_matched_recipes}")
        print(f"  Unmatched recipe occurrences: {total_unmatched_recipes}")
        if total_recipes > 0:
            print(
                f"  Recipe occurrence match rate: {total_matched_recipes / total_recipes * 100:.1f}%"
            )

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        raise


if __name__ == "__main__":
    main()
