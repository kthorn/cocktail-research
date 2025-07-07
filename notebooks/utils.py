import pandas as pd


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet("../data/rationalized_matrix_amount_20250706_124957.parquet")
    df_bool = pd.read_parquet(
        "../data/rationalized_matrix_boolean_20250706_124957.parquet"
    )

    grouped_cols = df.groupby(level=[0, 1], axis=1)
    df_collapsed = grouped_cols.sum()
    # Create normalized dataframe where each row sums to 1
    df_normalized = df_collapsed.div(df_collapsed.sum(axis=1), axis=0)
    # Drop columns that are all 0 or NaN
    df_normalized = df_normalized.fillna(0)
    df_normalized = df_normalized.loc[:, (df_normalized != 0).any(axis=0)]
    df_normalized = df_normalized.loc[(df_normalized != 0).any(axis=1), :]

    grouped_cols = df_bool.groupby(level=[0, 1], axis=1)
    df_bool_collapsed = grouped_cols.any()
    df_bool_collapsed = df_bool_collapsed.fillna(0).astype(bool)
    df_bool_collapsed = df_bool_collapsed.loc[
        :, df_bool_collapsed.any(axis=0)
    ]  # Remove columns with all False
    df_bool_collapsed = df_bool_collapsed.loc[
        df_bool_collapsed.any(axis=1), :
    ]  # Remove rows with all False
    return df_normalized, df_bool_collapsed


# Function to get recipe information for each cocktail
def get_recipe_string(df_normalized: pd.DataFrame, cocktail_name: str) -> str:
    """Get the recipe as a formatted string for a given cocktail."""
    recipe = df_normalized.loc[cocktail_name, :]
    ingredients = recipe[recipe > 0].sort_values(ascending=False)

    if len(ingredients) == 0:
        return "No ingredients found"

    # Format ingredients with proportions
    recipe_parts = []
    for ingredient, proportion in ingredients.items():
        # Handle multi-level column names if they exist
        if isinstance(ingredient, tuple):
            ingredient_name = " - ".join(str(part) for part in ingredient)
        else:
            ingredient_name = str(ingredient)
        recipe_parts.append(f"{ingredient_name}: {proportion:.3f}")

    return " | ".join(recipe_parts)
