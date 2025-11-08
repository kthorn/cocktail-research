"""
Distance calculation utilities for cocktail recipe analysis.

This module provides functions for calculating pairwise distances between recipes
based on their ingredient compositions, with support for ingredient substitutability.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm.auto import tqdm
from sklearn.metrics import pairwise_distances


def parse_path(path: str) -> List[int]:
    """Parse ingredient path string into list of IDs.

    Example: '/1/10/' -> [1, 10]
    """
    if not path:
        return []
    return [int(x) for x in path.strip("/").split("/") if x]


def are_ingredients_substitutable(
    path1: str, path2: str, sub_level1: int, sub_level2: int
) -> bool:
    """Check if two ingredients are substitutable based on paths and substitution levels.

    Args:
        path1: Path of first ingredient (e.g., '/1/10/')
        path2: Path of second ingredient (e.g., '/1/23/')
        sub_level1: Substitution level of first ingredient
        sub_level2: Substitution level of second ingredient

    Returns:
        True if ingredients are substitutable (distance should be reduced)
    """
    nodes1 = parse_path(path1)
    nodes2 = parse_path(path2)

    # Case 1: Both have substitution_level 1 and share the same parent
    # E.g., Rittenhouse Rye (/1/22/389/) and Woody Creek Rye (/1/22/406/)
    if sub_level1 == 1 and sub_level2 == 1:
        # Must have at least one parent level
        if len(nodes1) >= 2 and len(nodes2) >= 2:
            # Check if they have the same parent (all but last element should match)
            if nodes1[:-1] == nodes2[:-1]:
                return True

    # Case 2: One ingredient is the direct parent of another with substitution_level 1
    # E.g., Rye (/1/22/) and Rittenhouse Rye (/1/22/389/ with sub_level=1)
    if sub_level1 == 1 and len(nodes1) >= 2:
        # Check if ingredient2 is the parent of ingredient1
        if nodes2 == nodes1[:-1]:
            return True

    if sub_level2 == 1 and len(nodes2) >= 2:
        # Check if ingredient1 is the parent of ingredient2
        if nodes1 == nodes2[:-1]:
            return True

    return False


def build_recipe_ingredient_matrix(
    recipes_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build normalized recipe-ingredient matrices (amounts and boolean).

    Args:
        recipes_df: DataFrame with columns: recipe_name, ingredient_name, amount, conversion_to_ml

    Returns:
        Tuple of (normalized_amounts_df, boolean_df)
    """
    # Convert amounts to ml
    recipes_df = recipes_df.copy()
    recipes_df["amount_ml"] = recipes_df.apply(
        lambda row: row["amount"] * row["conversion_to_ml"]
        if pd.notna(row["conversion_to_ml"]) and pd.notna(row["amount"])
        else row["amount"]
        if pd.notna(row["amount"])
        else 1.0,  # Default to 1 if no amount
        axis=1,
    )

    # Create pivot table for amounts
    amount_matrix = recipes_df.pivot_table(
        index="recipe_name",
        columns="ingredient_name",
        values="amount_ml",
        aggfunc="sum",
        fill_value=0,
    )

    # Normalize each recipe to sum to 1 (proportions)
    normalized_matrix = amount_matrix.div(amount_matrix.sum(axis=1), axis=0)
    normalized_matrix = normalized_matrix.fillna(0)

    # Remove recipes/ingredients that are all zeros
    normalized_matrix = normalized_matrix.loc[(normalized_matrix != 0).any(axis=1), :]
    normalized_matrix = normalized_matrix.loc[:, (normalized_matrix != 0).any(axis=0)]

    # Create boolean matrix
    boolean_matrix = (amount_matrix > 0).astype(bool)
    boolean_matrix = boolean_matrix.loc[
        normalized_matrix.index, normalized_matrix.columns
    ]

    return normalized_matrix, boolean_matrix


def build_substitutability_matrix(
    recipes_df: pd.DataFrame, normalized_matrix: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Build substitutability matrices for ingredients.

    Args:
        recipes_df: DataFrame with ingredient metadata
        normalized_matrix: Normalized recipe-ingredient matrix

    Returns:
        Tuple of (substitutable_matrix, col_substitutable):
            - substitutable_matrix: (n_unique_ingredients, n_unique_ingredients) bool array
            - col_substitutable: (n_matrix_columns, n_matrix_columns) bool array
    """
    # Get unique ingredients with their metadata
    ingredient_info = recipes_df[
        ["ingredient_id", "ingredient_name", "ingredient_path", "substitution_level"]
    ].drop_duplicates()
    ingredient_info = ingredient_info.set_index("ingredient_id")

    # Build substitutability matrix
    ingredient_ids = ingredient_info.index.tolist()
    n_ingredients = len(ingredient_ids)
    substitutable_matrix = np.zeros((n_ingredients, n_ingredients), dtype=bool)

    for i in range(n_ingredients):
        for j in range(i + 1, n_ingredients):
            id1, id2 = ingredient_ids[i], ingredient_ids[j]
            path1 = ingredient_info.loc[id1, "ingredient_path"]
            path2 = ingredient_info.loc[id2, "ingredient_path"]
            sub1 = ingredient_info.loc[id1, "substitution_level"]
            sub2 = ingredient_info.loc[id2, "substitution_level"]

            if are_ingredients_substitutable(path1, path2, sub1, sub2):
                substitutable_matrix[i, j] = True
                substitutable_matrix[j, i] = True

    print(f"Found {substitutable_matrix.sum() // 2} substitutable ingredient pairs")

    # Pre-compute ingredient name to ID mapping for fast lookup
    ing_name_to_id = (
        recipes_df[["ingredient_name", "ingredient_id"]]
        .drop_duplicates()
        .set_index("ingredient_name")["ingredient_id"]
        .to_dict()
    )

    # Create a mapping from ingredient column index to ingredient ID position in substitutable_matrix
    col_to_id_pos = {}
    for col_idx, ing_name in enumerate(normalized_matrix.columns):
        if ing_name in ing_name_to_id:
            ing_id = ing_name_to_id[ing_name]
            if ing_id in ingredient_ids:
                col_to_id_pos[col_idx] = ingredient_ids.index(ing_id)

    # Build a substitutability matrix in the same space as normalized_matrix columns
    n_cols = len(normalized_matrix.columns)
    col_substitutable = np.zeros((n_cols, n_cols), dtype=bool)

    for col_i in range(n_cols):
        for col_j in range(col_i + 1, n_cols):
            if col_i in col_to_id_pos and col_j in col_to_id_pos:
                id_pos_i = col_to_id_pos[col_i]
                id_pos_j = col_to_id_pos[col_j]
                if substitutable_matrix[id_pos_i, id_pos_j]:
                    col_substitutable[col_i, col_j] = True
                    col_substitutable[col_j, col_i] = True

    return substitutable_matrix, col_substitutable


def calculate_weighted_ingredient_distance(
    recipes_df: pd.DataFrame,
    normalized_matrix: pd.DataFrame = None,
    batch_size: int = 100,
    substitution_weight: float = 0.5,
) -> pd.DataFrame:
    """Calculate weighted Manhattan distance considering ingredient substitutability.

    Args:
        recipes_df: DataFrame with recipe and ingredient information
        normalized_matrix: Pre-computed normalized matrix (optional, will be computed if None)
        batch_size: Number of recipes to process in each batch
        substitution_weight: Weight applied to substitutable ingredient differences (0.0 = no penalty, 1.0 = full penalty)
                           Default 0.5 means substitutable ingredients contribute half the distance

    Returns:
        Weighted distance matrix between recipes
    """
    # Build recipe matrices if not provided
    if normalized_matrix is None:
        normalized_matrix, _ = build_recipe_ingredient_matrix(recipes_df)

    # Build substitutability matrices
    _, col_substitutable = build_substitutability_matrix(recipes_df, normalized_matrix)

    # Convert to numpy for vectorized operations
    recipe_matrix = normalized_matrix.values
    n_recipes = len(recipe_matrix)
    n_cols = len(normalized_matrix.columns)

    # Calculate all pairwise differences at once (broadcasting)
    print("Computing pairwise differences...")

    # Use a more memory-efficient approach: compute in batches
    weighted_distance_matrix = np.zeros((n_recipes, n_recipes))

    for batch_start in tqdm(
        range(0, n_recipes, batch_size), desc="Processing recipe batches"
    ):
        batch_end = min(batch_start + batch_size, n_recipes)
        batch_recipes = recipe_matrix[batch_start:batch_end]

        # Compute differences between batch and all recipes
        # batch_recipes: (batch_size, n_ingredients)
        # recipe_matrix: (n_recipes, n_ingredients)
        # diff: (batch_size, n_recipes, n_ingredients)
        diff = np.abs(batch_recipes[:, np.newaxis, :] - recipe_matrix[np.newaxis, :, :])

        # Start with standard Manhattan distance
        weighted_diff = diff.copy()

        # Apply substitutability corrections
        # For each pair of substitutable ingredients, adjust the weighted difference
        for col_i in range(n_cols):
            for col_j in range(col_i + 1, n_cols):
                if col_substitutable[col_i, col_j]:
                    # Get amounts for both ingredients in all recipe pairs
                    batch_amounts_i = batch_recipes[:, col_i : col_i + 1]
                    batch_amounts_j = batch_recipes[:, col_j : col_j + 1]
                    all_amounts_i = recipe_matrix[:, col_i : col_i + 1].T
                    all_amounts_j = recipe_matrix[:, col_j : col_j + 1].T

                    # Combined amounts for substitutable ingredients
                    # Shapes: batch_amounts: (batch_size, 1), all_amounts: (1, n_recipes)
                    combined_batch = (
                        batch_amounts_i + batch_amounts_j
                    )  # (batch_size, 1)
                    combined_all = all_amounts_i + all_amounts_j  # (1, n_recipes)
                    combined_diff = np.abs(
                        combined_batch - combined_all
                    )  # (batch_size, n_recipes)

                    # Apply weighting when recipes differ in substitutable ingredients
                    # mask1: recipe1 has col_i, recipe2 has col_j (different substitutable ingredients)
                    # mask2: recipe1 has col_j, recipe2 has col_i (different substitutable ingredients)
                    mask1 = (batch_amounts_i > 0) & (
                        all_amounts_j > 0
                    )  # (batch_size, n_recipes)
                    mask2 = (batch_amounts_j > 0) & (
                        all_amounts_i > 0
                    )  # (batch_size, n_recipes)

                    # If either mask is true, apply the weighting to BOTH columns
                    substitution_applies = mask1 | mask2

                    weighted_diff[:, :, col_i] = np.where(
                        substitution_applies,
                        np.minimum(
                            weighted_diff[:, :, col_i],
                            combined_diff * substitution_weight,
                        ),
                        weighted_diff[:, :, col_i],
                    )

                    weighted_diff[:, :, col_j] = np.where(
                        substitution_applies,
                        np.minimum(
                            weighted_diff[:, :, col_j],
                            combined_diff * substitution_weight,
                        ),
                        weighted_diff[:, :, col_j],
                    )

        # Sum across ingredients to get distances
        weighted_distance_matrix[batch_start:batch_end, :] = weighted_diff.sum(axis=2)

    # Make symmetric
    weighted_distance_matrix = (
        weighted_distance_matrix + weighted_distance_matrix.T
    ) / 2

    return pd.DataFrame(
        weighted_distance_matrix,
        index=normalized_matrix.index,
        columns=normalized_matrix.index,
    )


def calculate_manhattan_distance(normalized_matrix: pd.DataFrame) -> pd.DataFrame:
    """Calculate standard Manhattan distance between recipes.

    Args:
        normalized_matrix: Normalized recipe-ingredient matrix

    Returns:
        Distance matrix DataFrame
    """
    distance_matrix = pairwise_distances(normalized_matrix, metric="manhattan")

    return pd.DataFrame(
        distance_matrix,
        index=normalized_matrix.index,
        columns=normalized_matrix.index,
    )
