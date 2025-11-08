"""Utility functions for working with the cocktail database."""

import sqlite3
import numpy as np
import pandas as pd
import ot
from scipy.optimize import linear_sum_assignment
from typing import List, Optional, Tuple, Dict, Callable


# Configuration
DB_PATH = "backup-2025-10-21_08-00-45.db"


def load_recipes_from_db(db_path: str = DB_PATH) -> pd.DataFrame:
    """Load recipes and their ingredients from SQLite database.

    Args:
        db_path: Path to SQLite database
        recipe_names: Optional list of recipe names to filter by

    Returns:
        DataFrame with recipe and ingredient information including:
        - recipe_id, recipe_name
        - ingredient_id, ingredient_name, ingredient_path
        - amount, unit_id, conversion_to_ml
        - volume_ml: calculated volume (amount * conversion_to_ml)
        - volume_fraction: fraction of ingredient volume relative to total recipe volume
    """
    conn = sqlite3.connect(db_path)
    # Load all recipes
    query = """
    SELECT
        r.id as recipe_id,
        r.name as recipe_name,
        i.id as ingredient_id,
        i.name as ingredient_name,
        i.path as ingredient_path,
        i.substitution_level as substitution_level,
        ri.amount,
        u.conversion_to_ml * ri.amount as volume_ml,
        (u.conversion_to_ml * ri.amount) / 
            SUM(u.conversion_to_ml * ri.amount) OVER (PARTITION BY r.id) as volume_fraction
    FROM recipes r
    JOIN recipe_ingredients ri ON r.id = ri.recipe_id
    JOIN ingredients i ON ri.ingredient_id = i.id
    LEFT JOIN units u ON ri.unit_id = u.id
    ORDER BY r.id, i.id
    """
    df = pd.read_sql_query(query, conn)

    conn.close()

    if df.empty:
        raise ValueError("No recipes found matching the specified names")

    return df


def load_ingredients_from_db(db_path: str = DB_PATH) -> pd.DataFrame:
    """Load ingredient substitution levels from the database.

    Args:
        db_path: Path to SQLite database

    Returns:
        DataFrame with columns: id, name, path, substitution_level
        substitution_level is 0 for NULL/NaN values
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT id, name, path, COALESCE(substitution_level, 0) as substitution_level FROM ingredients"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def ingredient_distance(
    path1: str, path2: str, substitution_levels: Dict[int, int]
) -> Tuple[int, int]:
    """Calculate distance between two ingredient paths based on their common ancestor.

    The distance is calculated by counting edges from each ingredient to their
    common ancestor, categorizing edges as substitutable (substitution_level=1)
    or not substitutable (substitution_level=0).

    Args:
        path1: First ingredient path (e.g., "/1/5/")
        path2: Second ingredient path (e.g., "/1/6/")
        substitution_levels: Dictionary mapping ingredient_id to substitution_level

    Returns:
        Tuple of (substitutable_links, not_substitutable_links)

    Examples:
        >>> # /1/5/ and /1/6/ with both having substitution_level=1
        >>> ingredient_distance("/1/5/", "/1/6/", {5: 1, 6: 1})
        (2, 0)
        >>> # /1/5/ and /1/ with 5 having substitution_level=1
        >>> ingredient_distance("/1/5/", "/1/", {5: 1})
        (1, 0)
        >>> # /1/8/ and /1/9/ with both having substitution_level=0
        >>> ingredient_distance("/1/8/", "/1/9/", {8: 0, 9: 0})
        (0, 2)
    """
    # Parse paths to get ingredient IDs
    # Paths are like "/1/5/" - split and filter empty strings
    ids1 = [int(x) for x in path1.strip("/").split("/") if x]
    ids2 = [int(x) for x in path2.strip("/").split("/") if x]

    # Find common ancestor by comparing paths from root
    common_length = 0
    for i in range(min(len(ids1), len(ids2))):
        if ids1[i] == ids2[i]:
            common_length = i + 1
        else:
            break

    # Get edges from each path to common ancestor
    # These are the ingredient IDs that differ from the common ancestor
    edges1 = ids1[common_length:]  # Edges from path1 to common ancestor
    edges2 = ids2[common_length:]  # Edges from path2 to common ancestor

    # Count substitutable and not substitutable edges
    substitutable = 0
    not_substitutable = 0

    for ingredient_id in edges1:
        level = substitution_levels.get(ingredient_id, 0)
        if level == 1:
            substitutable += 1
        else:
            not_substitutable += 1

    for ingredient_id in edges2:
        level = substitution_levels.get(ingredient_id, 0)
        if level == 1:
            substitutable += 1
        else:
            not_substitutable += 1

    return (substitutable, not_substitutable)


def weighted_volume_difference(
    ingredient_id1: int,
    volume_ml1: float,
    ingredient_id2: int,
    volume_ml2: float,
    ingredient_paths: Dict[int, str],
    substitution_levels: Dict[int, int],
    substitutable_weight_fn: Callable[[int], float],
    not_substitutable_weight_fn: Callable[[int], float],
) -> float:
    """Calculate weighted volume difference between two ingredients.

    The weight is calculated as:
        abs(volume_ml1 - volume_ml2) * (
            substitutable_weight_fn(substitutable_links) +
            not_substitutable_weight_fn(not_substitutable_links)
        )

    Args:
        ingredient_id1: First ingredient ID
        volume_ml1: Volume in ml for first ingredient
        ingredient_id2: Second ingredient ID
        volume_ml2: Volume in ml for second ingredient
        ingredient_paths: Dictionary mapping ingredient_id to path
        substitution_levels: Dictionary mapping ingredient_id to substitution_level
        substitutable_weight_fn: Callable that takes number of substitutable links
            and returns a weight multiplier
        not_substitutable_weight_fn: Callable that takes number of not substitutable
            links and returns a weight multiplier

    Returns:
        Weighted volume difference as a float

    Examples:
        >>> # Linear weights: accounts for both volume diff AND ingredient diff
        >>> paths = {5: "/1/5/", 6: "/1/6/"}
        >>> sub_levels = {5: 1, 6: 1}
        >>> result = weighted_volume_difference(
        ...     5, 30.0, 6, 15.0, paths, sub_levels,
        ...     lambda x: x, lambda x: x
        ... )
        >>> result
        47.0  # vol_diff=15, ing_weight=2: 15*(2+1) + 2 = 45 + 2 = 47
        >>> # Same ingredient with different volumes (no ingredient penalty)
        >>> result = weighted_volume_difference(
        ...     5, 30.0, 5, 15.0, paths, sub_levels,
        ...     lambda x: x, lambda x: x
        ... )
        >>> result
        15.0  # vol_diff=15, ing_weight=0: 15*(0+1) + 0 = 15
        >>> # Different ingredients, same volume (ingredient penalty only)
        >>> result = weighted_volume_difference(
        ...     5, 30.0, 6, 30.0, paths, sub_levels,
        ...     lambda x: x, lambda x: x
        ... )
        >>> result
        2.0  # vol_diff=0, ing_weight=2: 0*(2+1) + 2 = 2
    """
    # Get paths for both ingredients
    path1 = ingredient_paths[ingredient_id1]
    path2 = ingredient_paths[ingredient_id2]

    # Calculate distance components
    substitutable_links, not_substitutable_links = ingredient_distance(
        path1, path2, substitution_levels
    )

    # Calculate volume difference
    volume_diff = abs(volume_ml1 - volume_ml2)

    # Calculate ingredient distance weight (how different the ingredients are)
    ingredient_distance_weight = substitutable_weight_fn(
        substitutable_links
    ) + not_substitutable_weight_fn(not_substitutable_links)

    # Calculate total distance:
    # 1. Volume difference weighted by ingredient distance (including +1 base)
    # 2. Ingredient difference cost (even when volumes match)
    # This ensures different ingredients contribute to distance even with identical volumes
    return volume_diff * (ingredient_distance_weight + 1) + ingredient_distance_weight


def _build_ingredient_cost_matrix(
    recipe1_ingredient_ids: List[int],
    recipe2_ingredient_ids: List[int],
    ingredient_paths: Dict[int, str],
    substitution_levels: Dict[int, int],
    substitutable_weight_fn: Callable[[int], float],
    not_substitutable_weight_fn: Callable[[int], float],
) -> np.ndarray:
    """Build cost matrix for ingredient distance (cost per unit volume).

    Helper function used by both Hungarian and EMD algorithms to compute the
    base cost of transforming one ingredient into another.

    Args:
        recipe1_ingredient_ids: List of ingredient IDs for recipe 1
        recipe2_ingredient_ids: List of ingredient IDs for recipe 2
        ingredient_paths: Dictionary mapping ingredient_id to path
        substitution_levels: Dictionary mapping ingredient_id to substitution_level
        substitutable_weight_fn: Weight function for substitutable links
        not_substitutable_weight_fn: Weight function for not substitutable links

    Returns:
        Cost matrix of shape (len(recipe1), len(recipe2)) where cost[i,j] is the
        cost per unit volume to transform ingredient i to ingredient j
    """
    n1 = len(recipe1_ingredient_ids)
    n2 = len(recipe2_ingredient_ids)
    cost_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            path1 = ingredient_paths[recipe1_ingredient_ids[i]]
            path2 = ingredient_paths[recipe2_ingredient_ids[j]]
            substitutable_links, not_substitutable_links = ingredient_distance(
                path1, path2, substitution_levels
            )

            # Cost to transform ingredient i to ingredient j (per unit volume)
            cost_matrix[i, j] = substitutable_weight_fn(
                substitutable_links
            ) + not_substitutable_weight_fn(not_substitutable_links)

    return cost_matrix


def recipe_distance_raw(
    recipe1_ingredient_ids: List[int],
    recipe1_volume_fractions: List[float],
    recipe2_ingredient_ids: List[int],
    recipe2_volume_fractions: List[float],
    ingredient_paths: Dict[int, str],
    substitution_levels: Dict[int, int],
    substitutable_weight_fn: Callable[[int], float],
    not_substitutable_weight_fn: Callable[[int], float],
    unmatched_penalty_fn: Optional[Callable[[float], float]] = None,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Calculate volumetric edit distance between two recipes using Hungarian algorithm.

    Low-level function that works with raw ingredient lists. Consider using
    recipe_distance() for a more convenient API that works with DataFrames.

    Finds the optimal one-to-one matching of ingredients between two recipes that
    minimizes the total weighted volume difference. Handles recipes with different
    numbers of ingredients by allowing some ingredients to remain unmatched.

    Args:
        recipe1_ingredient_ids: List of ingredient IDs for recipe 1
        recipe1_volume_fractions: List of volume fractions for recipe 1
        recipe2_ingredient_ids: List of ingredient IDs for recipe 2
        recipe2_volume_fractions: List of volume fractions for recipe 2
        ingredient_paths: Dictionary mapping ingredient_id to path
        substitution_levels: Dictionary mapping ingredient_id to substitution_level
        substitutable_weight_fn: Weight function for substitutable links
        not_substitutable_weight_fn: Weight function for not substitutable links
        unmatched_penalty_fn: Optional function that takes volume_fraction and returns
            penalty for unmatched ingredients. If None, uses volume_fraction as penalty.

    Returns:
        Tuple of (total_distance, matches) where:
        - total_distance: The minimum total weighted volume difference
        - matches: List of tuples (recipe1_idx, recipe2_idx) showing matched pairs.
                   If recipe2_idx >= len(recipe2), ingredient is unmatched.

    Examples:
        >>> # Simple example with 2 ingredients each
        >>> r1_ids = [5, 6]
        >>> r1_fracs = [0.6, 0.4]
        >>> r2_ids = [5, 7]
        >>> r2_fracs = [0.7, 0.3]
        >>> paths = {5: "/1/5/", 6: "/1/6/", 7: "/1/7/"}
        >>> sub_levels = {5: 1, 6: 1, 7: 1}
        >>> distance, matches = recipe_distance_raw(
        ...     r1_ids, r1_fracs, r2_ids, r2_fracs,
        ...     paths, sub_levels, lambda x: x, lambda x: x
        ... )
    """
    # Default penalty function if none provided
    if unmatched_penalty_fn is None:

        def unmatched_penalty_fn(vf):
            return vf

    n1 = len(recipe1_ingredient_ids)
    n2 = len(recipe2_ingredient_ids)

    # Handle edge cases
    if n1 == 0 and n2 == 0:
        return 0.0, []
    if n1 == 0:
        return sum(unmatched_penalty_fn(vf) for vf in recipe2_volume_fractions), []
    if n2 == 0:
        return sum(unmatched_penalty_fn(vf) for vf in recipe1_volume_fractions), []

    # Determine matrix size (square matrix)
    max_size = max(n1, n2)

    # Create cost matrix
    cost_matrix = np.zeros((max_size, max_size))

    # Fill in actual costs for real ingredient pairs
    for i in range(n1):
        for j in range(n2):
            cost = weighted_volume_difference(
                recipe1_ingredient_ids[i],
                recipe1_volume_fractions[i],
                recipe2_ingredient_ids[j],
                recipe2_volume_fractions[j],
                ingredient_paths,
                substitution_levels,
                substitutable_weight_fn,
                not_substitutable_weight_fn,
            )
            cost_matrix[i, j] = cost

    # Add penalties for unmatched ingredients
    # If n1 > n2: pad columns (some recipe1 ingredients will be unmatched)
    if n1 > n2:
        for i in range(n1):
            for j in range(n2, max_size):
                cost_matrix[i, j] = unmatched_penalty_fn(recipe1_volume_fractions[i])

    # If n2 > n1: pad rows (some recipe2 ingredients will be unmatched)
    if n2 > n1:
        for i in range(n1, max_size):
            for j in range(n2):
                cost_matrix[i, j] = unmatched_penalty_fn(recipe2_volume_fractions[j])

    # Remaining corners (if any) get zero cost since they're dummy-to-dummy
    # (already initialized to zero)

    # Apply Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Calculate total distance
    total_distance = cost_matrix[row_indices, col_indices].sum()

    # Create list of matches (only for real ingredients)
    matches = []
    for row_idx, col_idx in zip(row_indices, col_indices):
        # Only include if at least one side is a real ingredient
        if row_idx < n1 or col_idx < n2:
            matches.append((int(row_idx), int(col_idx)))

    return total_distance, matches


def recipe_distance(
    recipes_df: pd.DataFrame,
    recipe_id1: int,
    recipe_id2: int,
    substitution_levels: Dict[int, int],
    substitutable_weight_fn: Callable[[int], float],
    not_substitutable_weight_fn: Callable[[int], float],
    unmatched_penalty_fn: Optional[Callable[[float], float]] = None,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Calculate volumetric edit distance between two recipes from a DataFrame.

    Convenient high-level function that extracts recipe data from a DataFrame
    (as returned by load_recipes_from_db) and calculates the distance.

    Args:
        recipes_df: DataFrame from load_recipes_from_db containing recipe and
            ingredient information with columns: recipe_id, ingredient_id,
            ingredient_path, volume_fraction
        recipe_id1: ID of first recipe
        recipe_id2: ID of second recipe
        substitution_levels: Dictionary mapping ingredient_id to substitution_level
        substitutable_weight_fn: Weight function for substitutable links
        not_substitutable_weight_fn: Weight function for not substitutable links
        unmatched_penalty_fn: Optional function that takes volume_fraction and returns
            penalty for unmatched ingredients. If None, uses volume_fraction as penalty.

    Returns:
        Tuple of (total_distance, matches) where:
        - total_distance: The minimum total weighted volume difference
        - matches: List of tuples (recipe1_idx, recipe2_idx) showing matched pairs.
                   If recipe2_idx >= len(recipe2), ingredient is unmatched.

    Raises:
        ValueError: If either recipe_id is not found in the DataFrame

    Examples:
        >>> df = load_recipes_from_db(recipe_names=['Martini', 'Manhattan'])
        >>> sub_levels = load_ingredient_substitution_levels()
        >>> distance, matches = recipe_distance(
        ...     df, recipe_id1=1, recipe_id2=2,
        ...     substitution_levels=sub_levels,
        ...     substitutable_weight_fn=lambda x: x,
        ...     not_substitutable_weight_fn=lambda x: 2*x
        ... )
    """
    # Extract recipe 1 data
    recipe1_df = recipes_df[recipes_df["recipe_id"] == recipe_id1]
    if recipe1_df.empty:
        raise ValueError(f"Recipe ID {recipe_id1} not found in DataFrame")

    # Extract recipe 2 data
    recipe2_df = recipes_df[recipes_df["recipe_id"] == recipe_id2]
    if recipe2_df.empty:
        raise ValueError(f"Recipe ID {recipe_id2} not found in DataFrame")

    # Build ingredient_paths dict from the DataFrame
    ingredient_paths = dict(
        zip(recipes_df["ingredient_id"], recipes_df["ingredient_path"])
    )

    # Call the raw function with extracted data
    return recipe_distance_raw(
        recipe1_df["ingredient_id"].tolist(),
        recipe1_df["volume_fraction"].tolist(),
        recipe2_df["ingredient_id"].tolist(),
        recipe2_df["volume_fraction"].tolist(),
        ingredient_paths,
        substitution_levels,
        substitutable_weight_fn,
        not_substitutable_weight_fn,
        unmatched_penalty_fn,
    )


def recipe_distance_emd_raw(
    recipe1_ingredient_ids: List[int],
    recipe1_volume_fractions: List[float],
    recipe2_ingredient_ids: List[int],
    recipe2_volume_fractions: List[float],
    ingredient_paths: Dict[int, str],
    substitution_levels: Dict[int, int],
    substitutable_weight_fn: Callable[[int], float],
    not_substitutable_weight_fn: Callable[[int], float],
) -> Tuple[float, List[Tuple[int, int, float, float]]]:
    """Calculate Earth Mover's Distance (Wasserstein distance) between two recipes.

    Low-level function that works with raw ingredient lists. Consider using
    recipe_distance_emd() for a more convenient API that works with DataFrames.

    Uses optimal transport to find the minimum cost to transform one recipe into
    another, allowing fractional/many-to-many matching. This is more natural than
    one-to-one matching: if recipe 1 has 0.5 of ingredient A and recipe 2 has 0.4
    of ingredient A, EMD accounts for the 0.4 in common at zero cost, then matches
    the remaining 0.1 to other ingredients.

    Args:
        recipe1_ingredient_ids: List of ingredient IDs for recipe 1
        recipe1_volume_fractions: List of volume fractions for recipe 1 (must sum to 1)
        recipe2_ingredient_ids: List of ingredient IDs for recipe 2
        recipe2_volume_fractions: List of volume fractions for recipe 2 (must sum to 1)
        ingredient_paths: Dictionary mapping ingredient_id to path
        substitution_levels: Dictionary mapping ingredient_id to substitution_level
        substitutable_weight_fn: Weight function for substitutable links
        not_substitutable_weight_fn: Weight function for not substitutable links

    Returns:
        Tuple of (distance, transport_plan) where:
        - distance: The Earth Mover's Distance (total minimum cost)
        - transport_plan: List of (from_idx, to_idx, amount, cost) tuples showing:
          * from_idx: Index in recipe 1's ingredient list
          * to_idx: Index in recipe 2's ingredient list
          * amount: Volume fraction transported (0 to 1)
          * cost: Total cost for this transport (amount * per-unit cost)
          Only includes non-zero flows (> 1e-10).

    Examples:
        >>> # Recipe with partial overlap
        >>> r1_ids = [5, 6]
        >>> r1_fracs = [0.6, 0.4]
        >>> r2_ids = [5, 7]
        >>> r2_fracs = [0.3, 0.7]
        >>> paths = {5: "/1/5/", 6: "/1/6/", 7: "/1/7/"}
        >>> sub_levels = {5: 1, 6: 1, 7: 1}
        >>> distance, plan = recipe_distance_emd_raw(
        ...     r1_ids, r1_fracs, r2_ids, r2_fracs,
        ...     paths, sub_levels, lambda x: x, lambda x: 2*x
        ... )
        >>> # EMD will match: 0.3 from A→A (cost 0), 0.3 from A→C, 0.4 from B→C
        >>> # plan = [(0, 0, 0.3, 0.0), (0, 1, 0.3, 0.6), (1, 1, 0.4, 0.8)]
    """
    n1 = len(recipe1_ingredient_ids)
    n2 = len(recipe2_ingredient_ids)

    # Handle edge cases
    if n1 == 0 and n2 == 0:
        return 0.0, []
    if n1 == 0:
        return sum(recipe2_volume_fractions), []
    if n2 == 0:
        return sum(recipe1_volume_fractions), []

    # Build cost matrix using shared helper function
    cost_matrix = _build_ingredient_cost_matrix(
        recipe1_ingredient_ids,
        recipe2_ingredient_ids,
        ingredient_paths,
        substitution_levels,
        substitutable_weight_fn,
        not_substitutable_weight_fn,
    )

    # Convert to numpy arrays
    a = np.array(recipe1_volume_fractions, dtype=float)
    b = np.array(recipe2_volume_fractions, dtype=float)

    # Normalize to ensure they sum to 1 (handle floating point errors)
    a = a / a.sum()
    b = b / b.sum()

    # Compute Earth Mover's Distance and transport plan using POT library
    # ot.emd returns the transport matrix (flow[i,j] = amount from i to j)
    transport_matrix = ot.emd(a, b, cost_matrix)

    # Calculate distance from transport matrix and cost matrix
    distance = float(np.sum(transport_matrix * cost_matrix))

    # Convert transport matrix to sparse list format (only non-zero flows)
    # Include the cost for each transport
    transport_plan = []
    for i in range(n1):
        for j in range(n2):
            flow = transport_matrix[i, j]
            if flow > 1e-10:  # Only include significant flows
                flow_cost = float(flow * cost_matrix[i, j])
                transport_plan.append((int(i), int(j), float(flow), flow_cost))

    return distance, transport_plan


def recipe_distance_emd(
    recipes_df: pd.DataFrame,
    recipe_id1: int,
    recipe_id2: int,
    substitution_levels: Dict[int, int],
    substitutable_weight_fn: Callable[[int], float],
    not_substitutable_weight_fn: Callable[[int], float],
) -> Tuple[float, List[Tuple[int, int, float, float]]]:
    """Calculate Earth Mover's Distance between two recipes from a DataFrame.

    High-level API that extracts recipe data from a DataFrame and computes the
    Earth Mover's Distance (Wasserstein distance) using optimal transport.

    EMD finds the minimum cost to transform one recipe into another by allowing
    fractional/many-to-many ingredient matching. This naturally handles partial
    overlaps: common ingredients are matched first at zero cost, then remaining
    amounts are optimally distributed.

    Args:
        recipes_df: DataFrame from load_recipes_from_db() containing recipe data
        recipe_id1: ID of first recipe
        recipe_id2: ID of second recipe
        substitution_levels: Dictionary mapping ingredient_id to substitution_level
        substitutable_weight_fn: Weight function for substitutable links
        not_substitutable_weight_fn: Weight function for not substitutable links

    Returns:
        Tuple of (distance, transport_plan) where:
        - distance: The Earth Mover's Distance between the recipes
        - transport_plan: List of (from_idx, to_idx, amount, cost) tuples showing:
          * from_idx: Index in recipe 1's ingredient list (DataFrame row order)
          * to_idx: Index in recipe 2's ingredient list (DataFrame row order)
          * amount: Volume fraction transported (0 to 1)
          * cost: Total cost for this transport (amount * per-unit cost)

    Raises:
        ValueError: If either recipe_id is not found in the DataFrame

    Examples:
        >>> df = load_recipes_from_db()
        >>> sub_levels = load_ingredient_substitution_levels()
        >>> distance, plan = recipe_distance_emd(
        ...     df, recipe_id1=21, recipe_id2=42,
        ...     substitution_levels=sub_levels,
        ...     substitutable_weight_fn=lambda x: x,
        ...     not_substitutable_weight_fn=lambda x: 10*x
        ... )
        >>> # plan shows how volume flows between ingredients with costs
        >>> for from_idx, to_idx, amount, cost in plan:
        ...     print(f"{amount:.2f} from ing{from_idx} to ing{to_idx} (cost: {cost:.2f})")
    """
    # Extract recipe 1 data
    recipe1_df = recipes_df[recipes_df["recipe_id"] == recipe_id1]
    if recipe1_df.empty:
        raise ValueError(f"Recipe ID {recipe_id1} not found in DataFrame")

    # Extract recipe 2 data
    recipe2_df = recipes_df[recipes_df["recipe_id"] == recipe_id2]
    if recipe2_df.empty:
        raise ValueError(f"Recipe ID {recipe_id2} not found in DataFrame")

    # Build ingredient_paths dict from the DataFrame
    ingredient_paths = dict(
        zip(recipes_df["ingredient_id"], recipes_df["ingredient_path"])
    )

    # Call the raw function with extracted data
    return recipe_distance_emd_raw(
        recipe1_df["ingredient_id"].tolist(),
        recipe1_df["volume_fraction"].tolist(),
        recipe2_df["ingredient_id"].tolist(),
        recipe2_df["volume_fraction"].tolist(),
        ingredient_paths,
        substitution_levels,
        substitutable_weight_fn,
        not_substitutable_weight_fn,
    )
