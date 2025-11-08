"""
Recipe NMF and UMAP Analysis from SQLite Database

This script extracts recipes from a SQLite database, calculates distances based on
hierarchical ingredient relationships, and performs NMF factorization and UMAP embeddings.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import umap
from sklearn.decomposition import NMF
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple

# Import distance calculation utilities
from distance_calculations import (
    build_recipe_ingredient_matrix,
    calculate_weighted_ingredient_distance,
    calculate_manhattan_distance,
)


# Configuration
DB_PATH = "backup-2025-10-17_08-00-45.db"
CLASSES_PATH = "recipe_classes.json"  # Path to recipe categories JSON file
DISTANCE_BASE = 3.0  # Exponential base for ingredient distance (easily tunable)
N_COMPONENTS = 60  # Number of NMF components
UMAP_N_NEIGHBORS = 5
UMAP_RANDOM_STATE = 42
UMAP_MIN_DIST = 0.05


def load_recipes_from_db(db_path: str) -> pd.DataFrame:
    """Load recipes and their ingredients from SQLite database."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        r.id as recipe_id,
        r.name as recipe_name,
        i.id as ingredient_id,
        i.name as ingredient_name,
        i.path as ingredient_path,
        i.substitution_level,
        ri.amount,
        ri.unit_id,
        u.conversion_to_ml
    FROM recipes r
    JOIN recipe_ingredients ri ON r.id = ri.recipe_id
    JOIN ingredients i ON ri.ingredient_id = i.id
    LEFT JOIN units u ON ri.unit_id = u.id
    ORDER BY r.id, i.id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def load_ingredients_hierarchy(db_path: str) -> pd.DataFrame:
    """Load ingredient hierarchy information."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        id,
        name,
        path,
        parent_id
    FROM ingredients
    ORDER BY id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def load_recipe_classes(classes_path: str) -> Dict[str, List[str]]:
    """Load recipe category mappings from JSON file.

    Args:
        classes_path: Path to JSON file containing category mappings

    Returns:
        Dictionary mapping category names to lists of recipe names
    """
    try:
        with open(classes_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: Recipe classes file '{classes_path}' not found. Using no categorization."
        )
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in '{classes_path}'. Using no categorization.")
        return {}


def assign_recipe_categories(
    recipe_names: List[str], recipe_classes: Dict[str, List[str]]
) -> pd.Series:
    """Assign categories to recipes based on name matching.

    Args:
        recipe_names: List of recipe names to categorize
        recipe_classes: Dictionary mapping category names to recipe name lists

    Returns:
        pandas Series with recipe names as index and categories as values
    """
    # Create reverse mapping from recipe name to category
    name_to_category = {}
    for category, recipes in recipe_classes.items():
        for recipe_name in recipes:
            name_to_category[recipe_name] = category

    # Assign categories, default to "other" for unmatched recipes
    categories = []
    for recipe_name in recipe_names:
        categories.append(name_to_category.get(recipe_name, "other"))

    return pd.Series(categories, index=recipe_names, name="category")


def calculate_tree_distance(
    path1: str, path2: str, base: float = DISTANCE_BASE
) -> float:
    """Calculate exponential distance between two ingredients based on their hierarchical paths.

    Args:
        path1: Path of first ingredient (e.g., '/1/10/')
        path2: Path of second ingredient (e.g., '/1/23/')
        base: Exponential base for distance calculation

    Returns:
        Exponential distance: base^(steps_to_common_ancestor)
    """
    from distance_calculations import parse_path

    nodes1 = parse_path(path1)
    nodes2 = parse_path(path2)

    # Find common ancestor depth
    common_depth = 0
    for n1, n2 in zip(nodes1, nodes2):
        if n1 == n2:
            common_depth += 1
        else:
            break

    # Calculate steps from each node to common ancestor
    steps1 = len(nodes1) - common_depth
    steps2 = len(nodes2) - common_depth
    total_steps = steps1 + steps2

    # If same ingredient
    if total_steps == 0:
        return 0.0

    # Exponential distance based on steps
    return base**total_steps


def build_ingredient_distance_matrix(
    ingredients_df: pd.DataFrame, base: float = DISTANCE_BASE
) -> pd.DataFrame:
    """Build pairwise distance matrix for all ingredients using exponential tree distance."""
    n_ingredients = len(ingredients_df)
    distance_matrix = np.zeros((n_ingredients, n_ingredients))

    paths = ingredients_df["path"].values
    ingredient_ids = ingredients_df["id"].values

    for i in range(n_ingredients):
        for j in range(i + 1, n_ingredients):
            dist = calculate_tree_distance(paths[i], paths[j], base)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    distance_df = pd.DataFrame(
        distance_matrix, index=ingredient_ids, columns=ingredient_ids
    )

    return distance_df


def get_recipe_string(recipe_name: str, normalized_matrix: pd.DataFrame) -> str:
    """Get formatted recipe string showing ingredient proportions."""
    recipe = normalized_matrix.loc[recipe_name, :]
    ingredients = recipe[recipe > 0].sort_values(ascending=False)

    if len(ingredients) == 0:
        return "No ingredients found"

    recipe_parts = [f"{ing}: {prop:.3f}" for ing, prop in ingredients.items()]
    return " | ".join(recipe_parts)


def perform_nmf_analysis(
    normalized_matrix: pd.DataFrame, n_components: int = N_COMPONENTS
) -> Tuple[np.ndarray, np.ndarray, NMF]:
    """Perform NMF factorization on recipe-ingredient matrix.

    Returns:
        Tuple of (W: recipe_factors, H: ingredient_factors, model)
    """
    nmf = NMF(
        n_components=n_components,
        beta_loss="kullback-leibler",
        solver="mu",
        l1_ratio=0.0,
        random_state=42,
    )

    W = nmf.fit_transform(normalized_matrix)  # Recipe factors
    H = nmf.components_  # Ingredient factors

    return W, H, nmf


def create_umap_embedding(
    distance_matrix: np.ndarray, metric: str = "precomputed"
) -> np.ndarray:
    """Create UMAP embedding from distance matrix or feature matrix."""
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric=metric,
        random_state=UMAP_RANDOM_STATE,
    )

    embedding = reducer.fit_transform(distance_matrix)
    return embedding


def plot_umap_embedding(
    embedding_df: pd.DataFrame,
    title: str,
    color_by: str = None,
    highlight_search: str = None,
):
    """Create interactive Plotly scatter plot of UMAP embedding.

    Args:
        embedding_df: DataFrame with UMAP coordinates and recipe info
        title: Plot title
        color_by: Column name to color points by
        highlight_search: Substring to search for in recipe names (case-insensitive)
    """
    # Add search highlighting
    if highlight_search:
        embedding_df = embedding_df.copy()
        embedding_df["highlight"] = embedding_df["name"].str.contains(
            highlight_search, case=False, na=False
        )

        # Create separate traces for highlighted and non-highlighted points
        fig = go.Figure()

        # Non-highlighted points
        non_highlighted = embedding_df[~embedding_df["highlight"]]
        fig.add_trace(
            go.Scatter(
                x=non_highlighted["UMAP1"],
                y=non_highlighted["UMAP2"],
                mode="markers",
                marker=dict(
                    size=8,
                    opacity=0.7,
                    color="steelblue",
                    line=dict(width=1, color="white"),
                ),
                text=non_highlighted["name"],
                customdata=non_highlighted["recipe"],
                hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
                name="Other recipes",
                showlegend=True,
            )
        )

        # Highlighted points
        highlighted = embedding_df[embedding_df["highlight"]]
        fig.add_trace(
            go.Scatter(
                x=highlighted["UMAP1"],
                y=highlighted["UMAP2"],
                mode="markers",
                marker=dict(
                    size=8,
                    opacity=0.7,
                    color="red",
                    line=dict(width=1, color="white"),
                ),
                text=highlighted["name"],
                customdata=highlighted["recipe"],
                hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
                name=f'Matches "{highlight_search}"',
                showlegend=True,
            )
        )

        fig.update_layout(
            title=f'{title}<br><sub>Highlighting: "{highlight_search}" ({len(highlighted)} matches)</sub>',
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            width=900,
            height=700,
        )

    elif color_by and color_by in embedding_df.columns:
        # Check if this is categorical data (strings/objects)
        if (
            embedding_df[color_by].dtype == "object"
            or embedding_df[color_by].dtype.name == "category"
        ):
            # Use a discrete color palette for categorical data
            fig = px.scatter(
                embedding_df,
                x="UMAP1",
                y="UMAP2",
                hover_data=["name", "recipe"],
                color=color_by,
                color_discrete_sequence=px.colors.qualitative.D3,
                title=title,
                width=900,
                height=700,
            )
        else:
            # Use continuous color scale for numerical data
            fig = px.scatter(
                embedding_df,
                x="UMAP1",
                y="UMAP2",
                hover_data=["name", "recipe"],
                color=color_by,
                title=title,
                width=900,
                height=700,
            )
        fig.update_traces(
            marker=dict(size=8, opacity=0.7, line=dict(width=1, color="white"))
        )
    else:
        fig = px.scatter(
            embedding_df,
            x="UMAP1",
            y="UMAP2",
            hover_data=["name", "recipe"],
            title=title,
            width=900,
            height=700,
        )
        fig.update_traces(
            marker=dict(size=8, opacity=0.7, line=dict(width=1, color="white"))
        )

    fig.show()

    return fig


def search_and_highlight(embedding_df: pd.DataFrame, search_term: str, title: str):
    """Search for recipes by name substring and create highlighted plot.

    Args:
        embedding_df: DataFrame with UMAP coordinates and recipe info
        search_term: Substring to search for in recipe names (case-insensitive)
        title: Base title for the plot

    Returns:
        Figure with highlighted matches
    """
    return plot_umap_embedding(embedding_df, title, highlight_search=search_term)


def main(highlight_search: str = None, color_by_category: bool = True):
    """Main analysis pipeline.

    Args:
        highlight_search: Optional substring to highlight in all plots (case-insensitive)
        color_by_category: Whether to color points by recipe category
    """
    print("Loading data from database...")
    recipes_df = load_recipes_from_db(DB_PATH)
    ingredients_df = load_ingredients_hierarchy(DB_PATH)

    print(
        f"Loaded {recipes_df['recipe_id'].nunique()} recipes and {len(ingredients_df)} ingredients"
    )
    print(f"Using exponential distance base: {DISTANCE_BASE}")
    if highlight_search:
        print(f'Highlighting recipes matching: "{highlight_search}"')

    # Load recipe categories
    recipe_classes = load_recipe_classes(CLASSES_PATH)
    print(f"Loaded {len(recipe_classes)} recipe categories")
    if recipe_classes:
        total_categorized = sum(len(recipes) for recipes in recipe_classes.values())
        print(f"Total categorized recipes: {total_categorized}")
        print(f"Categories: {list(recipe_classes.keys())}")

    # Build recipe-ingredient matrices
    print("\nBuilding recipe-ingredient matrices...")
    normalized_matrix, boolean_matrix = build_recipe_ingredient_matrix(recipes_df)
    print(
        f"Matrix shape: {normalized_matrix.shape[0]} recipes x {normalized_matrix.shape[1]} ingredients"
    )

    # Calculate pairwise distances
    print("\nCalculating recipe distance matrices...")

    # Manhattan distance on normalized amounts
    manhattan_df = calculate_manhattan_distance(normalized_matrix)
    distance_matrix_manhattan = manhattan_df.values

    # Debug: dump Manhattan distance matrix
    print(
        f"\nDEBUG: Manhattan Distance Matrix Shape: {distance_matrix_manhattan.shape}"
    )
    print(f"Distance matrix min: {distance_matrix_manhattan.min():.4f}")
    print(f"Distance matrix max: {distance_matrix_manhattan.max():.4f}")
    print(f"Distance matrix mean: {distance_matrix_manhattan.mean():.4f}")

    # Save Manhattan distance matrix to CSV for inspection
    manhattan_df.to_csv("manhattan_distance_matrix.csv")
    print("Manhattan distance matrix saved to 'manhattan_distance_matrix.csv'")

    # Show a sample of the distance matrix (first 10x10)
    print("\nFirst 10x10 section of Manhattan distance matrix:")
    print(manhattan_df.iloc[:10, :10].round(4))

    # Weighted Manhattan distance considering ingredient substitutability
    print("\nCalculating weighted Manhattan distance with substitution levels...")
    weighted_manhattan_df = calculate_weighted_ingredient_distance(
        recipes_df, substitution_weight=0.1
    )
    weighted_distance_matrix_manhattan = weighted_manhattan_df.values

    # Debug: dump weighted Manhattan distance matrix
    print(
        f"\nDEBUG: Weighted Manhattan Distance Matrix Shape: {weighted_distance_matrix_manhattan.shape}"
    )
    print(
        f"Weighted distance matrix min: {weighted_distance_matrix_manhattan.min():.4f}"
    )
    print(
        f"Weighted distance matrix max: {weighted_distance_matrix_manhattan.max():.4f}"
    )
    print(
        f"Weighted distance matrix mean: {weighted_distance_matrix_manhattan.mean():.4f}"
    )

    # Save weighted Manhattan distance matrix to CSV
    weighted_manhattan_df.to_csv("weighted_manhattan_distance_matrix.csv")
    print(
        "Weighted Manhattan distance matrix saved to 'weighted_manhattan_distance_matrix.csv'"
    )

    # Show a sample of the weighted distance matrix (first 10x10)
    print("\nFirst 10x10 section of weighted Manhattan distance matrix:")
    print(weighted_manhattan_df.iloc[:10, :10].round(4))

    # Create UMAP embeddings from distance matrices
    print("\nCreating UMAP embeddings from distance matrices...")

    # Manhattan distance embedding
    embedding_manhattan = create_umap_embedding(
        distance_matrix_manhattan, metric="precomputed"
    )
    embedding_manhattan_df = pd.DataFrame(
        embedding_manhattan, columns=["UMAP1", "UMAP2"], index=normalized_matrix.index
    )
    embedding_manhattan_df["name"] = embedding_manhattan_df.index
    embedding_manhattan_df["recipe"] = embedding_manhattan_df["name"].apply(
        lambda x: get_recipe_string(x, normalized_matrix)
    )

    # Add category information if available
    if recipe_classes and color_by_category:
        embedding_manhattan_df["category"] = assign_recipe_categories(
            embedding_manhattan_df["name"].tolist(), recipe_classes
        )
        color_column = "category"
    else:
        color_column = None

    plot_umap_embedding(
        embedding_manhattan_df,
        "UMAP Embedding - Volumetric Manhattan Distance",
        color_by=color_column,
        highlight_search=highlight_search,
    )

    # Weighted Manhattan distance embedding
    embedding_weighted_manhattan = create_umap_embedding(
        weighted_distance_matrix_manhattan, metric="precomputed"
    )
    embedding_weighted_manhattan_df = pd.DataFrame(
        embedding_weighted_manhattan,
        columns=["UMAP1", "UMAP2"],
        index=normalized_matrix.index,
    )
    embedding_weighted_manhattan_df["name"] = embedding_weighted_manhattan_df.index
    embedding_weighted_manhattan_df["recipe"] = embedding_weighted_manhattan_df[
        "name"
    ].apply(lambda x: get_recipe_string(x, normalized_matrix))

    # Add category information if available
    if recipe_classes and color_by_category:
        embedding_weighted_manhattan_df["category"] = assign_recipe_categories(
            embedding_weighted_manhattan_df["name"].tolist(), recipe_classes
        )
        weighted_color_column = "category"
    else:
        weighted_color_column = None

    plot_umap_embedding(
        embedding_weighted_manhattan_df,
        "UMAP Embedding - Weighted Manhattan Distance (Substitution-Aware)",
        color_by=weighted_color_column,
        highlight_search=highlight_search,
    )

    # Perform NMF with chosen number of components
    print(f"\nPerforming NMF with {N_COMPONENTS} components...")
    W, H, nmf_model = perform_nmf_analysis(normalized_matrix, N_COMPONENTS)
    print(f"Reconstruction error: {nmf_model.reconstruction_err_:.4f}")

    # Create UMAP embedding from NMF factors
    print("\nCreating UMAP embedding from NMF factors...")
    embedding_nmf = create_umap_embedding(W, metric="cosine")

    embedding_nmf_df = pd.DataFrame(
        embedding_nmf, columns=["UMAP1", "UMAP2"], index=normalized_matrix.index
    )
    embedding_nmf_df["name"] = embedding_nmf_df.index
    embedding_nmf_df["recipe"] = embedding_nmf_df["name"].apply(
        lambda x: get_recipe_string(x, normalized_matrix)
    )

    # Add category information if available
    if recipe_classes and color_by_category:
        embedding_nmf_df["category"] = assign_recipe_categories(
            embedding_nmf_df["name"].tolist(), recipe_classes
        )
        nmf_color_column = "category"
    else:
        nmf_color_column = None

    plot_umap_embedding(
        embedding_nmf_df,
        "NMF Recipe Clusters - UMAP Visualization",
        color_by=nmf_color_column,
        highlight_search=highlight_search,
    )

    print("\nAnalysis complete!")

    return {
        "normalized_matrix": normalized_matrix,
        "boolean_matrix": boolean_matrix,
        "W": W,
        "H": H,
        "embeddings": {
            "manhattan": embedding_manhattan_df,
            "weighted_manhattan": embedding_weighted_manhattan_df,
            "nmf": embedding_nmf_df,
        },
    }


if __name__ == "__main__":
    # Option 1: Run with category coloring and highlighting
    # results = main(highlight_search="negroni", color_by_category=True)

    # Option 2: Run with category coloring but no highlighting
    results = main(color_by_category=True)

    # Option 3: Run without category coloring or highlighting, then search specific embeddings
    # results = main(color_by_category=False)
    # search_and_highlight(results["embeddings"]["nmf"], "negroni", "NMF Recipe Clusters")
    # search_and_highlight(results['embeddings']['nmf'], 'sour', 'NMF Recipe Clusters')
    # search_and_highlight(results['embeddings']['manhattan'], 'old fashioned', 'Manhattan Distance')
    # search_and_highlight(results['embeddings']['weighted_manhattan'], 'whiskey', 'Weighted Manhattan Distance')
