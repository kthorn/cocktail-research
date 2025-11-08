"""
Recipe K-Medoids Clustering and UMAP Analysis from SQLite Database

This script extracts recipes from a SQLite database, calculates weighted distances
based on ingredient substitutability, performs k-medoids clustering, and visualizes
results with UMAP embeddings.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import umap
from sklearn_extra.cluster import KMedoids
import plotly.graph_objects as go
from typing import Dict, List, Tuple

# Import distance calculation utilities
from distance_calculations import (
    build_recipe_ingredient_matrix,
    calculate_weighted_ingredient_distance,
)


# Configuration
DB_PATH = "backup-2025-10-17_08-00-45.db"
CLASSES_PATH = "recipe_classes.json"  # Path to recipe categories JSON file
N_CLUSTERS = 30  # Number of k-means clusters
UMAP_N_NEIGHBORS = 5
UMAP_RANDOM_STATE = 42
UMAP_MIN_DIST = 0.05
SUBSTITUTION_WEIGHT = 0.1  # Weight for substitutable ingredients


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


def get_recipe_string(recipe_name: str, normalized_matrix: pd.DataFrame) -> str:
    """Get formatted recipe string showing ingredient proportions."""
    recipe = normalized_matrix.loc[recipe_name, :]
    ingredients = recipe[recipe > 0].sort_values(ascending=False)

    if len(ingredients) == 0:
        return "No ingredients found"

    recipe_parts = [f"{ing}: {prop:.3f}" for ing, prop in ingredients.items()]
    return " | ".join(recipe_parts)


def perform_kmedoids_clustering(
    distance_matrix: np.ndarray, n_clusters: int = N_CLUSTERS, random_state: int = 42
) -> Tuple[np.ndarray, KMedoids]:
    """Perform k-medoids clustering directly on distance matrix.

    Args:
        distance_matrix: Pairwise distance matrix between recipes
        n_clusters: Number of clusters to create
        random_state: Random state for reproducibility

    Returns:
        Tuple of (cluster_labels, kmedoids_model)
    """
    print(f"Performing k-medoids clustering with {n_clusters} clusters...")
    kmedoids = KMedoids(
        n_clusters=n_clusters,
        metric="precomputed",
        random_state=random_state,
        init="k-medoids++",
    )
    cluster_labels = kmedoids.fit_predict(distance_matrix)

    return cluster_labels, kmedoids


def create_umap_embedding(
    distance_matrix: np.ndarray, metric: str = "precomputed"
) -> np.ndarray:
    """Create UMAP embedding from distance matrix."""
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric=metric,
        random_state=UMAP_RANDOM_STATE,
    )

    embedding = reducer.fit_transform(distance_matrix)
    return embedding


def plot_clustered_umap(
    embedding_df: pd.DataFrame,
    title: str,
    cluster_column: str = "cluster",
    category_column: str = "category",
):
    """Create interactive Plotly scatter plot with clusters (colors) and categories (shapes).

    Args:
        embedding_df: DataFrame with UMAP coordinates, clusters, and categories
        title: Plot title
        cluster_column: Column name for cluster assignments (will be colored)
        category_column: Column name for labeled categories (will be shapes)
    """
    # Define shape mapping for categories (filled shapes)
    shape_map = {
        "daiquiri": "circle",
        "manhattan": "square",
        "boulevardier": "diamond",
        "sidecar": "star",
        "jungle bird": "triangle-up",
        "other": "hexagon",
    }

    # Define colors for clusters (using a qualitative palette)
    import plotly.express as px

    cluster_colors = px.colors.qualitative.Set2

    fig = go.Figure()

    # Get unique clusters and categories
    clusters = sorted(embedding_df[cluster_column].unique())
    categories = sorted(embedding_df[category_column].unique())

    # Create a trace for each combination of cluster and category
    for cluster in clusters:
        for category in categories:
            mask = (embedding_df[cluster_column] == cluster) & (
                embedding_df[category_column] == category
            )
            subset = embedding_df[mask]

            if len(subset) == 0:
                continue

            color = cluster_colors[cluster % len(cluster_colors)]
            shape = shape_map.get(category, "circle-open")

            # Create legend group by cluster
            legend_group = f"cluster_{cluster}"
            show_legend = (
                category == categories[0]
            )  # Only show one legend entry per cluster

            fig.add_trace(
                go.Scatter(
                    x=subset["UMAP1"],
                    y=subset["UMAP2"],
                    mode="markers",
                    marker=dict(
                        size=10,
                        opacity=0.8,
                        color=color,
                        symbol=shape,
                        line=dict(width=1, color="white"),
                    ),
                    text=subset["name"],
                    customdata=subset[["recipe", category_column]],
                    hovertemplate="<b>%{text}</b><br>Cluster: "
                    + str(cluster)
                    + "<br>Category: %{customdata[1]}<br>%{customdata[0]}<extra></extra>",
                    name=f"Cluster {cluster}",
                    legendgroup=legend_group,
                    showlegend=show_legend,
                )
            )

    # Add a separate legend for shapes/categories
    fig.update_layout(
        title=title,
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        width=1000,
        height=800,
        legend=dict(
            title="K-Medoids Clusters",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        annotations=[
            dict(
                text="<b>Shapes:</b><br>"
                + "<br>".join(
                    [
                        f"● {cat} = {shape_map.get(cat, 'circle-open')}"
                        for cat in categories
                    ]
                ),
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.3,
                xanchor="left",
                yanchor="top",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.9,
            )
        ],
    )

    fig.show()

    return fig


def main(n_clusters: int = N_CLUSTERS):
    """Main analysis pipeline.

    Args:
        n_clusters: Number of k-means clusters to create
    """
    print("Loading data from database...")
    recipes_df = load_recipes_from_db(DB_PATH)

    print(f"Loaded {recipes_df['recipe_id'].nunique()} recipes")

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

    # Calculate weighted Manhattan distance
    print("\nCalculating weighted Manhattan distance with substitution levels...")
    weighted_manhattan_df = calculate_weighted_ingredient_distance(
        recipes_df, normalized_matrix, substitution_weight=SUBSTITUTION_WEIGHT
    )
    weighted_distance_matrix = weighted_manhattan_df.values

    print(f"\nWeighted distance matrix shape: {weighted_distance_matrix.shape}")
    print(f"Distance matrix min: {weighted_distance_matrix.min():.4f}")
    print(f"Distance matrix max: {weighted_distance_matrix.max():.4f}")
    print(f"Distance matrix mean: {weighted_distance_matrix.mean():.4f}")

    # Perform k-medoids clustering
    print(f"\nPerforming k-medoids clustering with {n_clusters} clusters...")
    cluster_labels, kmedoids_model = perform_kmedoids_clustering(
        weighted_distance_matrix, n_clusters=n_clusters
    )

    # Print cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("\nCluster sizes:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} recipes")

    # Create UMAP embedding from weighted distance matrix
    print("\nCreating UMAP embedding from weighted distance matrix...")
    embedding = create_umap_embedding(weighted_distance_matrix, metric="precomputed")

    # Create DataFrame with embedding, clusters, and categories
    embedding_df = pd.DataFrame(
        embedding, columns=["UMAP1", "UMAP2"], index=normalized_matrix.index
    )
    embedding_df["name"] = embedding_df.index
    embedding_df["recipe"] = embedding_df["name"].apply(
        lambda x: get_recipe_string(x, normalized_matrix)
    )
    embedding_df["cluster"] = cluster_labels

    # Add category information
    if recipe_classes:
        embedding_df["category"] = assign_recipe_categories(
            embedding_df["name"].tolist(), recipe_classes
        )
    else:
        embedding_df["category"] = "other"

    # Plot the results
    plot_clustered_umap(
        embedding_df,
        f"UMAP Embedding - Weighted Distance with K-Medoids Clustering (k={n_clusters})",
        cluster_column="cluster",
        category_column="category",
    )

    print("\nAnalysis complete!")

    return {
        "normalized_matrix": normalized_matrix,
        "weighted_distance_matrix": weighted_distance_matrix,
        "embedding_df": embedding_df,
        "cluster_labels": cluster_labels,
        "kmedoids_model": kmedoids_model,
    }


if __name__ == "__main__":
    # Run analysis with default number of clusters
    results = main(n_clusters=N_CLUSTERS)

    # Optionally try different numbers of clusters
    # results = main(n_clusters=6)
    # results = main(n_clusters=10)
