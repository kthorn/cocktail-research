import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ot
import pandas as pd
from tqdm.auto import tqdm
import numpy as np


def build_ingredient_tree(
    df,
    id_col: str = "ingredient_id",
    name_col: str = "ingredient_name",
    path_col: str = "ingredient_path",
    weight_col: str = "substitution_level",
    root_id: str = "root",
    root_name: str = "root",
    default_edge_weight: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, Tuple[Optional[str], float]]]:
    """
    Build a D3-compatible ingredient hierarchy tree and a parent map for weighted distances.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ingredient information, including IDs, names, hierarchical paths, and optional edge weights.
    id_col : str, default "ingredient_id"
        Name of the column containing ingredient node IDs.
    name_col : str, default "ingredient_name"
        Name of the column containing ingredient names.
    path_col : str, default "ingredient_path"
        Name of the column containing hierarchical ingredient paths (e.g., '/1/10/17/').
    weight_col : str, default "substitution_level"
        Name of the column specifying optional edge weights (between parent and child in the hierarchy).
    root_id : str, default "root"
        ID assigned to the artificial root node.
    root_name : str, default "root"
        Name assigned to the artificial root node.
    default_edge_weight : float, default 1.0
        Default substitution or edge weight if no value is found in the DataFrame.
        In particular, root to first child edge weight is default_edge_weight.

    Returns
    -------
    tree_dict : dict
        Nested dictionary representing the tree structure, suitable for D3 visualization. Each node has:
            - "id": str, node ID
            - "name": str, node or ingredient name
            - "children": list of child node dicts
            - each child node dict may include "edge_weight" (for the parent to child edge)
    parent_map : dict
        Mapping of node IDs (str) to tuples of (parent_id: Optional[str], edge_weight: float).
        For the root node, parent is None and edge_weight is 0.0.

    Examples
    --------
    >>> tree, parent_map = build_ingredient_tree(df)
    >>> root = tree['id']
    >>> parent_map[root]
    (None, 0.0)
    """
    # Pre-index names/weights by id (if present in df)
    name_by_id = {}
    weight_by_id = {}
    for _, row in df[[id_col, name_col, weight_col]].iterrows():
        cid = str(row[id_col])
        if cid not in name_by_id and isinstance(row[name_col], str):
            name_by_id[cid] = row[name_col]
        # weight applies to edge (parent -> this node)
        if row.get(weight_col) is not None and not (
            isinstance(row[weight_col], float) and math.isnan(row[weight_col])
        ):
            weight_by_id[cid] = float(row[weight_col])

    # Collect nodes and edges from paths
    nodes: Dict[str, Dict[str, Any]] = {}
    children_map: Dict[str, set] = {}
    edge_w: Dict[Tuple[str, str], float] = {}
    parent_map: Dict[str, Tuple[Optional[str], float]] = {}

    def ensure_node(nid: str):
        if nid not in nodes:
            nodes[nid] = {"id": nid, "name": name_by_id.get(nid, nid)}
            children_map.setdefault(nid, set())

    # Create implicit root
    ensure_node(root_id)
    nodes[root_id]["name"] = root_name
    parent_map[root_id] = (None, 0.0)

    # Build structure from each path
    for _, row in df[[id_col, path_col]].iterrows():
        path = str(row[path_col]).strip()
        # split "/1/10/17/" -> ["1","10","17"]
        parts = [p for p in path.split("/") if p]
        if not parts:
            continue

        # Link root -> first
        prev = root_id
        for idx, raw_id in enumerate(parts):
            nid = str(raw_id)
            ensure_node(nid)
            # connect prev -> nid
            if nid not in children_map[prev]:
                children_map[prev].add(nid)
            # assign edge weight (default first, overridden later if available)
            if (prev, nid) not in edge_w:
                edge_w[(prev, nid)] = default_edge_weight
            # set parent map if first time
            if nid not in parent_map:
                parent_map[nid] = (prev, edge_w[(prev, nid)])
            prev = nid

    # Now apply any specific weights from the df to override defaults
    # (weight is for edge parent->node; we need the immediate parent from parent_map)
    for child_id, (parent_id, _) in list(parent_map.items()):
        if parent_id is None:
            continue
        if child_id in weight_by_id:
            w = weight_by_id[child_id]
            edge_w[(parent_id, child_id)] = w
            parent_map[child_id] = (parent_id, w)

    # Build nested dict recursively for D3
    def build_subtree(pid: str) -> Dict[str, Any]:
        node = {"id": pid, "name": nodes[pid]["name"]}
        kids = []
        for cid in children_map.get(pid, []):
            child = build_subtree(cid)
            # attach edge weight on the child (meaning parent->child)
            child["edge_weight"] = edge_w.get((pid, cid), default_edge_weight)
            kids.append(child)
        if kids:
            node["children"] = kids
        return node

    tree_dict = build_subtree(root_id)
    return tree_dict, parent_map


def weighted_distance(
    u: str | int,
    v: str | int,
    parent_map: Dict[str, Tuple[Optional[str], float]],
) -> float:
    """
    Compute the weighted distance between two nodes in a tree.

    The distance is defined as the sum of edge weights from node `u` to their
    lowest common ancestor (LCA) and from node `v` to the same LCA, using the
    `parent_map` produced by `build_tree_for_d3`.

    Parameters
    ----------
    u : str or int
        The node ID or name of the first node.
    v : str or int
        The node ID or name of the second node.
    parent_map : dict of str to tuple (str or None, float)
        Dictionary mapping each node (as a string) to its parent and the edge weight
        connecting it (parent_id, edge_weight). Root nodes have parent_id as None.

    Returns
    -------
    float
        Weighted distance between `u` and `v` along the tree structure.

    Raises
    ------
    KeyError
        If the two nodes do not share a common ancestor (i.e., the input is not a tree
        or the nodes are not connected in the tree).
    """
    # Normalize inputs to match parent_map keys (which are strings)
    u_key = str(u)
    v_key = str(v)
    # Ancestors of u with cumulative cost to reach them
    anc_cost = {}
    cur, acc = u_key, 0.0
    while cur is not None:
        anc_cost[cur] = acc
        p, w = parent_map.get(cur, (None, 0.0))
        cur, acc = p, acc + (w if p is not None else 0.0)

    # Walk up from v until we hit an ancestor of u
    cur, acc = v_key, 0.0
    while cur is not None:
        if cur in anc_cost:
            return anc_cost[cur] + acc
        p, w = parent_map.get(cur, (None, 0.0))
        cur, acc = p, acc + (w if p is not None else 0.0)

    raise KeyError(
        f"Nodes do not share a common ancestor (is it a tree?). u={u_key}, v={v_key}"
    )


def build_ingredient_distance_matrix(
    parent_map: Dict[str, Tuple[Optional[str], float]],
) -> Tuple[np.ndarray, dict[str, int]]:
    """
    Build a pairwise distance matrix for all ingredients in the tree.

    Computes the weighted distance between every pair of nodes (ingredients) in the tree,
    using the parent_map produced by `build_ingredient_tree`. The result is a symmetric
    matrix where each entry (i, j) represents the weighted distance between nodes with
    identifiers corresponding to those indices in the id_to_index dictionary.

    Parameters
    ----------
    parent_map : dict of str to tuple (str or None, float)
        Dictionary mapping each node (ingredient ID as a string) to a tuple of (parent_id, edge_weight),
        where parent_id is the parent node ID (or None for the root),
        and edge_weight is the cost to reach that parent.

    Returns
    -------
    distance_matrix : np.ndarray
        A symmetric 2D array of shape (n, n) where n is the number of nodes in parent_map.
        Each entry (i, j) is the weighted tree distance between nodes i and j.
    id_to_index : dict of str to int
        Mapping from node/ingredient ID (as a string) to its corresponding index in the matrix.

    Notes
    -----
    The node IDs/keys in parent_map are used as matrix indices via id_to_index.
    The function relies on `weighted_distance` to compute tree distances.
    """
    ingredient_ids = list(parent_map.keys())
    id_to_index = {id: i for i, id in enumerate(ingredient_ids)}
    distance_matrix = np.zeros((len(ingredient_ids), len(ingredient_ids)))
    for i in range(len(ingredient_ids)):
        for j in range(i + 1, len(ingredient_ids)):
            distance_matrix[i, j] = weighted_distance(
                ingredient_ids[i], ingredient_ids[j], parent_map
            )
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix, id_to_index


def build_recipe_volume_matrix(
    recipes_df: pd.DataFrame,
    ingredient_id_to_index: dict[str, int],
    recipe_id_col: str = "recipe_id",
    ingredient_id_col: str = "ingredient_id",
    volume_col: str = "volume_fraction",
    volume_error_tolerance: float = 1e-6,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Construct a matrix of recipe ingredient volume fractions.

    Builds a matrix of shape (n_recipes, m_ingredients) where each entry [i, j] is
    the volume fraction of ingredient j in recipe i from the supplied DataFrame.

    Parameters
    ----------
    recipes_df : pd.DataFrame
        DataFrame containing at least recipe IDs, ingredient IDs, and volume fractions.
    ingredient_id_to_index : dict[str, int]
        Mapping from ingredient IDs to matrix columns.
    recipe_id_col : str, optional
        Column name for recipe IDs. Default is "recipe_id".
    ingredient_id_col : str, optional
        Column name for ingredient IDs. Default is "ingredient_id".
    volume_col : str, optional
        Column name for the ingredient volume fraction in the recipe. Default is "volume_fraction".
    volume_error_tolerance : float, optional
        Tolerance for checking that all rows of volume_matrix sum to 1. Default is 1e-6.

    Returns
    -------
    volume_matrix : np.ndarray
        Array of shape (n_recipes, m_ingredients); entry [i, j] is the volume fraction of ingredient j in recipe i.
        Rows correspond to recipes as dictated by recipe_id_to_index;
        columns to ingredients as dictated by ingredient_id_to_index.
    recipe_id_to_index : dict[str, int]
        Mapping from recipe IDs to matrix rows.

    Raises
    ------
    ValueError
        If the volume fraction column is missing or contains NaNs.

    Notes
    -----
    If a recipe does not include an ingredient, the corresponding matrix entry will be zero.
    Each row sums to at most 1, depending on whether all ingredient fractions for a recipe are included.
    """
    # Validate presence of volume_fraction and ensure no NaNs
    if volume_col not in recipes_df.columns:
        raise ValueError("recipes_df must contain a 'volume_fraction' column")
    if recipes_df[volume_col].isna().any():
        raise ValueError(
            f"recipes_df['{volume_col}'] contains NaNs; please clean first"
        )

    recipe_ids = list(recipes_df[recipe_id_col].unique())
    recipe_id_to_index = {str(id): i for i, id in enumerate(recipe_ids)}
    volume_matrix = np.zeros((len(recipe_ids), len(ingredient_id_to_index)))
    for _, row in recipes_df.iterrows():
        recipe_index = recipe_id_to_index[str(row[recipe_id_col])]
        ingredient_index = ingredient_id_to_index[str(row[ingredient_id_col])]
        volume_matrix[recipe_index, ingredient_index] = float(row[volume_col])
    # Check that all rows of volume_matrix sum to 1 within numerical error
    row_sums = volume_matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=volume_error_tolerance):
        bad_rows = np.where(~np.isclose(row_sums, 1.0, atol=volume_error_tolerance))[0]
        bad_recipe_ids = [str(recipe_ids[i]) for i in bad_rows]
        raise ValueError(
            f"Not all rows of volume_matrix sum to 1. "
            f"Offending rows: {bad_rows}. \n"
            f"Row sums: {row_sums[bad_rows]}; recipe ids: {bad_recipe_ids}"
        )
    return volume_matrix, recipe_id_to_index


def compute_emd(
    a: np.ndarray,
    b: np.ndarray,
    cost_matrix: np.ndarray,
    return_plan: bool = False,
    support_idx: Optional[np.ndarray] = None,
) -> Union[float, Tuple[float, List[Tuple[int, int, float, float]]]]:
    """
    Compute the Earth Mover's Distance (EMD) between two distributions a and b.

    Parameters
    ----------
    a : np.ndarray
        Source distribution, shape (n,)
    b : np.ndarray
        Target distribution, shape (n,)
    cost_matrix : np.ndarray
        Cost matrix, shape (n, n)
    return_plan : bool, optional
        If True, also return the transport plan as a list of flows, by default False.

    Returns
    -------
    distance : float
        The Earth Mover's Distance (total minimum cost) between the two distributions.
    transport_plan : list[tuple[int, int, float, float]]
        Only returned if return_plan is True.
        Each tuple is (from_idx, to_idx, amount, cost):
            - from_idx: Index in source a
            - to_idx: Index in target b
            - amount: mass transported (typically between 0 and 1)
            - cost: amount * per-unit cost for this flow
    """
    n_ingredients = a.shape[0]
    if len(b) != n_ingredients:
        raise ValueError(f"b must have {n_ingredients} ingredients")
    if cost_matrix.shape[0] != n_ingredients or cost_matrix.shape[1] != n_ingredients:
        raise ValueError(
            f"cost_matrix must be of shape ({n_ingredients}, {n_ingredients})"
        )

    # Reduce to the union of supports to dramatically shrink problem size
    if support_idx is None:
        support_idx = np.nonzero((a > 0) | (b > 0))[0]

    if support_idx.size == 0:
        return 0.0 if not return_plan else (0.0, [])

    a_sub = a[support_idx]
    b_sub = b[support_idx]
    cost_sub = cost_matrix[np.ix_(support_idx, support_idx)]

    if not return_plan:
        # Use ot.emd2 when only the objective value is needed (faster than full plan)
        distance = float(ot.emd2(a_sub, b_sub, cost_sub))
        return distance
    else:
        transport_matrix = ot.emd(a_sub, b_sub, cost_sub)
        distance = float(np.sum(transport_matrix * cost_sub))
        transport_plan: List[Tuple[int, int, float, float]] = []
        rows, cols = np.nonzero(transport_matrix > 1e-10)
        for ii, jj in zip(rows, cols):
            flow = float(transport_matrix[ii, jj])
            flow_cost = float(flow * cost_sub[ii, jj])
            # Map back to original indices via support_idx
            transport_plan.append(
                (int(support_idx[ii]), int(support_idx[jj]), flow, flow_cost)
            )
        return distance, transport_plan


def emd_matrix(
    volume_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Compute the Earth Mover's Distance matrix between all recipes in the volume matrix.
    """
    n_recipes = volume_matrix.shape[0]
    emd_matrix = np.zeros((n_recipes, n_recipes))

    # Precompute supports for each recipe to avoid repeated nonzero scans
    supports: List[np.ndarray] = [
        np.nonzero(volume_matrix[i] > 0)[0] for i in range(n_recipes)
    ]

    if n_jobs == 1:
        for i in tqdm(range(n_recipes), desc="Computing EMD matrix"):
            for j in range(i + 1, n_recipes):
                union_idx = np.union1d(supports[i], supports[j])
                distance = compute_emd(
                    volume_matrix[i],
                    volume_matrix[j],
                    cost_matrix,
                    return_plan=False,
                    support_idx=union_idx,
                )
                emd_matrix[i, j] = distance
                emd_matrix[j, i] = distance
        return emd_matrix

    # Parallel path (shared memory threads to avoid copying large matrices)
    try:
        from joblib import Parallel, delayed
    except ImportError:
        # Fallback to sequential if joblib is not available
        for i in tqdm(range(n_recipes), desc="Computing EMD matrix"):
            for j in range(i + 1, n_recipes):
                union_idx = np.union1d(supports[i], supports[j])
                distance = compute_emd(
                    volume_matrix[i],
                    volume_matrix[j],
                    cost_matrix,
                    return_plan=False,
                    support_idx=union_idx,
                )
                emd_matrix[i, j] = distance
                emd_matrix[j, i] = distance
        return emd_matrix

    pairs: List[Tuple[int, int]] = [
        (i, j) for i in range(n_recipes) for j in range(i + 1, n_recipes)
    ]

    def _pair_distance(i: int, j: int) -> Tuple[int, int, float]:
        union_idx = np.union1d(supports[i], supports[j])
        d = compute_emd(
            volume_matrix[i],
            volume_matrix[j],
            cost_matrix,
            return_plan=False,
            support_idx=union_idx,
        )
        return i, j, float(d)

    results = Parallel(n_jobs=n_jobs, prefer="threads", require="sharedmem")(
        delayed(_pair_distance)(i, j) for (i, j) in pairs
    )
    for i, j, d in results:
        emd_matrix[i, j] = d
        emd_matrix[j, i] = d
    return emd_matrix


def knn_matrix(
    distance_matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Compute the k-nearest neighbors (kNN) indices and distances from a distance matrix.

    Given a symmetric pairwise distance matrix, this function finds the indices and
    corresponding distances of the k nearest neighbors (excluding self) for each row.

    Parameters
    ----------
    distance_matrix : np.ndarray
        A 2D array of shape (n, n) representing pairwise distances, where n is the number of samples.
    k : int
        The number of nearest neighbors to select for each item.

    Returns
    -------
    nn_idx : np.ndarray
        Array of shape (n, k) with indices of the k nearest neighbors for each item.
    nn_dist : np.ndarray
        Array of shape (n, k) with the distances to the k nearest neighbors for each item.

    Notes
    -----
    The diagonal (self-distances) and any non-finite values (NaN/-Inf) are replaced with +Inf
    so they are not selected as neighbors. The neighbor selection uses `np.argsort`;
    in the case of ties, the order is determined by the index order.
    """
    dmat = distance_matrix.copy()
    # Replace diagonal, NaN/-Inf with +Inf so they sort to the end
    np.fill_diagonal(dmat, np.inf)
    non_finite_mask = ~np.isfinite(dmat)
    if non_finite_mask.any():
        dmat[non_finite_mask] = np.inf
    nn_idx = np.argsort(dmat, axis=1)[:, :k]
    nn_dist = np.take_along_axis(dmat, nn_idx, axis=1)
    return nn_idx, nn_dist
