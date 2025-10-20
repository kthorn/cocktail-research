import math
from typing import Dict, Tuple, Any, Optional
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
