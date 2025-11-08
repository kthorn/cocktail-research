# ot_cost_learner.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, Any
import ot
from tqdm.auto import tqdm

Array = np.ndarray

# ----------------------------- Utilities -----------------------------


def _symmetrize(C: Array) -> Array:
    return 0.5 * (C + C.T)


def _project_cost_matrix(
    C: Array,
    clamp: Tuple[float, float],
    zero_diag: bool = True,
    enforce_symmetry: bool = True,
) -> Array:
    if enforce_symmetry:
        C = _symmetrize(C)
    if zero_diag:
        np.fill_diagonal(C, 0.0)
    if clamp is not None:
        lo, hi = clamp
        C = np.clip(C, lo, hi)
    return C


def _fix_dummy(C: Array, dummy_cost: float) -> Array:
    """Locks last row/col to dummy_cost and sets dummy self-cost to 0."""
    C = C.copy()
    C[:, -1] = dummy_cost
    C[-1, :] = dummy_cost
    C[-1, -1] = 0.0
    return C


def _median_rescale(C: Array, target: float = 1.0) -> Array:
    mask = ~np.eye(C.shape[0], dtype=bool)
    med = np.median(C[mask])
    scale = (med / (target + 1e-12)) if med > 0 else 1.0
    return C / (scale + 1e-12)


def _augment_with_dummy(A: Array, dummy_mass: float = 0.0) -> Array:
    """Append a dummy dimension; if dummy_mass=0, just append zeros and renormalize."""
    n, m = A.shape
    pad = np.full((n, 1), dummy_mass, dtype=A.dtype)
    V = np.hstack([A, pad])
    # Renormalize just in case; typically A rows sum to 1 already.
    V /= V.sum(axis=1, keepdims=True)
    return V


def _sinkhorn_plan(
    a: Array,
    b: Array,
    C: Array,
    eps: float,
    unbalanced_tau: Optional[float],
    max_iter: int = 500,
) -> Array:
    """Returns the transport plan T."""
    if unbalanced_tau is None:
        # Balanced (rows/cols must match)
        T = ot.sinkhorn(a, b, C, reg=eps, numItermax=max_iter)
    else:
        # Unbalanced (KL-penalized marginals) ~ soft dummy
        T = ot.unbalanced.sinkhorn_unbalanced(
            a, b, C, reg=eps, reg_m=unbalanced_tau, numItermax=max_iter
        )
    return T


def _ot_distance(
    a: Array, b: Array, C: Array, eps: float, unbalanced_tau: Optional[float]
) -> Tuple[float, Array]:
    """Returns (distance, plan)."""
    T = _sinkhorn_plan(a, b, C, eps, unbalanced_tau)
    return float((T * C).sum()), T


def _sparsify_plan(T: Array, topk: Optional[int], min_frac: float) -> Array:
    """Keep row-wise top-k and drop tiny dust (fraction of total mass)."""
    if topk is not None and topk > 0:
        idx = np.argpartition(-T, min(topk, T.shape[1]) - 1, axis=1)[:, :topk]
        mask = np.zeros_like(T, dtype=bool)
        rows = np.arange(T.shape[0])[:, None]
        mask[rows, idx] = True
        T = T * mask
    cutoff = max(min_frac * T.sum(), 0.0)
    T = np.where(T >= cutoff, T, 0.0)
    return T


# ----------------------------- Config -----------------------------


@dataclass
class LearnConfig:
    # Neighbour selection
    k_nn: int = 10
    beta: float = 6.0  # Boltzmann weight inverse "temperature"
    # OT
    eps: float = 3e-3
    unbalanced_tau: Optional[float] = (
        None  # set e.g. 0.1 for unbalanced; None for balanced
    )
    sinkhorn_iters: int = 500
    # Plan sparsification
    plan_topk: Optional[int] = 2
    plan_minfrac: float = 1e-3
    # EM loop
    iters: int = 15
    tol: float = 1e-3
    # Cost post-processing
    clamp: Tuple[float, float] = (0.0, 8.0)
    median_target: float = 1.0  # keep median off-diagonal ~ this
    # Prior anchoring (used by both M-steps)
    prior_blend: float = 0.4  # blend toward C0 each M-step
    ema: float = 0.25  # EMA smoothing vs previous C
    # Dummy handling
    use_dummy: bool = False
    dummy_cost: Optional[float] = None  # if None, set to percentile of C0
    dummy_percentile: float = 85.0
    # M-step choice
    mstep: Literal["blosum", "convex"] = "blosum"
    # BLOSUM params
    blosum_alpha: float = 2.0
    # Convex (metric-learning) params
    convex_lambda: float = 1e-2  # regularization (distance to prior)
    convex_closed_form: bool = True  # closed form vs small gradient step
    convex_lr: float = 0.1  # used if closed_form=False


# ----------------------------- Learner -----------------------------


class OTCostLearner:
    """
    Learn an ingredient-ingredient OT cost matrix C from cocktail distributions A
    via EM-style alternating optimization.

    Shapes:
        A_raw: (n_recipes, m_ingredients)
        C0:    (m (+1 if dummy), m (+1 if dummy))

    Returns:
        learned C, plus a training log with deltas & (optionally) distances.
    """

    def __init__(self, A_raw: Array, C0: Array, config: LearnConfig):
        self.cfg = config
        self.n, self.m_raw = A_raw.shape
        self.use_dummy = config.use_dummy
        if self.use_dummy:
            self.A = _augment_with_dummy(A_raw, dummy_mass=0.0)
            assert C0.shape == (self.m_raw + 1, self.m_raw + 1), (
                "With dummy=True, C0 must be (m+1, m+1)."
            )
            self.m = self.m_raw + 1
        else:
            self.A = A_raw
            assert C0.shape == (self.m_raw, self.m_raw), (
                "With dummy=False, C0 must be (m, m)."
            )
            self.m = self.m_raw

        self.C = C0.copy().astype(float)
        np.fill_diagonal(self.C, 0.0)
        self.C0 = C0.copy().astype(float)
        np.fill_diagonal(self.C0, 0.0)

        # Set/derive dummy cost
        if self.use_dummy:
            if self.cfg.dummy_cost is None:
                base = self.C0[:-1, :-1]
                self.dummy_cost = float(np.percentile(base, self.cfg.dummy_percentile))
            else:
                self.dummy_cost = float(self.cfg.dummy_cost)
            self.C = _fix_dummy(self.C, self.dummy_cost)
            self.C0 = _fix_dummy(self.C0, self.dummy_cost)
        else:
            self.dummy_cost = None

        # Keep costs on a sane scale
        self.C = _median_rescale(self.C, target=self.cfg.median_target)

    # ------------------------- Core steps -------------------------

    def _pairwise_knn(self) -> Tuple[Array, Array]:
        """Compute top-k neighbours for each recipe under current C.
        Returns (nn_idx, nn_dist) with shapes (n,k) each."""
        n, k = self.n, self.cfg.k_nn
        # Never request >= n neighbours; exclude self explicitly below
        k = max(1, min(k, n - 1))
        dmat = np.full((n, n), np.inf, dtype=float)
        for i in range(n):
            ai = self.A[i]
            for j in range(i + 1, n):
                d, _ = _ot_distance(
                    ai, self.A[j], self.C, self.cfg.eps, self.cfg.unbalanced_tau
                )
                dmat[i, j] = dmat[j, i] = d
        # Robustness: exclude self and any non-finite distances from consideration
        np.fill_diagonal(dmat, np.inf)
        # Replace NaN/-Inf with +Inf so they sort to the end
        non_finite_mask = ~np.isfinite(dmat)
        if non_finite_mask.any():
            dmat[non_finite_mask] = np.inf

        # If a row has all equal distances, argsort will fall back to index order.
        # That can look like identical neighbour lists across rows. Warn to aid debugging.
        row_min = np.min(dmat, axis=1)
        row_max = np.max(dmat, axis=1)
        if np.allclose(row_min, row_max):
            print(
                "Warning: all pairwise distances equal (or non-finite); neighbour ordering is arbitrary."
            )

        nn_idx = np.argsort(dmat, axis=1)[:, :k]
        nn_dist = np.take_along_axis(dmat, nn_idx, axis=1)
        return nn_idx, nn_dist

    def _e_step(self) -> Tuple[Array, Array]:
        """Compute aggregate expected counts (T_sum) using kNN positives with Boltzmann weights.
        Returns (T_sum, N_pairs)."""
        T_sum = np.zeros((self.m, self.m), dtype=float)
        nn_idx, nn_dist = self._pairwise_knn()
        print("knn complete")
        N_pairs = 0
        for r in range(self.n):
            nbrs = nn_idx[r]
            d = nn_dist[r]
            # Boltzmann weights (closer pairs count more)
            w = np.exp(-self.cfg.beta * (d - d.min()))
            w /= w.sum() + 1e-12
            a = self.A[r]
            for w_rs, s in zip(w, nbrs):
                b = self.A[s]
                T = _sinkhorn_plan(
                    a,
                    b,
                    self.C,
                    self.cfg.eps,
                    self.cfg.unbalanced_tau,
                    self.cfg.sinkhorn_iters,
                )
                # sparsify to avoid teaching from tiny dust
                T = _sparsify_plan(T, self.cfg.plan_topk, self.cfg.plan_minfrac)
                T_sum += w_rs * T
                N_pairs += 1
        # Symmetrize counts (nice but optional)
        T_sum = 0.5 * (T_sum + T_sum.T)
        return T_sum, N_pairs

    def _m_step_blosum(self, T_sum: Array) -> Array:
        """BLOSUM-like log-odds update; convert expected matches to costs."""
        N = T_sum.copy()
        # Laplace smoothing
        alpha = self.cfg.blosum_alpha
        total = N.sum()
        row = N.sum(axis=1, keepdims=True)
        col = N.sum(axis=0, keepdims=True)
        # Expected under independence
        E = ((row + alpha) @ (col + alpha)) / (total + alpha * N.size)
        S = (N + alpha) / (E + 1e-12)
        C_new = -np.log(S + 1e-12)
        C_new = _project_cost_matrix(
            C_new, clamp=self.cfg.clamp, zero_diag=True, enforce_symmetry=True
        )

        # Prior blend and EMA smoothing (stability)
        C_target = (1.0 - self.cfg.prior_blend) * C_new + self.cfg.prior_blend * self.C0
        C_out = (1.0 - self.cfg.ema) * self.C + self.cfg.ema * C_target
        C_out = _project_cost_matrix(
            C_out, clamp=self.cfg.clamp, zero_diag=True, enforce_symmetry=True
        )

        # Keep scale consistent
        C_out = _median_rescale(C_out, self.cfg.median_target)

        # Re-lock dummy if used
        if self.use_dummy:
            C_out = _fix_dummy(C_out, self.dummy_cost)

        return C_out

    def _m_step_convex(self, T_sum: Array) -> Array:
        """
        Convex/metric-learning M-step:
            minimize <C, T_sum> + lambda ||C - C0||^2  (with T frozen)
        Closed-form unconstrained optimum: C* = C0 - (1/(2*lambda)) * T_sum
        Then project and stabilize.
        """
        lam = self.cfg.convex_lambda
        if self.cfg.convex_closed_form:
            C_prop = self.C0 - (T_sum / (2.0 * lam))
        else:
            # small gradient step from current C toward the convex minimum
            grad = T_sum + 2.0 * lam * (self.C - self.C0)
            C_prop = self.C - self.cfg.convex_lr * grad

        C_prop = _project_cost_matrix(
            C_prop, clamp=self.cfg.clamp, zero_diag=True, enforce_symmetry=True
        )

        # Prior blend + EMA for extra stability
        C_target = (
            1.0 - self.cfg.prior_blend
        ) * C_prop + self.cfg.prior_blend * self.C0
        C_out = (1.0 - self.cfg.ema) * self.C + self.cfg.ema * C_target
        C_out = _project_cost_matrix(
            C_out, clamp=self.cfg.clamp, zero_diag=True, enforce_symmetry=True
        )
        C_out = _median_rescale(C_out, self.cfg.median_target)

        if self.use_dummy:
            C_out = _fix_dummy(C_out, self.dummy_cost)

        return C_out

    # ------------------------- Public API --------------------------

    def fit(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run EM iterations. Returns a log dict with deltas and configuration.
        """
        log = {"delta": [], "config": self.cfg}
        prev = self.C.copy()
        for t in tqdm(range(self.cfg.iters), disable=not verbose):
            print(f"e_step {t + 1} of {self.cfg.iters}")
            T_sum, n_pairs = self._e_step()

            print(f"m_step {t + 1} of {self.cfg.iters}")
            if self.cfg.mstep == "blosum":
                self.C = self._m_step_blosum(T_sum)
            elif self.cfg.mstep == "convex":
                self.C = self._m_step_convex(T_sum)
            else:
                raise ValueError("mstep must be 'blosum' or 'convex'.")

            # Convergence check
            num = np.linalg.norm(self.C - prev)
            den = np.linalg.norm(prev) + 1e-12
            delta = num / den
            log["delta"].append(float(delta))
            prev = self.C.copy()

            # Gentle dummy anneal to encourage real matches (optional)
            if self.use_dummy and self.dummy_cost is not None:
                self.dummy_cost *= 0.98
                self.C = _fix_dummy(self.C, self.dummy_cost)

            if verbose:
                print(f"[iter {t + 1:02d}] pairs={n_pairs} delta={delta:.4e}")

            if delta < self.cfg.tol:
                if verbose:
                    print("Converged.")
                break

        return log

    def transform_distance(self, A_other: Optional[Array] = None) -> Array:
        """
        Compute pairwise OT distances between rows of A (or vs A_other) under learned C.
        Returns a dense distance matrix.
        """
        X = self.A
        Y = (
            self.A
            if A_other is None
            else (_augment_with_dummy(A_other) if self.use_dummy else A_other)
        )
        n, m = X.shape[0], Y.shape[0]
        D = np.zeros((n, m), dtype=float)
        for i in range(n):
            for j in range(m):
                D[i, j], _ = _ot_distance(
                    X[i], Y[j], self.C, self.cfg.eps, self.cfg.unbalanced_tau
                )
        return D

    @property
    def cost_matrix(self) -> Array:
        return self.C.copy()


# ------------------------- Example usage -------------------------
if __name__ == "__main__":
    # Fake tiny example to illustrate API
    rng = np.random.default_rng(7)
    n, m = 20, 6  # 20 cocktails, 6 ingredients
    A = rng.random((n, m))
    A /= A.sum(axis=1, keepdims=True)

    # Heuristic prior costs (lower within same "group")
    groups = np.array([0, 0, 1, 1, 2, 2])
    C0 = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            C0[i, j] = 0.3 if groups[i] == groups[j] else 1.2
    np.fill_diagonal(C0, 0.0)

    cfg = LearnConfig(
        k_nn=5,
        beta=6.0,
        eps=3e-3,
        unbalanced_tau=None,  # set e.g. 0.1 to enable unbalanced OT
        plan_topk=2,
        plan_minfrac=1e-3,
        iters=10,
        use_dummy=False,  # flip True to enable explicit dummy
        mstep="blosum",  # or "convex"
        blosum_alpha=2.0,
        convex_lambda=1e-2,
        clamp=(0.0, 8.0),
        median_target=1.0,
        prior_blend=0.4,
        ema=0.25,
    )

    learner = OTCostLearner(A, C0, cfg)
    train_log = learner.fit(verbose=True)
    C_learned = learner.cost_matrix
    print("Learned cost matrix:\n", np.round(C_learned, 3))
