"""Selection bias among correlated backtests: certified bounds on E max of correlated z-scores.

When N correlated strategy variants are backtested, the best Sharpe ratio is inflated by
selection. Under the null, the standardised Sharpe estimates are approximately
Z ~ N(0, C), and the Deflated Sharpe Ratio needs E max_i Z_i. Assuming independent trials
overstates it; Monte Carlo needs a factorisation of C and gives no structural insight.

This module applies the comparison method of the paper (Sudakov-Fernique against tree
processes) to an empirical correlation matrix. With the canonical distance
d(i, j) = (E (Z_i - Z_j)^2)^{1/2} = sqrt(2 (1 - C_ij)), every dendrogram shape gives

* a majorant: give each merge the largest distance across its split (raised to its
  children's heights); the resulting ultrametric u satisfies u >= d, so the tree process
  dominates Z in the Sudakov-Fernique sense and E max Z <= E max(tree);
* a minorant: give each merge the smallest distance across its split (lowered to its
  parent's height); then u <= d and E max(tree) <= E max Z, also on any subset of the
  strategies (dropping points can only lower E max Z, but can raise the minorant).

E max of a tree process is computed exactly (up to quadrature) by a CDF recursion over
the dendrogram, without Monte Carlo. The bracket is tight when the correlation structure
is hierarchical (families of related strategies) and loose when it is flat.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats
from scipy.cluster.hierarchy import linkage
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import squareform

from roughmax import theory

EULER_GAMMA = 0.5772156649015329
_GRID_POINTS = 4097


def corr_distance(C: np.ndarray) -> np.ndarray:
    """Canonical distance sqrt(2 (1 - C_ij)) of unit-variance Gaussians."""
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("correlation matrix must be square")
    d = np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return (d + d.T) / 2.0


TOPOLOGIES = ("single", "complete", "average", "weighted", "ward")


def linkage_tree(C: np.ndarray, method: str) -> np.ndarray:
    """Hierarchical clustering of the canonical distance with one of TOPOLOGIES.

    Single and complete linkage heights are already one-sided ultrametric bounds; for the
    other methods only the tree shape is used, through refit_heights.
    """
    if method not in TOPOLOGIES:
        raise ValueError(f"method must be one of {TOPOLOGIES}")
    return linkage(squareform(corr_distance(C), checks=False), method=method)


def _members(Z: np.ndarray, n: int) -> list[np.ndarray]:
    members: list[np.ndarray] = [np.array([i]) for i in range(n)]
    for a, b in Z[:, :2].astype(int):
        members.append(np.concatenate([members[a], members[b]]))
    return members


def refit_heights(Z: np.ndarray, d: np.ndarray, mode: str) -> np.ndarray:
    """Tightest ultrametric bound with the tree shape of Z.

    mode="upper": the merge height of each node is the largest distance across its split,
    raised to its children's heights, so u >= d (a Sudakov-Fernique majorant).
    mode="lower": the merge height is the smallest distance across its split, lowered to
    its parent's height, so u <= d (a minorant).
    """
    Z = np.array(Z, dtype=float)
    n = Z.shape[0] + 1
    members = _members(Z, n)
    across = np.empty(n - 1)
    for k, (a, b) in enumerate(Z[:, :2].astype(int)):
        block = d[np.ix_(members[a], members[b])]
        across[k] = block.max() if mode == "upper" else block.min()
    heights = np.zeros(2 * n - 1)
    if mode == "upper":
        for k, (a, b) in enumerate(Z[:, :2].astype(int)):
            heights[n + k] = max(across[k], heights[a], heights[b])
    elif mode == "lower":
        heights[n:] = across
        for k in range(n - 2, -1, -1):  # parents come after children: sweep top-down
            for child in Z[k, :2].astype(int):
                if child >= n:
                    heights[child] = min(heights[child], heights[n + k])
    else:
        raise ValueError("mode must be 'upper' or 'lower'")
    Z[:, 2] = heights[n:]
    return Z


def _smooth(cdf: np.ndarray, sd: float, dx: float) -> np.ndarray:
    """CDF of X + N(0, sd^2) given the CDF of X on a uniform grid."""
    if sd < 0.25 * dx:
        return cdf
    return gaussian_filter1d(cdf, sigma=sd / dx, mode="nearest", truncate=8.0)


def tree_expected_max(Z: np.ndarray, floor: float = 0.0) -> float:
    """E max over the leaves of the Gaussian tree process of a monotone dendrogram.

    Two leaves merged at height h have E (Y_i - Y_j)^2 = max(h, floor)^2 - floor^2, so a
    positive floor removes the variance below that height (for decompositions).
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0] + 1
    heights = np.concatenate([np.zeros(n), Z[:, 2]])
    for k, (a, b) in enumerate(Z[:, :2].astype(int)):
        if heights[a] > heights[n + k] + 1e-12 or heights[b] > heights[n + k] + 1e-12:
            raise ValueError("dendrogram heights must be monotone (child <= parent)")
    eff = np.maximum(heights, floor)
    root = 2 * n - 2
    if n == 1 or eff[root] <= floor:
        return 0.0
    span = math.sqrt(eff[root] ** 2 - floor**2)
    x = np.linspace(-8.0 * span, span * (math.sqrt(2.0 * math.log(n)) + 8.0), _GRID_POINTS)
    dx = x[1] - x[0]
    cdf_after_edge: dict[int, np.ndarray] = {}

    def edge_cdf(child: int, parent: int) -> np.ndarray:
        var = (eff[parent] ** 2 - eff[child] ** 2) / 2.0
        sd = math.sqrt(max(var, 0.0))
        if child < n:  # leaf: the subtree maximum is 0
            return stats.norm.cdf(x / sd) if sd > 0 else (x >= 0).astype(float)
        return _smooth(cdf_after_edge.pop(child), sd, dx)

    for k in range(n - 1):
        node = n + k
        a, b = int(Z[k, 0]), int(Z[k, 1])
        cdf = edge_cdf(a, node) * edge_cdf(b, node)
        cdf_after_edge[node] = cdf
    # E M = x_0 + int_{x_0}^inf (1 - F), valid since M >= x_0 up to negligible mass.
    return float(x[0] + np.trapezoid(1.0 - cdf_after_edge[root], x))


def _separated_subset(d: np.ndarray, radius: float) -> np.ndarray:
    """Greedy maximal subset whose pairwise distances are all at least radius."""
    chosen: list[int] = []
    for i in range(d.shape[0]):
        if not chosen or d[i, chosen].min() >= radius:
            chosen.append(i)
    return np.asarray(chosen)


def sandwich_bounds(
    C: np.ndarray,
    topologies: tuple[str, ...] = TOPOLOGIES,
    subset_radii: list[float] | None = None,
) -> dict:
    """Certified bounds lower <= E max Z <= upper for Z ~ N(0, C).

    Every tree shape gives a valid majorant and a valid minorant after refit_heights; the
    upper bound is the best majorant over the shapes, and the lower bound the best minorant
    over the shapes on the full set and on well-separated subsets (the Cantor idea of the
    paper: dropping points can only lower E max, but can raise the minorant).
    """
    C = np.asarray(C, dtype=float)
    if C.shape[0] == 1:
        return {
            "lower": 0.0,
            "upper": 0.0,
            "lower_method": "trivial",
            "upper_method": "trivial",
            "subset_size": 1,
        }
    d = corr_distance(C)
    upper, upper_method = math.inf, ""
    lower, lower_method, subset_size = -math.inf, "", C.shape[0]
    for method in topologies:
        Z = linkage_tree(C, method)
        value = tree_expected_max(refit_heights(Z, d, "upper"))
        if value < upper:
            upper, upper_method = value, method
        value = tree_expected_max(refit_heights(Z, d, "lower"))
        if value > lower:
            lower, lower_method = value, method
    if subset_radii is None:
        off = d[np.triu_indices_from(d, k=1)]
        subset_radii = list(np.quantile(off, [0.05, 0.1, 0.2, 0.35, 0.5])) if off.size else []
    for radius in subset_radii:
        idx = _separated_subset(d, radius)
        if idx.size < 2:
            continue
        sub_C, sub_d = C[np.ix_(idx, idx)], d[np.ix_(idx, idx)]
        for method in topologies:
            value = tree_expected_max(refit_heights(linkage_tree(sub_C, method), sub_d, "lower"))
            if value > lower:
                lower, subset_size = value, idx.size
                lower_method = f"{method} on {radius:.3f}-separated subset"
    return {
        "lower": lower,
        "upper": upper,
        "lower_method": lower_method,
        "upper_method": upper_method,
        "subset_size": int(subset_size),
    }


def mc_expected_max(
    C: np.ndarray, size: int, rng: np.random.Generator, batch: int = 2000
) -> tuple[float, float]:
    """Monte Carlo E max of N(0, C); C is projected onto the PSD cone if needed."""
    C = np.asarray(C, dtype=float)
    vals, vecs = np.linalg.eigh((C + C.T) / 2.0)
    root = vecs * np.sqrt(np.clip(vals, 0.0, None))
    maxima = np.empty(size)
    done = 0
    while done < size:
        k = min(batch, size - done)
        maxima[done : done + k] = (rng.standard_normal((k, C.shape[0])) @ root.T).max(axis=1)
        done += k
    return float(maxima.mean()), float(maxima.std(ddof=1) / math.sqrt(size))


def iid_expected_max(N: int) -> float:
    """E max of N independent standard normals."""
    return theory.mu_n(N)


def bl_expected_max(N: float) -> float:
    """Approximation of E max of N i.i.d. normals used in the Deflated Sharpe Ratio."""
    if N < 2:
        raise ValueError("the approximation needs N >= 2")
    q1 = stats.norm.ppf(1.0 - 1.0 / N)
    q2 = stats.norm.ppf(1.0 - 1.0 / (N * math.e))
    return (1.0 - EULER_GAMMA) * q1 + EULER_GAMMA * q2


def effective_trials(emax: float) -> float:
    """Number N of independent trials with the same expected maximum, mu_N = emax."""
    if emax <= 0.0:
        return 1.0
    if emax >= theory.mu_n(math.exp(700.0)):
        raise ValueError("expected maximum too large to invert")
    root = optimize.brentq(lambda s: theory.mu_n(math.exp(s)) - emax, 1e-9, 700.0, xtol=1e-12)
    return math.exp(root)


def deflated_sharpe(sr: float, T: int, skew: float, kurt: float, sr0: float) -> float:
    """Probabilistic Sharpe ratio of a per-period Sharpe sr against the benchmark sr0.

    With sr0 = sd(SR) * E max Z this is the Deflated Sharpe Ratio of Bailey and Lopez de
    Prado (2014); kurt is the raw (non-excess) kurtosis of the returns.
    """
    var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if var <= 0.0:
        raise ValueError("non-positive variance of the Sharpe estimator; check skew and kurt")
    denom = math.sqrt(var)
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))
