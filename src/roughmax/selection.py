"""Selection bias among correlated backtests: certified bounds on E max of correlated z-scores.

When N correlated strategy variants are backtested, the best Sharpe ratio is inflated by
selection. Under the null, the standardised Sharpe estimates are approximately
Z ~ N(0, C), and the Deflated Sharpe Ratio needs E max_i Z_i. Assuming independent trials
overstates it; Monte Carlo needs a factorisation of C and gives no structural insight.

This module applies the comparison method of the paper (Sudakov-Fernique against tree
processes) to an empirical correlation matrix. With the canonical distance
d(i, j) = (E (Z_i - Z_j)^2)^{1/2} = sqrt(2 (1 - C_ij)):

* complete-linkage clustering yields an ultrametric u >= d (merge height = largest
  distance between the merged clusters), so the tree process with increments u dominates
  Z and E max Z <= E max(tree);
* single-linkage clustering yields the subdominant ultrametric u <= d, and so does single
  linkage on any subset, so E max(subset tree) <= E max Z.

E max of a tree process is computed exactly (up to quadrature) by a CDF recursion over
the dendrogram, without Monte Carlo.
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


def linkage_tree(C: np.ndarray, method: str) -> np.ndarray:
    """Hierarchical clustering of the canonical distance ('complete' or 'single')."""
    if method not in ("complete", "single"):
        raise ValueError("only complete and single linkage give one-sided ultrametric bounds")
    return linkage(squareform(corr_distance(C), checks=False), method=method)


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


def sandwich_bounds(C: np.ndarray, subset_radii: list[float] | None = None) -> dict:
    """Certified bounds lower <= E max Z <= upper for Z ~ N(0, C)."""
    C = np.asarray(C, dtype=float)
    d = corr_distance(C)
    upper = tree_expected_max(linkage_tree(C, "complete"))
    lower = tree_expected_max(linkage_tree(C, "single"))
    method, subset_size = "single linkage", C.shape[0]
    if subset_radii is None:
        off = d[np.triu_indices_from(d, k=1)]
        subset_radii = list(np.quantile(off, [0.05, 0.1, 0.2, 0.35, 0.5])) if off.size else []
    for radius in subset_radii:
        idx = _separated_subset(d, radius)
        if idx.size < 2:
            continue
        value = tree_expected_max(linkage_tree(C[np.ix_(idx, idx)], "single"))
        if value > lower:
            lower, method, subset_size = value, f"single linkage on {radius:.3f}-separated subset", idx.size
    return {"lower": lower, "upper": upper, "lower_method": method, "subset_size": int(subset_size)}


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
    q1 = stats.norm.ppf(1.0 - 1.0 / N)
    q2 = stats.norm.ppf(1.0 - 1.0 / (N * math.e))
    return (1.0 - EULER_GAMMA) * q1 + EULER_GAMMA * q2


def effective_trials(emax: float) -> float:
    """Number N of independent trials with the same expected maximum, mu_N = emax."""
    if emax <= 0.0:
        return 1.0
    root = optimize.brentq(lambda s: theory.mu_n(math.exp(s)) - emax, 1e-9, 60.0, xtol=1e-12)
    return math.exp(root)


def deflated_sharpe(sr: float, T: int, skew: float, kurt: float, sr0: float) -> float:
    """Probabilistic Sharpe ratio of a per-period Sharpe sr against the benchmark sr0.

    With sr0 = sd(SR) * E max Z this is the Deflated Sharpe Ratio of Bailey and Lopez de
    Prado (2014); kurt is the raw (non-excess) kurtosis of the returns.
    """
    denom = math.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))
