"""
jsd_consistency.divergence
==========================
Jensen–Shannon Divergence (JSD) computations.

Two public functions are provided:

* :func:`jsd_distributions` – JSD between two ``scipy``-compatible frozen
  distributions, evaluated on a shared numerical grid.  This is the
  workhorse used by :func:`~jsd_consistency.core.consistency`.

* :func:`jsd_samples_kde` – JSD between two *empirical* samples, estimated
  via kernel density estimation.  Useful when you only have raw draws rather
  than closed-form distributions.

Background
----------
The JSD between two probability distributions P and Q is:

    JSD(P ∥ Q) = ½ KL(P ∥ M) + ½ KL(Q ∥ M),   M = ½(P + Q)

where KL is the Kullback–Leibler divergence.  Unlike KL, JSD is:

* symmetric: JSD(P ∥ Q) = JSD(Q ∥ P)
* always finite, even when the supports of P and Q differ
* bounded in [0, ln 2]  (using natural log) or [0, 1] (using log base 2)

This implementation uses natural logarithms, so 0 ≤ JSD ≤ ln(2) ≈ 0.693.

Reference
---------
Endres & Schindelin (2003).  "A new metric for probability distributions."
IEEE Trans. Inf. Theory, 49(7), 1858–1860.
"""

from __future__ import annotations

import numpy as np
from scipy.special import rel_entr     # element-wise  p * log(p/q), 0 when p=0
from scipy import stats as sp_stats


__all__ = ["jsd_distributions", "jsd_samples_kde"]


def jsd_distributions(
    p_dist,
    q_dist,
    *,
    n_grid: int = 1000,
    eps: float = 1e-300,
) -> float:
    """JSD between two frozen ``scipy.stats``-compatible distributions.

    The distributions are evaluated on a shared grid whose bounds are set
    to cover ±5 standard deviations from each distribution's mean, or the
    distribution's own ``support()`` if the mean/std are not directly
    accessible.

    Parameters
    ----------
    p_dist, q_dist : frozen distribution objects
        Any object that exposes a ``.pdf(x)`` method.  Typically a
        ``scipy.stats`` frozen rv (e.g. ``scipy.stats.norm(0, 1)``), but
        any object with a compatible ``.pdf`` is accepted.
    n_grid : int
        Number of equally-spaced evaluation points on the shared grid.
    eps : float
        Small value added to every PDF evaluation to avoid ``log(0)``.

    Returns
    -------
    float
        JSD value in [0, ln 2] ≈ [0, 0.693] (natural-log convention).

    Examples
    --------
    >>> from scipy import stats
    >>> from jsd_consistency.divergence import jsd_distributions
    >>> p = stats.norm(0, 1)
    >>> q = stats.norm(1, 1)
    >>> round(jsd_distributions(p, q), 4)
    0.1115
    """
    x_min, x_max = _shared_grid_bounds(p_dist, q_dist)
    grid = np.linspace(x_min, x_max, n_grid)
    dx   = grid[1] - grid[0]

    fp = np.asarray(p_dist.pdf(grid), dtype=float) + eps
    fq = np.asarray(q_dist.pdf(grid), dtype=float) + eps

    # Normalise so they integrate to 1 on the grid (trapezoidal correction)
    _trapz = getattr(np, "trapezoid", None) or np.trapz

    fp /= _trapz(fp, grid)
    fq /= _trapz(fq, grid)

    m = 0.5 * (fp + fq)

    # rel_entr(a, b) = a*log(a/b) elementwise, 0 when a == 0
    kl_pm = np.sum(rel_entr(fp, m)) * dx
    kl_qm = np.sum(rel_entr(fq, m)) * dx

    return float(0.5 * (kl_pm + kl_qm))


def jsd_samples_kde(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    *,
    n_grid: int = 1000,
    eps: float = 1e-300,
    bw_method: str | float | None = None,
) -> float:
    """JSD between two empirical distributions estimated via KDE.

    Use this when you only have samples rather than closed-form distributions.

    Parameters
    ----------
    p_samples, q_samples : array-like, shape (n,)
        1-D arrays of samples from each distribution.
    n_grid : int
        Number of grid points for numerical integration.
    eps : float
        Density floor to avoid ``log(0)``.
    bw_method : str, scalar, or None
        Bandwidth selection method passed to ``scipy.stats.gaussian_kde``.

    Returns
    -------
    float
        JSD value in [0, ln 2].

    Examples
    --------
    >>> import numpy as np
    >>> from jsd_consistency.divergence import jsd_samples_kde
    >>> rng = np.random.default_rng(0)
    >>> p = rng.normal(0, 1, 1000)
    >>> q = rng.normal(1, 1, 1000)
    >>> round(jsd_samples_kde(p, q), 2)
    0.11
    """
    p_arr = np.asarray(p_samples, dtype=float).ravel()
    q_arr = np.asarray(q_samples, dtype=float).ravel()

    all_vals = np.concatenate([p_arr, q_arr])
    x_min = all_vals.min() - 3 * all_vals.std()
    x_max = all_vals.max() + 3 * all_vals.std()
    grid  = np.linspace(x_min, x_max, n_grid)
    dx    = grid[1] - grid[0]

    kde_p = sp_stats.gaussian_kde(p_arr, bw_method=bw_method)(grid) + eps
    kde_q = sp_stats.gaussian_kde(q_arr, bw_method=bw_method)(grid) + eps

    _trapz = getattr(np, "trapezoid", None) or np.trapz
    kde_p /= _trapz(kde_p, grid)
    kde_q /= _trapz(kde_q, grid)

    m = 0.5 * (kde_p + kde_q)
    kl_pm = np.sum(rel_entr(kde_p, m)) * dx
    kl_qm = np.sum(rel_entr(kde_q, m)) * dx

    return float(0.5 * (kl_pm + kl_qm))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shared_grid_bounds(p_dist, q_dist, n_sigma: float = 5.0):
    """Determine a shared evaluation domain covering both distributions."""
    bounds = []
    for dist in (p_dist, q_dist):
        lo, hi = _dist_bounds(dist, n_sigma)
        bounds.append((lo, hi))
    x_min = min(b[0] for b in bounds)
    x_max = max(b[1] for b in bounds)
    return x_min, x_max


def _dist_bounds(dist, n_sigma: float = 5.0):
    """Return (lower, upper) bounds for a single distribution."""
    # Try mean/std first (works for most scipy frozen distributions)
    try:
        mu  = float(dist.mean())
        std = float(dist.std())
        return mu - n_sigma * std, mu + n_sigma * std
    except Exception:
        pass

    # Fall back to ppf (quantile function)
    try:
        return float(dist.ppf(1e-6)), float(dist.ppf(1 - 1e-6))
    except Exception:
        pass

    # Last resort: support
    try:
        lo, hi = dist.support()
        return float(lo), float(hi)
    except Exception:
        raise ValueError(
            "Could not determine the support of the distribution. "
            "Make sure the distribution object has .mean(), .std(), "
            ".ppf(), or .support() methods."
        )
