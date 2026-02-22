"""
jsd_consistency.core
====================
Central public API:  the :func:`consistency` function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats as sp_stats

from .divergence import jsd_distributions


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

#: A *model callable* must accept a single input (scalar, array, or any object
#: the user chooses) and return a ``scipy.stats`` frozen distribution—or any
#: object that exposes a ``.pdf(x)`` method and ``.ppf`` / ``.support()`` for
#: support detection.
ModelCallable = Callable[[Any], Any]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyResult:
    """Holds the full output of :func:`consistency`.

    Attributes
    ----------
    jsd_samples : np.ndarray, shape (n_samples,)
        JSD value between ``f(x_ref)`` and ``f(x_k)`` for every input
        sample ``x_k``.
    mean : float
        Mean of the JSD distribution.  Lower → more consistent.
    variance : float
        Variance of the JSD distribution.  Lower → tighter consistency.
    std : float
        Standard deviation (convenience alias: ``sqrt(variance)``).
    input_samples : sequence
        The input samples ``X`` that were passed in (stored for reference).
    reference_input : any
        The reference input ``x_ref`` that was passed in.
    """

    jsd_samples:     np.ndarray
    mean:            float
    variance:        float
    std:             float
    input_samples:   Any
    reference_input: Any
    # extra moments / percentiles computed lazily
    _percentiles: dict = field(default_factory=dict, repr=False)

    def summary(self) -> str:
        """Human-readable summary string."""
        p5, p95 = np.percentile(self.jsd_samples, [5, 95])
        return (
            f"ConsistencyResult\n"
            f"  n_samples : {len(self.jsd_samples)}\n"
            f"  mean JSD  : {self.mean:.6f}\n"
            f"  std  JSD  : {self.std:.6f}\n"
            f"  var  JSD  : {self.variance:.6f}\n"
            f"  5th–95th  : [{p5:.6f}, {p95:.6f}]\n"
        )

    def percentile(self, q: float | Sequence[float]) -> np.ndarray:
        """Return the *q*-th percentile(s) of the JSD distribution."""
        return np.percentile(self.jsd_samples, q)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def consistency(
    model: ModelCallable,
    reference_input: Any,
    input_samples: Any,
    *,
    n_grid: int = 1000,
    eps: float = 1e-300,
) -> ConsistencyResult:
    """Compute the JSD-based predictive consistency of a probabilistic model.

    Parameters
    ----------
    model : callable
        A function ``model(x) -> frozen_distribution`` where the returned
        object must support ``.pdf(grid)`` evaluation.  The input ``x``
        can be anything—a scalar, a numpy array, a dict, etc.—as long as
        ``model`` knows how to handle it.

        Examples::

            # scipy frozen distribution
            model = lambda x: scipy.stats.norm(loc=f(x), scale=sigma)

            # custom class with .pdf()
            model = lambda x: MyGaussianProcess().predict(x)

    reference_input : any
        The reference input ``x_ref``.  ``model(reference_input)`` defines
        the *anchor* distribution against which all others are compared.

    input_samples : array-like or sequence
        A collection of ``n`` inputs sampled from the intended-use
        (operating-condition) distribution.  Each element is passed
        individually to ``model``.  Can be:

        * a 1-D array of scalars: ``[x_0, x_1, …]``
        * a 2-D array of row-vectors: ``shape (n, d)``
        * any sequence of objects that ``model`` accepts.

    n_grid : int, optional
        Number of evaluation points used for numerical JSD integration.
        Default 1000 is sufficient for smooth unimodal distributions.
        Increase for multimodal or heavy-tailed distributions.

    eps : float, optional
        Small constant added to PDF evaluations to avoid log(0).
        Default ``1e-300``.

    Returns
    -------
    ConsistencyResult
        Container with ``jsd_samples``, ``mean``, ``variance``, ``std``,
        and the original ``input_samples`` / ``reference_input``.

    Raises
    ------
    ValueError
        If ``input_samples`` is empty.

    Examples
    --------
    >>> import numpy as np, scipy.stats as st
    >>> from jsd_consistency import consistency
    >>>
    >>> # Simple Gaussian model: output is N(x, 1)
    >>> model = lambda x: st.norm(loc=x, scale=1.0)
    >>> x_ref  = 0.0
    >>> X      = np.random.normal(0, 0.5, size=200)
    >>> result = consistency(model, x_ref, X)
    >>> print(result.summary())
    """
    # ------------------------------------------------------------------
    # Validate / normalise inputs
    # ------------------------------------------------------------------
    samples = _to_list(input_samples)
    if len(samples) == 0:
        raise ValueError("`input_samples` must contain at least one element.")

    # ------------------------------------------------------------------
    # Reference distribution
    # ------------------------------------------------------------------
    ref_dist = model(reference_input)

    # ------------------------------------------------------------------
    # JSD for every sample
    # ------------------------------------------------------------------
    jsd_vals = np.array([
        jsd_distributions(ref_dist, model(x_k), n_grid=n_grid, eps=eps)
        for x_k in samples
    ])

    mean = float(np.mean(jsd_vals))
    var  = float(np.var(jsd_vals, ddof=0))
    std  = float(np.std(jsd_vals, ddof=0))

    return ConsistencyResult(
        jsd_samples     = jsd_vals,
        mean            = mean,
        variance        = var,
        std             = std,
        input_samples   = input_samples,
        reference_input = reference_input,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_list(samples: Any) -> list:
    """Convert *samples* to a plain Python list of individual inputs."""
    if isinstance(samples, np.ndarray):
        if samples.ndim == 1:
            return list(samples)           # [x0, x1, …]
        elif samples.ndim == 2:
            return list(samples)           # [row0, row1, …]  each row is one input
        else:
            raise ValueError(
                "numpy array input_samples must be 1-D (scalars) or "
                "2-D (one input per row)."
            )
    return list(samples)
