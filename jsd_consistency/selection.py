"""
jsd_consistency.selection
=========================
Model selection via JSD-based consistency comparison.

The central function :func:`select_model` answers the question:

    *Given a model family (one callable parameterised by different dicts)
    and an operating-condition input distribution, which parameter setting
    produces the most consistent predictions?*

Workflow
--------
1. Draw (or reuse) ``n_samples`` inputs from the operating-condition
   distribution.
2. For each parameter dict, instantiate ``model_factory(**params)`` to
   get a model callable, then compute the full JSD distribution via
   :func:`~jsd_consistency.core.consistency`.
3. Score every candidate by the chosen criterion (``"mean"`` or
   ``"max"`` of its JSD distribution).
4. Rank candidates ascending (lower score = more consistent = better).
5. Return a :class:`SelectionResult` containing the full ranking,
   every candidate's :class:`~jsd_consistency.core.ConsistencyResult`,
   and the winning parameter dict.

Input distribution
------------------
The operating-condition distribution can be supplied as:

* **Samples** — any array-like / sequence already drawn by the caller.
* **A scipy frozen distribution** — ``n_samples`` inputs are drawn
  automatically using ``.rvs(n_samples, random_state=...)``.

Model factory
-------------
``model_factory`` must have the signature::

    model_factory(**params) -> callable

where the returned callable maps a single input ``x`` to a frozen
distribution (anything with a ``.pdf`` method).

Example::

    def make_model(loc_scale, sigma):
        return lambda x: scipy.stats.norm(loc=loc_scale * x, scale=sigma)

    param_list = [
        {"loc_scale": 1.0, "sigma": 0.5},
        {"loc_scale": 1.0, "sigma": 1.0},
        {"loc_scale": 2.0, "sigma": 0.5},
    ]

    result = select_model(make_model, param_list, x_ref=0.0,
                          input_distribution=scipy.stats.norm(0, 1),
                          n_samples=300, criterion="mean")
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from scipy import stats as sp_stats

from .core import ConsistencyResult, consistency, _to_list


__all__ = ["ModelCandidate", "SelectionResult", "select_model"]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ModelCandidate:
    """One entry in the model selection race.

    Attributes
    ----------
    params : dict
        The parameter dict used to instantiate this candidate via
        ``model_factory(**params)``.
    label : str
        Human-readable name.  Auto-generated from *params* if not given.
    consistency_result : ConsistencyResult
        Full JSD distribution for this candidate (set after fitting).
    score : float
        Scalar selection score (mean or max JSD, depending on criterion).
    rank : int
        Rank among all candidates (1 = best = lowest score).
    """
    params:             dict
    label:              str
    consistency_result: ConsistencyResult | None = field(default=None, repr=False)
    score:              float = field(default=float("inf"))
    rank:               int   = field(default=0)

    @staticmethod
    def _auto_label(params: dict) -> str:
        parts = [f"{k}={v}" for k, v in params.items()]
        return "{" + ", ".join(parts) + "}"


@dataclass
class SelectionResult:
    """Full output of :func:`select_model`.

    Attributes
    ----------
    best_params : dict
        Parameter dict of the winning candidate.
    best_label : str
        Label of the winning candidate.
    best_score : float
        JSD score (mean or max) of the winning candidate.
    criterion : str
        ``"mean"`` or ``"max"`` — the selection criterion used.
    candidates : list of ModelCandidate
        All candidates, sorted ascending by score (best first).
    input_samples : np.ndarray
        The actual input samples used (useful when drawn automatically).
    reference_input : any
        The reference input that was passed in.
    """
    best_params:     dict
    best_label:      str
    best_score:      float
    criterion:       str
    candidates:      list[ModelCandidate]
    input_samples:   np.ndarray
    reference_input: Any

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable leaderboard."""
        width = max(len(c.label) for c in self.candidates) + 2
        header = (
            f"SelectionResult  (criterion: {self.criterion} JSD)\n"
            f"  Reference input : {self.reference_input}\n"
            f"  n_samples       : {len(self.input_samples)}\n"
            f"  n_candidates    : {len(self.candidates)}\n\n"
            f"  {'Rank':<6} {'Label':<{width}} "
            f"{'Score':>10}  {'Mean JSD':>10}  {'Max JSD':>10}  {'Std JSD':>10}\n"
            f"  {'-'*6} {'-'*width} {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}"
        )
        rows = []
        for c in self.candidates:
            cr   = c.consistency_result
            flag = " ◀ best" if c.rank == 1 else ""
            rows.append(
                f"  {c.rank:<6} {c.label:<{width}} "
                f"{c.score:>10.6f}  "
                f"{cr.mean:>10.6f}  "
                f"{cr.jsd_samples.max():>10.6f}  "
                f"{cr.std:>10.6f}"
                f"{flag}"
            )
        return header + "\n" + "\n".join(rows) + "\n"

    def get(self, label: str) -> ModelCandidate:
        """Retrieve a candidate by label."""
        for c in self.candidates:
            if c.label == label:
                return c
        raise KeyError(f"No candidate with label {label!r}")

    def scores(self) -> dict[str, float]:
        """Return ``{label: score}`` for all candidates."""
        return {c.label: c.score for c in self.candidates}


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def select_model(
    model_factory: Callable[..., Callable],
    param_list: list[dict],
    reference_input: Any,
    input_distribution: Any,
    *,
    criterion: Literal["mean", "max"] = "mean",
    n_samples: int = 300,
    random_state: int | None = None,
    labels: list[str] | None = None,
    n_grid: int = 1000,
    eps: float = 1e-300,
    verbose: bool = False,
) -> SelectionResult:
    """Select the most consistent model variant by JSD comparison.

    Parameters
    ----------
    model_factory : callable
        A factory function with signature ``model_factory(**params) ->
        model_callable``, where ``model_callable(x)`` returns a frozen
        distribution.

    param_list : list of dict
        Each dict is one candidate.  ``model_factory(**params)`` is called
        once per candidate.

    reference_input : any
        The anchor input ``x_ref``.  JSD is measured relative to the
        prediction at this input.

    input_distribution : array-like **or** scipy frozen distribution
        The operating-condition input distribution.  Two formats accepted:

        * **Samples** — a 1-D or 2-D array / sequence already drawn by
          the caller.  Used as-is; ``n_samples`` is ignored.
        * **scipy frozen distribution** — must expose ``.rvs()``.
          ``n_samples`` draws are taken automatically.

    criterion : {"mean", "max"}, default "mean"
        Scoring rule applied to each candidate's JSD distribution:

        * ``"mean"`` — lowest mean JSD wins (most consistent on average).
        * ``"max"``  — lowest max JSD wins (most robust in the worst case).

    n_samples : int, default 300
        Number of samples to draw when *input_distribution* is a scipy
        distribution.  Ignored when samples are provided directly.

    random_state : int or None, default None
        Seed for reproducible sampling (only used when drawing from a
        scipy distribution).

    labels : list of str, optional
        Human-readable names for each candidate.  Must match the length
        of ``param_list``.  Auto-generated from param dicts if omitted.

    n_grid : int, default 1000
        Grid resolution for numerical JSD integration.

    eps : float, default 1e-300
        PDF floor to avoid ``log(0)``.

    verbose : bool, default False
        If ``True``, print progress as each candidate is evaluated.

    Returns
    -------
    SelectionResult
        Contains the winning params, full ranking, and every candidate's
        :class:`~jsd_consistency.core.ConsistencyResult`.

    Raises
    ------
    ValueError
        If ``param_list`` is empty, ``criterion`` is unknown, or labels
        length mismatches ``param_list``.

    Examples
    --------
    >>> import numpy as np, scipy.stats as st
    >>> from jsd_consistency import select_model
    >>>
    >>> # Factory: model(x) = N(a*x, sigma)
    >>> def make(a, sigma):
    ...     return lambda x: st.norm(loc=a * float(x), scale=sigma)
    >>>
    >>> params = [{"a": 1.0, "sigma": 0.5},
    ...           {"a": 1.0, "sigma": 2.0},
    ...           {"a": 2.0, "sigma": 0.5}]
    >>>
    >>> result = select_model(
    ...     make, params,
    ...     reference_input=0.0,
    ...     input_distribution=st.norm(0, 0.5),
    ...     criterion="mean", n_samples=200,
    ... )
    >>> print(result.best_params)
    >>> print(result.summary())
    """
    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if len(param_list) == 0:
        raise ValueError("param_list must contain at least one entry.")

    if criterion not in ("mean", "max"):
        raise ValueError(f"criterion must be 'mean' or 'max', got {criterion!r}.")

    if labels is not None and len(labels) != len(param_list):
        raise ValueError(
            f"labels length ({len(labels)}) must match param_list length "
            f"({len(param_list)})."
        )

    # ------------------------------------------------------------------
    # Resolve input samples
    # ------------------------------------------------------------------
    samples_arr = _resolve_samples(input_distribution, n_samples, random_state)

    # ------------------------------------------------------------------
    # Evaluate each candidate
    # ------------------------------------------------------------------
    candidates: list[ModelCandidate] = []

    for i, params in enumerate(param_list):
        label = (labels[i] if labels is not None
                 else ModelCandidate._auto_label(params))

        if verbose:
            print(f"[select_model] Evaluating {i+1}/{len(param_list)}: {label} ...")

        model = model_factory(**params)
        cr    = consistency(model, reference_input, samples_arr,
                            n_grid=n_grid, eps=eps)

        score = float(cr.mean) if criterion == "mean" else float(cr.jsd_samples.max())

        candidates.append(ModelCandidate(
            params=params, label=label,
            consistency_result=cr, score=score,
        ))

    # ------------------------------------------------------------------
    # Rank ascending (lower score = more consistent = better)
    # ------------------------------------------------------------------
    candidates.sort(key=lambda c: c.score)
    for rank, c in enumerate(candidates, start=1):
        c.rank = rank

    best = candidates[0]

    if verbose:
        print(f"[select_model] Done. Best: {best.label}  "
              f"({criterion} JSD = {best.score:.6f})")

    return SelectionResult(
        best_params     = best.params,
        best_label      = best.label,
        best_score      = best.score,
        criterion       = criterion,
        candidates      = candidates,
        input_samples   = samples_arr,
        reference_input = reference_input,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_samples(
    input_distribution: Any,
    n_samples: int,
    random_state: int | None,
) -> np.ndarray:
    """Return a numpy array of input samples.

    If *input_distribution* looks like a scipy frozen distribution
    (has ``.rvs``), draw ``n_samples`` from it.  Otherwise treat it
    as a pre-drawn sample array.
    """
    if hasattr(input_distribution, "rvs"):
        # scipy frozen distribution (or anything with .rvs)
        raw = input_distribution.rvs(n_samples, random_state=random_state)
        return np.asarray(raw, dtype=float)

    # Pre-drawn samples
    arr = np.asarray(input_distribution, dtype=float)
    if arr.ndim == 0:
        raise ValueError(
            "input_distribution must be a sample array (1-D or 2-D) "
            "or a scipy frozen distribution with .rvs()."
        )
    return arr
