"""
jsd_consistency.plot
====================
Visualization utilities for JSD consistency results.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.figure


__all__ = ["plot_jsd", "plot_rul_families", "plot_selection"]


def plot_jsd(
    results: dict,
    *,
    title: str = "JSD Distribution",
    xlabel: str = "JS Divergence",
    highlight_best: bool = True,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the JSD distribution for one or more models.

    Parameters
    ----------
    results : dict
        Mapping of ``{label: ConsistencyResult}``.  Pass a single-key dict
        when comparing one model, or multiple keys to compare models.
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis.
    highlight_best : bool
        If ``True`` and multiple models are supplied, draw the model with
        the lowest mean JSD with a thicker, solid line.
    ax : matplotlib Axes, optional
        Axes to draw into.  A new figure is created if ``None``.
    save_path : str, optional
        If given, the figure is saved to this path before being returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    colors   = plt.cm.tab10(np.linspace(0, 0.85, max(len(results), 1)))
    best_key = min(results, key=lambda k: results[k].mean) if highlight_best else None

    for (label, res), color in zip(results.items(), colors):
        jsd_vals = res.jsd_samples
        if len(jsd_vals) < 3:
            ax.axvline(jsd_vals.mean(), color=color, label=label)
            continue

        kde = sp_stats.gaussian_kde(jsd_vals)
        x   = np.linspace(max(0, jsd_vals.min() * 0.5),
                           jsd_vals.max() * 1.3, 500)
        lw  = 2.5 if label == best_key else 1.5
        ls  = "-"  if label == best_key else "--"
        ax.plot(
            x, kde(x),
            label=f"{label}  (μ={res.mean:.4f}, σ={res.std:.4f})",
            color=color, linewidth=lw, linestyle=ls,
        )
        ax.axvline(res.mean, color=color, linewidth=0.8, linestyle=":")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(left=0)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_rul_families(
    model,
    reference_input,
    input_samples,
    *,
    n_show: int = 40,
    x_label: str = "Model output",
    title: str = "Families of predicted distributions",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the family of predicted distributions over a set of input samples.

    The reference distribution is drawn as a thick dashed line; sample
    distributions are overlaid as thin translucent curves.

    Parameters
    ----------
    model : callable
        Same signature as in :func:`~jsd_consistency.core.consistency`.
    reference_input : any
        The reference input.
    input_samples : sequence
        Input samples (same as passed to :func:`~jsd_consistency.core.consistency`).
    n_show : int
        Maximum number of sample distributions to draw (randomly subsampled).
    x_label : str
        Label for the horizontal axis.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .core import _to_list

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    samples = _to_list(input_samples)
    rng     = np.random.default_rng(0)
    idxs    = rng.choice(len(samples),
                         size=min(n_show, len(samples)),
                         replace=False)

    ref_dist = model(reference_input)
    lo_ref, hi_ref = _dist_ui_bounds(ref_dist)

    # Gather bounds over all sample distributions
    lo_all, hi_all = lo_ref, hi_ref
    for idx in idxs:
        d = model(samples[idx])
        lo, hi = _dist_ui_bounds(d)
        lo_all = min(lo_all, lo)
        hi_all = max(hi_all, hi)

    grid = np.linspace(lo_all, hi_all, 600)

    # Sample distributions
    for i, idx in enumerate(idxs):
        dist = model(samples[idx])
        ax.plot(grid, dist.pdf(grid),
                color="#4C72B0", alpha=0.25, linewidth=0.7,
                label="Samples" if i == 0 else "_nolegend_")

    # Reference distribution
    ax.plot(grid, ref_dist.pdf(grid),
            color="#C44E52", linewidth=2.5, linestyle="--",
            label=f"Reference")

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("PDF", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_selection(
    selection_result,
    *,
    title: str | None = None,
    max_candidates: int = 10,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualise a :class:`~jsd_consistency.selection.SelectionResult`.

    Draws two panels side by side:

    * **Left** — KDE of JSD distributions for each candidate (like
      :func:`plot_jsd` but built from a SelectionResult directly).
    * **Right** — bar chart of scores (mean or max JSD) so the ranking
      is immediately readable.

    Parameters
    ----------
    selection_result : SelectionResult
        Output of :func:`~jsd_consistency.selection.select_model`.
    title : str, optional
        Overall figure title.  Defaults to
        ``"Model Selection  (criterion: <criterion>)"``.
    max_candidates : int, default 10
        Cap on how many candidates to display (top-ranked first).
    ax : ignored
        Kept for API consistency; this function always creates its own figure.
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    sr = selection_result
    shown = sr.candidates[:max_candidates]
    n     = len(shown)
    cmap  = plt.cm.tab10(np.linspace(0, 0.85, max(n, 1)))

    fig, (ax_kde, ax_bar) = plt.subplots(1, 2, figsize=(13, 4))

    # ── Left: KDE of JSD distributions ──────────────────────────────
    for cand, color in zip(shown, cmap):
        cr  = cand.consistency_result
        jsd = cr.jsd_samples
        if len(jsd) < 3:
            ax_kde.axvline(jsd.mean(), color=color, label=cand.label)
            continue
        kde = sp_stats.gaussian_kde(jsd)
        x   = np.linspace(max(0, jsd.min() * 0.5), jsd.max() * 1.3, 500)
        lw  = 2.5 if cand.rank == 1 else 1.2
        ls  = "-"  if cand.rank == 1 else "--"
        ax_kde.plot(
            x, kde(x),
            color=color, linewidth=lw, linestyle=ls,
            label=f"#{cand.rank} {cand.label}",
        )
        ax_kde.axvline(cr.mean, color=color, linewidth=0.7, linestyle=":")

    ax_kde.set_xlabel("JS Divergence", fontsize=11)
    ax_kde.set_ylabel("Density",       fontsize=11)
    ax_kde.set_xlim(left=0)
    ax_kde.set_title("JSD Distributions", fontsize=12)
    ax_kde.legend(fontsize=8, loc="upper right")

    # ── Right: bar chart of scores ───────────────────────────────────
    labels = [f"#{c.rank} {c.label}" for c in shown]
    scores = [c.score for c in shown]
    colors_bar = [cmap[i] for i in range(n)]
    # highlight winner
    colors_bar[0] = "#C44E52"

    bars = ax_bar.barh(
        range(n)[::-1], scores,
        color=colors_bar, edgecolor="white", linewidth=0.5,
    )
    ax_bar.set_yticks(range(n)[::-1])
    ax_bar.set_yticklabels(labels, fontsize=8)
    ax_bar.set_xlabel(f"{sr.criterion.capitalize()} JSD", fontsize=11)
    ax_bar.set_title(f"Score Ranking  ({sr.criterion})", fontsize=12)
    ax_bar.axvline(shown[0].score, color="#C44E52",
                   linewidth=1.0, linestyle="--", alpha=0.6)

    for bar, score in zip(bars, scores[::-1]):
        ax_bar.text(
            score + ax_bar.get_xlim()[1] * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}", va="center", fontsize=7,
        )

    fig.suptitle(
        title or f"Model Selection  (criterion: {sr.criterion} JSD)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dist_ui_bounds(dist, n_sigma: float = 4.0):
    try:
        mu  = float(dist.mean())
        std = float(dist.std())
        return mu - n_sigma * std, mu + n_sigma * std
    except Exception:
        pass
    try:
        return float(dist.ppf(1e-5)), float(dist.ppf(1 - 1e-5))
    except Exception:
        pass
    lo, hi = dist.support()
    return float(lo), float(hi)
