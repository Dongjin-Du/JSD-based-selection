"""
jsd_consistency
===============
A lightweight package for measuring **predictive consistency** of probabilistic
models using the Jensen–Shannon Divergence (JSD).

Given:
  - a probabilistic model  f(x) -> distribution
  - a reference input      x_ref
  - a sample of inputs     X  (drawn from the intended-use distribution)

the package computes, for each x_k in X:

    JSD_k  =  JSD( f(x_ref) || f(x_k) )

and returns the full JSD sample together with its mean and variance.

A model is *more consistent* when small perturbations of the input produce
small changes in the predicted distribution, i.e., when mean(JSD) and
var(JSD) are low.

Reference
---------
Du, D., Karve, P., & Mahadevan, S. (2024).
"Calibration, validation, and selection of hydrostatic testing-based
remaining useful life prediction models for polyethylene pipes."
International Journal of Pressure Vessels and Piping, 207, 105108.
https://doi.org/10.1016/j.ijpvp.2023.105108
"""

from .core import (
    ConsistencyResult,
    consistency,
)
from .divergence import jsd_distributions, jsd_samples_kde
from .plot import plot_jsd, plot_rul_families

__all__ = [
    "consistency",
    "ConsistencyResult",
    "jsd_distributions",
    "jsd_samples_kde",
    "plot_jsd",
    "plot_rul_families",
]

__version__ = "0.1.0"
__author__  = "Du, D., Karve, P., & Mahadevan, S. (implementation)"
