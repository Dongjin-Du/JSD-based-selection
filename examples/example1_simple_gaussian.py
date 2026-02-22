"""
Example 1 – Simple Gaussian model
==================================
The simplest possible usage: a model whose output is a Normal distribution
with mean equal to the input and fixed standard deviation.

This shows that the package is completely model-agnostic—you only need to
provide a callable that maps an input to a scipy-compatible frozen distribution.
"""

import numpy as np
import scipy.stats as st
from jsd_consistency import consistency, plot_jsd, plot_rul_families

# ---------------------------------------------------------------------------
# 1. Define the model
# ---------------------------------------------------------------------------
# model(x) must return a frozen distribution.
SIGMA = 1.0

def model(x: float):
    """Output distribution: N(x, 1).  Mean shifts with the input."""
    return st.norm(loc=x, scale=SIGMA)


# ---------------------------------------------------------------------------
# 2. Define the reference input and sample inputs
# ---------------------------------------------------------------------------
x_ref = 0.0                                           # reference / anchor point

rng      = np.random.default_rng(42)
X_samples = rng.normal(loc=0.0, scale=0.5, size=300)  # samples from OC distribution


# ---------------------------------------------------------------------------
# 3. Compute consistency
# ---------------------------------------------------------------------------
result = consistency(model, x_ref, X_samples)

print(result.summary())
print(f"5th / 95th percentiles: {result.percentile([5, 95])}")


# ---------------------------------------------------------------------------
# 4. Visualise
# ---------------------------------------------------------------------------
fig1 = plot_jsd(
    {"Gaussian model": result},
    title="JSD Distribution – Simple Gaussian Example",
    save_path="example1_jsd.png",
)

fig2 = plot_rul_families(
    model, x_ref, X_samples,
    title="Predictive distributions over input samples",
    x_label="Output value",
    save_path="example1_families.png",
)

print("Plots saved: example1_jsd.png, example1_families.png")
