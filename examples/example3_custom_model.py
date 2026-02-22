"""
Example 3 – Custom Non-Gaussian Model
=======================================
Demonstrates that jsd_consistency works with *any* distribution, not just
Gaussians.  Here the model produces a LogNormal predictive distribution
whose parameters depend on the input vector.
"""

import numpy as np
import scipy.stats as st
from jsd_consistency import consistency, plot_jsd, plot_rul_families

# ---------------------------------------------------------------------------
# Model: output is LogNormal(mu=f(x), sigma=0.5)
# Input: 2-D vector [x1, x2]
# ---------------------------------------------------------------------------
def lognormal_model(x):
    x1, x2 = x[0], x[1]
    mu = 0.5 * x1 + 0.3 * x2           # log-space mean depends on both inputs
    return st.lognorm(s=0.5, scale=np.exp(mu))


# ---------------------------------------------------------------------------
# Reference and sample inputs
# ---------------------------------------------------------------------------
x_ref = np.array([1.0, 1.0])

rng = np.random.default_rng(7)
X   = rng.multivariate_normal(
    mean  = [1.0, 1.0],
    cov   = [[0.1, 0.02], [0.02, 0.1]],
    size  = 400,
)  # shape (400, 2)


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------
result = consistency(lognormal_model, x_ref, X, n_grid=1500)

print("LogNormal model consistency:")
print(result.summary())


# ---------------------------------------------------------------------------
# Visualise
# ---------------------------------------------------------------------------
plot_jsd(
    {"LogNormal model": result},
    title="JSD Distribution – LogNormal Model",
    save_path="example3_jsd.png",
)

plot_rul_families(
    lognormal_model, x_ref, X,
    title="LogNormal predictive families",
    x_label="Output value",
    save_path="example3_families.png",
)

print("Plots saved.")
