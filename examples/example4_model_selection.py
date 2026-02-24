"""
Example 4 – Model Selection with select_model()
================================================
Demonstrates the new model-selection API.

Scenario: you have a Gaussian predictive model whose parameters (mean
scaling factor `a` and noise `sigma`) are uncertain. You want to know
which (a, sigma) combination produces the most consistent predictions
over your expected operating conditions.

Two input-distribution formats are shown:
  A) scipy frozen distribution  → samples drawn automatically
  B) pre-drawn numpy array      → passed directly
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from jsd_consistency import select_model
from jsd_consistency.plot import plot_selection

# ---------------------------------------------------------------------------
# 1. Define the model factory
#    model_factory(**params) must return a callable  x -> frozen distribution
# ---------------------------------------------------------------------------
def make_gaussian_model(a: float, sigma: float):
    """Return a model where output ~ N(a * x, sigma)."""
    def model(x):
        return st.norm(loc=a * float(x), scale=sigma)
    return model


# ---------------------------------------------------------------------------
# 2. Define candidate parameter sets
# ---------------------------------------------------------------------------
param_list = [
    {"a": 1.0, "sigma": 0.3},   # tight, low-gain
    {"a": 1.0, "sigma": 1.0},   # tight, higher noise
    {"a": 2.0, "sigma": 0.3},   # high-gain, tight
    {"a": 2.0, "sigma": 1.0},   # high-gain, noisy
    {"a": 0.5, "sigma": 0.3},   # low-gain, tight
]

labels = [
    "a=1.0, σ=0.3",
    "a=1.0, σ=1.0",
    "a=2.0, σ=0.3",
    "a=2.0, σ=1.0",
    "a=0.5, σ=0.3",
]

reference_input = 0.0   # anchor point


# ---------------------------------------------------------------------------
# 3A. Select using a scipy distribution as the operating-condition input
# ---------------------------------------------------------------------------
print("=" * 60)
print("Format A: scipy frozen distribution as input_distribution")
print("=" * 60)

result_mean = select_model(
    make_gaussian_model,
    param_list,
    reference_input  = reference_input,
    input_distribution = st.norm(loc=0.0, scale=0.5),   # OC distribution
    criterion  = "mean",
    n_samples  = 300,
    random_state = 42,
    labels     = labels,
    verbose    = True,
)

print()
print(result_mean.summary())
print(f"Best params (mean criterion): {result_mean.best_params}")

# ---------------------------------------------------------------------------
# 3B. Same thing with "max" criterion
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Criterion: max JSD  (most robust worst-case)")
print("=" * 60)

result_max = select_model(
    make_gaussian_model,
    param_list,
    reference_input    = reference_input,
    input_distribution = st.norm(loc=0.0, scale=0.5),
    criterion          = "max",
    n_samples          = 300,
    random_state       = 42,
    labels             = labels,
)
print(result_max.summary())


# ---------------------------------------------------------------------------
# 3C. Format B: pre-drawn samples instead of scipy distribution
# ---------------------------------------------------------------------------
print("=" * 60)
print("Format B: pre-drawn numpy array as input_distribution")
print("=" * 60)

X_predrawn = np.random.default_rng(0).normal(0, 0.5, 300)

result_samples = select_model(
    make_gaussian_model,
    param_list,
    reference_input    = reference_input,
    input_distribution = X_predrawn,   # ← array, not scipy dist
    criterion          = "mean",
    labels             = labels,
)
print(result_samples.summary())


# ---------------------------------------------------------------------------
# 4. PE pipe RUL example — selecting between RPM and NB2 model parameters
# ---------------------------------------------------------------------------
print("=" * 60)
print("PE Pipe RUL: selecting best (A, B, C, sigma) for RPM model")
print("=" * 60)

def make_rpm(A, B, C, sigma):
    """Rate-Process Method: ln(RUL) ~ N(A + B/T + C*ln(S), sigma)."""
    def model(x):
        S, T = x[0], x[1]
        mu = A + B / T + C * np.log(S)
        return st.norm(loc=mu, scale=sigma)
    return model

rpm_params = [
    {"A": -37, "B": 16620, "C": -1149, "sigma": 0.5},
    {"A": -37, "B": 16620, "C": -1149, "sigma": 1.0},
    {"A": -40, "B": 17000, "C": -1100, "sigma": 0.5},
    {"A": -35, "B": 16000, "C": -1200, "sigma": 0.5},
]

rpm_labels = [
    "RPM σ=0.5 (paper)",
    "RPM σ=1.0",
    "RPM alt-A σ=0.5",
    "RPM alt-B σ=0.5",
]

rng = np.random.default_rng(1)
S_oc = rng.lognormal(np.log(10), 0.2, 300)
T_oc = rng.uniform(291, 295, 300)
X_oc = np.column_stack([S_oc, T_oc])    # pre-drawn 2-D samples

rpm_result = select_model(
    make_rpm,
    rpm_params,
    reference_input    = np.array([10.0, 293.0]),
    input_distribution = X_oc,           # 2-D array: each row is one [S,T] input
    criterion          = "mean",
    labels             = rpm_labels,
    verbose            = True,
)
print()
print(rpm_result.summary())


# ---------------------------------------------------------------------------
# 5. Visualise
# ---------------------------------------------------------------------------
fig1 = plot_selection(result_mean, title="Model Selection – Mean JSD criterion",
                      save_path="example4_mean.png")
fig2 = plot_selection(result_max,  title="Model Selection – Max JSD criterion",
                      save_path="example4_max.png")
fig3 = plot_selection(rpm_result,  title="PE Pipe RPM – Parameter Selection",
                      save_path="example4_rpm.png")
plt.show()
print("Plots saved.")
