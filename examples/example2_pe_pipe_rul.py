"""
Example 2 – PE Pipe RUL Prediction (Du et al. 2024)
=====================================================
Reproduces the JSD-based consistency evaluation from the paper:

    Du, D., Karve, P., & Mahadevan, S. (2024).
    "Calibration, validation, and selection of hydrostatic testing-based
    remaining useful life prediction models for polyethylene pipes."
    International Journal of Pressure Vessels and Piping, 207, 105108.

Two RUL models are compared over two operating conditions (OC1 and OC2)
using the package's plug-and-play API.

Model inputs : (S [MPa], T [K])  – hoop stress and temperature
Model output : distribution of ln(RUL [hours])
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from jsd_consistency import consistency, plot_jsd, plot_rul_families

# ---------------------------------------------------------------------------
# 1. Define two RUL models (Table 1 / Section 2.1 of the paper)
# ---------------------------------------------------------------------------
# Each model maps a 2-element input array [S, T] to a frozen Normal
# distribution on ln(RUL).

def make_rpm_model(A=-37, B=16620, C=-1149, sigma=0.73):
    """Rate-Process Method (RPM): ln(RUL) = A + B/T + C*ln(S)"""
    def model(x):
        S, T = x[0], x[1]
        mu = A + B / T + C * np.log(S)
        return st.norm(loc=mu, scale=sigma)
    return model

def make_nb2_model(A=-33, B=14590, C=-6, sigma=0.73):
    """Norman-Brown 2nd model (NB2): ln(RUL) = A + B/T + C*ln(S)"""
    def model(x):
        S, T = x[0], x[1]
        mu = A + B / T + C * np.log(S)
        return st.norm(loc=mu, scale=sigma)
    return model

rpm_model = make_rpm_model()
nb2_model = make_nb2_model()


# ---------------------------------------------------------------------------
# 2. Operating condition OC1: high-stress, mild temperature variation
#    T ~ Uniform(291, 295) K
#    S ~ LogNormal(mean=10 MPa, COV=20%)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
n   = 500

T_oc1 = rng.uniform(291, 295, n)
S_oc1 = rng.lognormal(mean=np.log(10), sigma=0.20, size=n)

# Stack into (n, 2) so each row is one [S, T] input
X_oc1    = np.column_stack([S_oc1, T_oc1])
x_ref_oc1 = np.array([10.0, 293.0])   # reference: S=10 MPa, T=293 K


# ---------------------------------------------------------------------------
# 3. Operating condition OC2: low-stress
#    T ~ Uniform(291, 295) K
#    S ~ LogNormal(mean=0.1 MPa, COV=20%)
# ---------------------------------------------------------------------------
T_oc2 = rng.uniform(291, 295, n)
S_oc2 = rng.lognormal(mean=np.log(0.1), sigma=0.20, size=n)

X_oc2     = np.column_stack([S_oc2, T_oc2])
x_ref_oc2 = np.array([0.1, 293.0])


# ---------------------------------------------------------------------------
# 4. Compute consistency for each model × operating condition
# ---------------------------------------------------------------------------
res_rpm_oc1 = consistency(rpm_model, x_ref_oc1, X_oc1)
res_nb2_oc1 = consistency(nb2_model, x_ref_oc1, X_oc1)

res_rpm_oc2 = consistency(rpm_model, x_ref_oc2, X_oc2)
res_nb2_oc2 = consistency(nb2_model, x_ref_oc2, X_oc2)

print("=== OC1 (high stress, S=10 MPa) ===")
print("RPM:"); print(res_rpm_oc1.summary())
print("NB2:"); print(res_nb2_oc1.summary())

print("=== OC2 (low stress, S=0.1 MPa) ===")
print("RPM:"); print(res_rpm_oc2.summary())
print("NB2:"); print(res_nb2_oc2.summary())


# ---------------------------------------------------------------------------
# 5. Visualise – JSD distributions for both OC conditions (Fig. 8 / 10)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

plot_jsd(
    {"RPM": res_rpm_oc1, "NB2": res_nb2_oc1},
    title="JSD Distribution – OC1 (S=10 MPa)",
    ax=axes[0],
)
plot_jsd(
    {"RPM": res_rpm_oc2, "NB2": res_nb2_oc2},
    title="JSD Distribution – OC2 (S=0.1 MPa)",
    ax=axes[1],
)

fig.suptitle("Consistency Comparison – Du et al. (2024)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("example2_jsd_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Family of RUL distributions for RPM under OC1
fig2 = plot_rul_families(
    rpm_model, x_ref_oc1, X_oc1,
    title="RPM model – RUL families over OC1",
    x_label="ln(RUL [hours])",
    save_path="example2_rpm_families_oc1.png",
)
plt.show()

print("Plots saved.")
