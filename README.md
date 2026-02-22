# jsd-consistency

A lightweight Python package for measuring **predictive consistency** of probabilistic models using the **Jensen–Shannon Divergence (JSD)**.

Given a model, a reference input, and a sample of inputs drawn from the intended operating distribution, `jsd-consistency` quantifies how much the model's predictions change — a model is *more consistent* when small input perturbations cause small changes in predicted distributions.

---

## Background

This package implements the consistency-based model selection metric proposed in:

> Du, D., Karve, P., & Mahadevan, S. (2024). *Calibration, validation, and selection of hydrostatic testing-based remaining useful life prediction models for polyethylene pipes.* International Journal of Pressure Vessels and Piping, 207, 105108. https://doi.org/10.1016/j.ijpvp.2023.105108

**Core idea:** for a probabilistic model $f(x) \to p(y \mid x)$ and a reference input $x_r$, sample $N$ inputs $\{x_k\}$ from the intended-use distribution and compute:

$$\text{JSD}_k = \text{JSD}\bigl(f(x_r) \;\|\; f(x_k)\bigr), \quad k = 1, \dots, N$$

The resulting JSD distribution — characterised by its **mean** and **variance** — measures the model's predictive consistency over the operating conditions. Lower mean/variance = more consistent.

**Why JSD?** Unlike KL divergence, JSD is symmetric, always finite (even when distribution supports differ), and bounded in $[0, \ln 2]$.

---

## Installation

```bash
pip install jsd-consistency
```

Or from source:

```bash
git clone https://github.com/Dongjin-Du/JSD-based-selection.git
cd jsd-consistency
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy, SciPy, Matplotlib.

---

## Quick start

```python
import numpy as np
import scipy.stats as st
from jsd_consistency import consistency

# 1. Define your model: input → frozen scipy distribution
def model(x):
    return st.norm(loc=x, scale=1.0)

# 2. Reference input and operating-condition samples
x_ref    = 0.0
X_samples = np.random.normal(0, 0.5, size=300)

# 3. Compute consistency
result = consistency(model, x_ref, X_samples)
print(result.summary())
```

```
ConsistencyResult
  n_samples : 300
  mean JSD  : 0.030241
  std  JSD  : 0.021837
  var  JSD  : 0.000477
  5th–95th  : [0.000378, 0.072903]
```

---

## API reference

### `consistency(model, reference_input, input_samples, *, n_grid=1000, eps=1e-300)`

Main entry point.

| Argument | Type | Description |
|---|---|---|
| `model` | `callable` | `model(x) → frozen distribution`.  The distribution must expose `.pdf(grid)`. |
| `reference_input` | any | The anchor input $x_r$. |
| `input_samples` | array-like or list | Inputs $\{x_k\}$ sampled from the OC distribution.  Can be 1-D scalars, 2-D row vectors, or any sequence of objects `model` accepts. |
| `n_grid` | int | Grid resolution for numerical JSD integration (default 1000). |
| `eps` | float | PDF floor to avoid `log(0)` (default 1e-300). |

**Returns** a `ConsistencyResult` with:

| Attribute | Description |
|---|---|
| `.jsd_samples` | `np.ndarray` of JSD values, one per input sample |
| `.mean` | Mean of the JSD distribution |
| `.variance` | Variance of the JSD distribution |
| `.std` | Standard deviation |
| `.summary()` | Human-readable string |
| `.percentile(q)` | Percentile(s) of the JSD distribution |

---

### `jsd_distributions(p_dist, q_dist, *, n_grid=1000, eps=1e-300)`

JSD between two frozen `scipy.stats`-compatible distributions, evaluated on a shared numerical grid.

```python
from jsd_consistency.divergence import jsd_distributions
import scipy.stats as st

p = st.norm(0, 1)
q = st.norm(1, 1)
print(jsd_distributions(p, q))   # ≈ 0.1115
```

---

### `jsd_samples_kde(p_samples, q_samples, *, n_grid=1000, eps=1e-300, bw_method=None)`

JSD between two empirical distributions estimated via KDE. Use when you only have raw samples.

```python
from jsd_consistency.divergence import jsd_samples_kde
import numpy as np

p = np.random.normal(0, 1, 1000)
q = np.random.normal(1, 1, 1000)
print(jsd_samples_kde(p, q))    # ≈ 0.11
```

---

### `plot_jsd(results, *, title, highlight_best, ax, save_path)`

Plot the JSD distribution(s) for one or more models.

```python
from jsd_consistency import plot_jsd

fig = plot_jsd(
    {"Model A": result_a, "Model B": result_b},
    title="JSD Comparison",
    save_path="jsd.png",
)
```

---

### `plot_rul_families(model, reference_input, input_samples, *, n_show, ax, save_path)`

Plot the family of predicted distributions over a set of input samples.

---

## Design philosophy

`jsd-consistency` is deliberately **model-agnostic and input-agnostic**:

- The model is just a Python callable — no base class, no registration.
- Inputs can be scalars, NumPy row vectors, dicts, or any custom objects.
- Outputs can be any distribution with a `.pdf()` method — not just Gaussians.

---

## Examples

| File | Description |
|---|---|
| `examples/example1_simple_gaussian.py` | Minimal Gaussian model |
| `examples/example2_pe_pipe_rul.py` | Full PE pipe RUL example from the paper |
| `examples/example3_custom_model.py` | LogNormal output, 2-D inputs |

Run any example from the repo root:

```bash
python examples/example1_simple_gaussian.py
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you use this package in academic work, please cite the original paper:

```bibtex
@article{du2024calibration,
  title   = {Calibration, validation, and selection of hydrostatic testing-based
             remaining useful life prediction models for polyethylene pipes},
  author  = {Du, Dongjin and Karve, Pranav and Mahadevan, Sankaran},
  journal = {International Journal of Pressure Vessels and Piping},
  volume  = {207},
  pages   = {105108},
  year    = {2024},
  doi     = {10.1016/j.ijpvp.2023.105108}
}
```
