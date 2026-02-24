# jsd-consistency

A lightweight Python package for measuring **predictive consistency** of probabilistic models using the **Jensen–Shannon Divergence (JSD)** — and for **selecting the most consistent model** from a family of candidates.

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
git clone https://github.com/Dongjin-Du/JSD-model-selection.git
cd jsd-consistency
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, NumPy, SciPy, Matplotlib.

---

## Two workflows

### 1. Single-model consistency

Measure how consistent *one* model is across a range of operating conditions.

```python
import numpy as np
import scipy.stats as st
from jsd_consistency import consistency

# Model: input → frozen scipy distribution
def model(x):
    return st.norm(loc=x, scale=1.0)

result = consistency(model, reference_input=0.0, input_samples=np.random.normal(0, 0.5, 300))
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

### 2. Multi-model selection

Given a model family (one callable, many parameter dicts), find the **most consistent** parameter setting — similar to `sklearn.GridSearchCV` but scored by JSD rather than accuracy.

```python
import scipy.stats as st
from jsd_consistency import select_model

# Factory: returns a model callable for a given parameter set
def make_model(a, sigma):
    return lambda x: st.norm(loc=a * float(x), scale=sigma)

# Candidate parameter sets
param_list = [
    {"a": 1.0, "sigma": 0.5},
    {"a": 1.0, "sigma": 2.0},
    {"a": 2.0, "sigma": 0.5},
    {"a": 2.0, "sigma": 2.0},
]

result = select_model(
    make_model,
    param_list,
    reference_input    = 0.0,
    input_distribution = st.norm(0, 0.5),  # scipy dist → samples drawn automatically
    criterion          = "mean",            # or "max"
    n_samples          = 300,
    random_state       = 42,
    labels             = ["a=1 σ=0.5", "a=1 σ=2", "a=2 σ=0.5", "a=2 σ=2"],
    verbose            = True,
)

print(result.summary())
print("Best params:", result.best_params)
```

```
SelectionResult  (criterion: mean JSD)
  Reference input : 0.0
  n_samples       : 300
  n_candidates    : 4

  Rank   Label        Score       Mean JSD    Max JSD     Std JSD
  ------ ------------ ----------  ----------  ----------  ----------
  1      a=1 σ=0.5    0.030241    0.030241    0.128374    0.021837  ◀ best
  2      a=2 σ=0.5    0.108763    0.108763    0.390211    0.074512
  3      a=1 σ=2      0.186432    0.186432    0.512847    0.091034
  4      a=2 σ=2      0.241109    0.241109    0.631024    0.108762
```

---

## API reference

### `select_model(model_factory, param_list, reference_input, input_distribution, *, criterion, n_samples, random_state, labels, n_grid, eps, verbose)`

Select the most consistent model from a family of parameter settings.

| Argument | Type | Description |
|---|---|---|
| `model_factory` | `callable` | `model_factory(**params) → model_callable`. Called once per candidate. |
| `param_list` | `list[dict]` | One dict per candidate. |
| `reference_input` | any | Anchor input $x_r$. |
| `input_distribution` | array-like **or** scipy frozen dist | Operating-condition inputs. Pass a **scipy distribution** to draw samples automatically, or a **numpy array / list** to use pre-drawn samples directly. |
| `criterion` | `"mean"` or `"max"` | `"mean"` selects lowest average JSD; `"max"` selects lowest worst-case JSD. |
| `n_samples` | int | Samples to draw when `input_distribution` is a scipy dist (default 300). Ignored otherwise. |
| `random_state` | int or None | Seed for reproducible sampling. |
| `labels` | list of str | Human-readable names for candidates (auto-generated if omitted). |
| `verbose` | bool | Print progress per candidate. |

**Returns** a `SelectionResult` with:

| Attribute / Method | Description |
|---|---|
| `.best_params` | Winning parameter dict |
| `.best_label` | Label of the winner |
| `.best_score` | Score (mean or max JSD) of the winner |
| `.criterion` | `"mean"` or `"max"` |
| `.candidates` | All `ModelCandidate` objects, sorted best → worst |
| `.input_samples` | Numpy array of inputs actually used |
| `.summary()` | Leaderboard table as a string |
| `.scores()` | `{label: score}` dict for all candidates |
| `.get(label)` | Retrieve a specific `ModelCandidate` by label |

Each `ModelCandidate` exposes:

| Attribute | Description |
|---|---|
| `.params` | Parameter dict |
| `.label` | Name |
| `.score` | Selection score (mean or max JSD) |
| `.rank` | Integer rank (1 = best) |
| `.consistency_result` | Full `ConsistencyResult` with `.jsd_samples`, `.mean`, `.std`, etc. |

---

### `consistency(model, reference_input, input_samples, *, n_grid=1000, eps=1e-300)`

Compute the JSD distribution for a single, pre-configured model.

| Argument | Type | Description |
|---|---|---|
| `model` | `callable` | `model(x) → frozen distribution` with `.pdf()`. |
| `reference_input` | any | Anchor input $x_r$. |
| `input_samples` | array-like or list | 1-D scalars, 2-D row vectors, or any sequence `model` accepts. |

**Returns** a `ConsistencyResult` with `.jsd_samples`, `.mean`, `.variance`, `.std`, `.summary()`, `.percentile(q)`.

---

### `jsd_distributions(p_dist, q_dist, *, n_grid=1000, eps=1e-300)`

JSD between two frozen scipy distributions, evaluated on a shared numerical grid.

```python
from jsd_consistency.divergence import jsd_distributions
import scipy.stats as st

print(jsd_distributions(st.norm(0, 1), st.norm(1, 1)))  # ≈ 0.1115
```

---

### `jsd_samples_kde(p_samples, q_samples, *, n_grid=1000, eps=1e-300, bw_method=None)`

JSD between two empirical distributions estimated via KDE.

```python
from jsd_consistency.divergence import jsd_samples_kde
import numpy as np

p = np.random.normal(0, 1, 1000)
q = np.random.normal(1, 1, 1000)
print(jsd_samples_kde(p, q))   # ≈ 0.11
```

---

### `plot_selection(selection_result, *, title, max_candidates, save_path)`

Visualise a `SelectionResult` — two panels: KDE of JSD distributions + score bar chart.

```python
from jsd_consistency.plot import plot_selection
plot_selection(result, title="Model Selection", save_path="selection.png")
```

---

### `plot_jsd(results, *, title, highlight_best, ax, save_path)`

Plot JSD distributions for one or more `ConsistencyResult` objects.

```python
from jsd_consistency import plot_jsd
fig = plot_jsd({"Model A": result_a, "Model B": result_b}, title="JSD Comparison")
```

---

### `plot_rul_families(model, reference_input, input_samples, *, n_show, ax, save_path)`

Plot the family of predicted distributions overlaid over a set of input samples.

---

## Input distribution formats

`select_model` accepts three formats for `input_distribution`:

```python
# Format A: scipy frozen distribution → n_samples drawn automatically
select_model(..., input_distribution=scipy.stats.norm(0, 0.5), n_samples=300)

# Format B: pre-drawn 1-D array → used directly, n_samples ignored
X = np.random.normal(0, 0.5, 300)
select_model(..., input_distribution=X)

# Format C: pre-drawn 2-D array for multi-dimensional inputs (one row per sample)
X2d = np.random.normal(0, 0.5, (300, 2))
select_model(..., input_distribution=X2d)
```

---

## Design philosophy

`jsd-consistency` is deliberately **model-agnostic and input-agnostic**:

- The model factory is a plain Python callable — no base class, no registration.
- Inputs can be scalars, NumPy row vectors, dicts, or any custom objects.
- Output distributions can be anything with a `.pdf()` method — not just Gaussians.

---

## Examples

| File | Description |
|---|---|
| `examples/example1_simple_gaussian.py` | Single-model consistency, minimal Gaussian |
| `examples/example2_pe_pipe_rul.py` | PE pipe RUL consistency (paper example) |
| `examples/example3_custom_model.py` | LogNormal output, 2-D inputs |
| `examples/example4_model_selection.py` | `select_model` with multiple criteria and input formats |

Run any example from the repo root:

```bash
python examples/example4_model_selection.py
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
