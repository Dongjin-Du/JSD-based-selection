"""
Tests for jsd_consistency
"""

import numpy as np
import pytest
import scipy.stats as st

from jsd_consistency import consistency, ConsistencyResult
from jsd_consistency.divergence import jsd_distributions, jsd_samples_kde


# ---------------------------------------------------------------------------
# Divergence tests
# ---------------------------------------------------------------------------

class TestJsdDistributions:
    def test_identical_distributions_zero(self):
        p = st.norm(0, 1)
        jsd = jsd_distributions(p, p)
        assert jsd < 1e-6, "JSD of identical distributions must be ~0"

    def test_symmetry(self):
        p = st.norm(0, 1)
        q = st.norm(2, 1)
        assert abs(jsd_distributions(p, q) - jsd_distributions(q, p)) < 1e-8

    def test_bounded(self):
        # JSD ∈ [0, ln(2)] under natural log convention
        p = st.norm(0, 1)
        q = st.norm(100, 1)  # very far apart
        jsd = jsd_distributions(p, q)
        assert 0 <= jsd <= np.log(2) + 1e-6

    def test_increases_with_distance(self):
        ref = st.norm(0, 1)
        close  = jsd_distributions(ref, st.norm(0.5, 1))
        far    = jsd_distributions(ref, st.norm(5.0, 1))
        assert close < far

    def test_lognormal_distributions(self):
        p = st.lognorm(s=0.5, scale=np.exp(1))
        q = st.lognorm(s=0.5, scale=np.exp(2))
        jsd = jsd_distributions(p, q, n_grid=2000)
        assert 0 < jsd <= np.log(2) + 1e-6

    def test_different_supports_no_infinity(self):
        """JSD must stay finite even when distributions have different supports."""
        p = st.norm(0, 0.1)
        q = st.norm(10, 0.1)
        jsd = jsd_distributions(p, q)
        assert np.isfinite(jsd)

    def test_returns_float(self):
        jsd = jsd_distributions(st.norm(0, 1), st.norm(1, 1))
        assert isinstance(jsd, float)


class TestJsdSamplesKde:
    def test_identical_samples_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 500)
        jsd = jsd_samples_kde(x, x)
        assert jsd < 0.01

    def test_symmetry(self):
        rng = np.random.default_rng(1)
        p = rng.normal(0, 1, 500)
        q = rng.normal(2, 1, 500)
        assert abs(jsd_samples_kde(p, q) - jsd_samples_kde(q, p)) < 1e-6

    def test_bounded(self):
        rng = np.random.default_rng(2)
        p = rng.normal(0, 1, 500)
        q = rng.normal(50, 1, 500)
        jsd = jsd_samples_kde(p, q)
        assert 0 <= jsd <= np.log(2) + 1e-4


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------

class TestConsistency:
    def _simple_model(self, sigma=1.0):
        return lambda x: st.norm(loc=float(x), scale=sigma)

    def test_returns_consistency_result(self):
        model = self._simple_model()
        X     = np.linspace(-1, 1, 20)
        result = consistency(model, 0.0, X)
        assert isinstance(result, ConsistencyResult)

    def test_jsd_samples_shape(self):
        model  = self._simple_model()
        X      = np.random.default_rng(0).normal(0, 0.5, 50)
        result = consistency(model, 0.0, X)
        assert result.jsd_samples.shape == (50,)

    def test_mean_variance_std_consistent(self):
        model  = self._simple_model()
        X      = np.random.default_rng(3).normal(0, 0.5, 100)
        result = consistency(model, 0.0, X)
        assert abs(result.mean     - result.jsd_samples.mean())   < 1e-10
        assert abs(result.variance - result.jsd_samples.var())    < 1e-10
        assert abs(result.std      - result.jsd_samples.std())    < 1e-10

    def test_non_negative_jsd(self):
        model  = self._simple_model()
        X      = np.random.default_rng(4).normal(0, 1, 100)
        result = consistency(model, 0.0, X)
        assert np.all(result.jsd_samples >= 0)

    def test_reference_equals_sample_gives_near_zero(self):
        """When every sample equals x_ref, JSD should be ~0."""
        model  = self._simple_model()
        X      = np.full(30, 0.0)  # all equal to x_ref
        result = consistency(model, 0.0, X)
        assert result.mean < 1e-5

    def test_higher_sigma_oc_gives_higher_mean_jsd(self):
        """Wider OC distribution → larger input spread → larger JSD."""
        model   = self._simple_model(sigma=1.0)
        x_ref   = 0.0
        rng     = np.random.default_rng(5)
        X_tight = rng.normal(0, 0.1, 200)
        X_wide  = rng.normal(0, 2.0, 200)
        res_tight = consistency(model, x_ref, X_tight)
        res_wide  = consistency(model, x_ref, X_wide)
        assert res_wide.mean > res_tight.mean

    def test_multidimensional_input(self):
        """Model can accept 1-D numpy arrays as inputs (row vectors)."""
        def model_2d(x):
            mu = x[0] + 0.5 * x[1]
            return st.norm(loc=mu, scale=1.0)

        rng    = np.random.default_rng(6)
        X      = rng.normal(0, 0.5, size=(100, 2))   # (100, 2)
        x_ref  = np.array([0.0, 0.0])
        result = consistency(model_2d, x_ref, X)
        assert result.jsd_samples.shape == (100,)

    def test_empty_samples_raises(self):
        model = self._simple_model()
        with pytest.raises(ValueError, match="at least one element"):
            consistency(model, 0.0, [])

    def test_lognormal_model(self):
        """Package works with non-Gaussian output distributions."""
        def model(x):
            return st.lognorm(s=0.4, scale=np.exp(float(x)))

        rng    = np.random.default_rng(7)
        X      = rng.normal(1.0, 0.3, 80)
        result = consistency(model, 1.0, X, n_grid=1500)
        assert np.all(np.isfinite(result.jsd_samples))
        assert result.mean >= 0

    def test_summary_is_string(self):
        model  = self._simple_model()
        result = consistency(model, 0.0, [0.5, -0.5, 1.0])
        assert isinstance(result.summary(), str)

    def test_percentile(self):
        model  = self._simple_model()
        X      = np.random.default_rng(8).normal(0, 0.5, 100)
        result = consistency(model, 0.0, X)
        p5, p95 = result.percentile([5, 95])
        assert p5 < result.mean < p95


# ---------------------------------------------------------------------------
# Smoke test: full pipeline on PE-pipe example
# ---------------------------------------------------------------------------

class TestPePipePipeline:
    def test_rpm_model_consistency(self):
        def rpm(x):
            S, T = x[0], x[1]
            mu = -37 + 16620 / T - 1149 * np.log(S)
            return st.norm(loc=mu, scale=0.73)

        rng   = np.random.default_rng(99)
        S     = rng.lognormal(np.log(10), 0.2, 200)
        T     = rng.uniform(291, 295, 200)
        X     = np.column_stack([S, T])
        x_ref = np.array([10.0, 293.0])

        result = consistency(rpm, x_ref, X)
        assert result.jsd_samples.shape == (200,)
        assert np.all(np.isfinite(result.jsd_samples))
        assert result.mean > 0
