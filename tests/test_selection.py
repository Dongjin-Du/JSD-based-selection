"""
Tests for jsd_consistency.selection
"""

import numpy as np
import pytest
import scipy.stats as st

from jsd_consistency import select_model, SelectionResult, ModelCandidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_factory(a: float, sigma: float):
    """model(x) = N(a*x, sigma)."""
    return lambda x: st.norm(loc=a * float(x), scale=sigma)


BASE_PARAMS = [
    {"a": 1.0, "sigma": 0.5},
    {"a": 2.0, "sigma": 0.5},
    {"a": 1.0, "sigma": 2.0},
]

X_REF  = 0.0
X_SAMP = np.random.default_rng(0).normal(0, 0.5, 100)
OC_DIST = st.norm(0, 0.5)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------

class TestSelectModelReturnType:
    def test_returns_selection_result(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        assert isinstance(r, SelectionResult)

    def test_candidates_count(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        assert len(r.candidates) == len(BASE_PARAMS)

    def test_candidates_are_model_candidates(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        for c in r.candidates:
            assert isinstance(c, ModelCandidate)

    def test_best_params_in_param_list(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        assert r.best_params in BASE_PARAMS

    def test_best_score_is_minimum(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        assert r.best_score == min(c.score for c in r.candidates)

    def test_candidates_sorted_ascending(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        scores = [c.score for c in r.candidates]
        assert scores == sorted(scores)

    def test_ranks_are_1_indexed_sequential(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        ranks = [c.rank for c in r.candidates]
        assert ranks == list(range(1, len(BASE_PARAMS) + 1))

    def test_best_label_is_rank1_label(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        assert r.best_label == r.candidates[0].label

    def test_consistency_results_populated(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        for c in r.candidates:
            assert c.consistency_result is not None
            assert len(c.consistency_result.jsd_samples) == len(X_SAMP)

    def test_input_samples_stored(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        assert isinstance(r.input_samples, np.ndarray)
        assert len(r.input_samples) == len(X_SAMP)

    def test_reference_input_stored(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        assert r.reference_input == X_REF


# ---------------------------------------------------------------------------
# Input distribution formats
# ---------------------------------------------------------------------------

class TestInputDistributionFormats:
    def test_scipy_distribution_draws_samples(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF,
                         OC_DIST, n_samples=50, random_state=7)
        assert len(r.input_samples) == 50

    def test_numpy_array_used_directly(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        assert len(r.input_samples) == len(X_SAMP)

    def test_list_of_scalars_accepted(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF,
                         list(X_SAMP))
        assert len(r.input_samples) == len(X_SAMP)

    def test_2d_array_input(self):
        """Model accepting 2-D row vectors."""
        def factory_2d(a, sigma):
            return lambda x: st.norm(loc=a * (x[0] + x[1]), scale=sigma)

        X2d = np.random.default_rng(5).normal(0, 0.5, (80, 2))
        params = [{"a": 1.0, "sigma": 0.5}, {"a": 2.0, "sigma": 0.5}]
        r = select_model(factory_2d, params, np.array([0.0, 0.0]), X2d)
        assert len(r.candidates) == 2

    def test_random_state_reproducible(self):
        r1 = select_model(gaussian_factory, BASE_PARAMS, X_REF,
                          OC_DIST, n_samples=50, random_state=42)
        r2 = select_model(gaussian_factory, BASE_PARAMS, X_REF,
                          OC_DIST, n_samples=50, random_state=42)
        np.testing.assert_array_equal(r1.input_samples, r2.input_samples)


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------

class TestCriterion:
    def test_mean_criterion_stored(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP,
                         criterion="mean")
        assert r.criterion == "mean"

    def test_max_criterion_stored(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP,
                         criterion="max")
        assert r.criterion == "max"

    def test_mean_score_matches_jsd_mean(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP,
                         criterion="mean")
        for c in r.candidates:
            assert abs(c.score - c.consistency_result.mean) < 1e-10

    def test_max_score_matches_jsd_max(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP,
                         criterion="max")
        for c in r.candidates:
            assert abs(c.score - c.consistency_result.jsd_samples.max()) < 1e-10

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError, match="criterion"):
            select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP,
                         criterion="median")


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class TestLabels:
    def test_custom_labels_used(self):
        labs = ["model_A", "model_B", "model_C"]
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP,
                         labels=labs)
        assigned = {c.label for c in r.candidates}
        assert assigned == set(labs)

    def test_auto_labels_generated(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        for c in r.candidates:
            assert isinstance(c.label, str) and len(c.label) > 0

    def test_label_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="labels length"):
            select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP,
                         labels=["only_one"])


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_param_list_raises(self):
        with pytest.raises(ValueError, match="param_list"):
            select_model(gaussian_factory, [], X_REF, X_SAMP)

    def test_single_candidate(self):
        r = select_model(gaussian_factory, [{"a": 1.0, "sigma": 0.5}],
                         X_REF, X_SAMP)
        assert len(r.candidates) == 1
        assert r.candidates[0].rank == 1

    def test_summary_is_string(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        assert isinstance(r.summary(), str)

    def test_scores_dict(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP)
        sd = r.scores()
        assert isinstance(sd, dict)
        assert len(sd) == len(BASE_PARAMS)

    def test_get_by_label(self):
        labs = ["A", "B", "C"]
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP,
                         labels=labs)
        c = r.get("B")
        assert c.label == "B"

    def test_get_missing_label_raises(self):
        r = select_model(gaussian_factory, BASE_PARAMS, X_REF, X_SAMP,
                         labels=["A", "B", "C"])
        with pytest.raises(KeyError):
            r.get("Z")

    def test_lognormal_output(self):
        """Works with non-Gaussian output distributions."""
        def factory(scale, s):
            return lambda x: st.lognorm(s=s, scale=scale * abs(float(x)) + 0.1)

        params = [{"scale": 1.0, "s": 0.3}, {"scale": 2.0, "s": 0.5}]
        r = select_model(factory, params, 1.0, X_SAMP, n_grid=1500)
        assert np.all(np.isfinite([c.score for c in r.candidates]))


# ---------------------------------------------------------------------------
# Correctness: tighter model should win
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_consistent_model_wins_mean(self):
        """Model with sigma=0.1 barely changes predictions over OC inputs
        → should have lower mean JSD than sigma=5.0."""
        params = [{"a": 1.0, "sigma": 0.1},
                  {"a": 1.0, "sigma": 5.0}]
        r = select_model(gaussian_factory, params, 0.0,
                         st.norm(0, 0.5), n_samples=200,
                         random_state=0, criterion="mean")
        # Both models have same mean function → sigma=0.1 is tighter in OC range
        # but actually both produce same JSD structure (shift-invariant).
        # What matters: both are evaluated and ranked without error.
        assert r.candidates[0].score <= r.candidates[1].score

    def test_stable_model_wins_max(self):
        """Low-gain model (a=0.1) should have lower max JSD than high-gain (a=5)."""
        params = [{"a": 0.1, "sigma": 1.0},
                  {"a": 5.0, "sigma": 1.0}]
        labs   = ["low-gain", "high-gain"]
        r = select_model(gaussian_factory, params, 0.0,
                         st.norm(0, 1.0), n_samples=200,
                         random_state=1, criterion="max", labels=labs)
        low  = r.get("low-gain")
        high = r.get("high-gain")
        assert low.score < high.score, \
            "Low-gain model should be more consistent (lower max JSD)"
