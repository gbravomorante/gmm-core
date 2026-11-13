import numpy as np
import pytest

from gmm_core.regression import adjust_p_values, dummy_encode, fit_ols, rrpp_test, sum_of_squares


def test_fit_ols_recovers_known_linear_relationship():
    rng = np.random.default_rng(0)
    n = 50
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    true_slope = 3.0
    Y = (true_slope * x + rng.normal(scale=0.01, size=n)).reshape(-1, 1)
    coefficients, fitted = fit_ols(X, Y)
    assert np.isclose(coefficients[1, 0], true_slope, atol=0.05)
    assert np.allclose(fitted.flatten(), Y.flatten(), atol=0.5)


def test_sum_of_squares():
    assert sum_of_squares(np.array([3.0, 4.0])) == 25.0
    assert sum_of_squares(np.zeros((3, 2))) == 0.0


def test_dummy_encode_sorted_and_drop_first():
    matrix, names = dummy_encode(["b", "a", "c", "a"], drop_first=True)
    assert names == ["b", "c"]  # "a" (first sorted level) dropped as reference
    assert matrix.shape == (4, 2)


def test_dummy_encode_no_drop():
    matrix, names = dummy_encode(["x", "y"], drop_first=False)
    assert names == ["x", "y"]
    assert matrix.shape == (2, 2)


def _multivariate_response(n, p, rng, x=None, noise_scale=1.0, true_effect=None):
    """p-column response; if x and true_effect given, each column gets the
    same linear effect of x plus independent noise."""
    Y = rng.normal(scale=noise_scale, size=(n, p))
    if x is not None and true_effect is not None:
        Y = Y + np.outer(x, true_effect)
    return Y


def test_rrpp_detects_strong_true_effect():
    rng = np.random.default_rng(1)
    n, p = 40, 6
    x = rng.normal(size=n)
    true_effect = rng.normal(scale=5.0, size=p)  # strong signal relative to noise
    Y = _multivariate_response(n, p, rng, x=x, noise_scale=0.3, true_effect=true_effect)

    X_reduced = np.ones((n, 1))
    X_full = np.column_stack([np.ones(n), x])

    result = rrpp_test(Y, X_reduced, X_full, "x", n_permutations=499, seed=42)
    assert result.p_value < 0.05
    assert result.r2 > 0.5


def test_rrpp_null_case_is_not_falsely_significant():
    rng = np.random.default_rng(2)
    n, p = 40, 6
    x = rng.normal(size=n)
    Y = _multivariate_response(n, p, rng, x=None)  # pure noise, unrelated to x

    X_reduced = np.ones((n, 1))
    X_full = np.column_stack([np.ones(n), x])

    result = rrpp_test(Y, X_reduced, X_full, "x", n_permutations=499, seed=42)
    assert result.p_value > 0.05
    assert result.r2 < 0.2


def test_rrpp_interaction_term_significant_when_slopes_genuinely_differ():
    rng = np.random.default_rng(3)
    n_per_group, p = 25, 5
    x = np.concatenate([rng.normal(size=n_per_group), rng.normal(size=n_per_group)])
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    group_dummy, _ = dummy_encode(group, drop_first=True)  # (n, 1): 1 for "B"

    # Group A: slope ~0; Group B: strong slope - genuinely different trajectories.
    slope_effect = rng.normal(scale=4.0, size=p)
    Y = rng.normal(scale=0.3, size=(2 * n_per_group, p))
    Y[n_per_group:] += np.outer(x[n_per_group:], slope_effect)

    intercept = np.ones((2 * n_per_group, 1))
    X_main = np.column_stack([intercept, x.reshape(-1, 1), group_dummy])
    interaction = group_dummy * x.reshape(-1, 1)
    X_full = np.column_stack([X_main, interaction])

    result = rrpp_test(Y, X_main, X_full, "x:Group", n_permutations=499, seed=7)
    assert result.p_value < 0.05


def test_rrpp_interaction_term_not_significant_with_common_slope():
    rng = np.random.default_rng(4)
    n_per_group, p = 25, 5
    x = np.concatenate([rng.normal(size=n_per_group), rng.normal(size=n_per_group)])
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    group_dummy, _ = dummy_encode(group, drop_first=True)

    common_slope = rng.normal(scale=3.0, size=p)
    Y = rng.normal(scale=0.3, size=(2 * n_per_group, p)) + np.outer(x, common_slope)

    intercept = np.ones((2 * n_per_group, 1))
    X_main = np.column_stack([intercept, x.reshape(-1, 1), group_dummy])
    interaction = group_dummy * x.reshape(-1, 1)
    X_full = np.column_stack([X_main, interaction])

    result = rrpp_test(Y, X_main, X_full, "x:Group", n_permutations=499, seed=7)
    assert result.p_value > 0.05


def test_rrpp_p_value_bounds_and_reproducibility():
    rng = np.random.default_rng(5)
    n, p = 20, 3
    x = rng.normal(size=n)
    Y = rng.normal(size=(n, p))
    X_reduced = np.ones((n, 1))
    X_full = np.column_stack([np.ones(n), x])

    r1 = rrpp_test(Y, X_reduced, X_full, "x", n_permutations=199, seed=123)
    r2 = rrpp_test(Y, X_reduced, X_full, "x", n_permutations=199, seed=123)
    assert 0.0 < r1.p_value <= 1.0
    assert r1.p_value == r2.p_value  # same seed -> reproducible
    assert r1.ss_effect == r2.ss_effect


# --- calibration of the permutation null --------------------------------------
# These target a real bug: the permuted statistic used to be compared against
# the reduced fit of the ORIGINAL response instead of the permuted one, which
# inflated every permuted draw and made the test wildly conservative (p = 1.000
# on ~98% of null datasets). The pre-existing "detects a strong effect" tests
# all still passed with that bug, so calibration needs its own checks.


def _interaction_design(n=60, p=20, n_groups=3, interaction_strength=0.0, seed=0):
    rng = np.random.default_rng(seed)
    log_size = rng.normal(size=n)
    g = np.array([i % n_groups for i in range(n)])
    dummies = np.zeros((n, n_groups - 1))
    for k in range(1, n_groups):
        dummies[:, k - 1] = (g == k).astype(float)

    Y = rng.normal(scale=1.0, size=(n, p))
    Y += np.outer(log_size, rng.normal(scale=1.5, size=p))
    Y += dummies @ rng.normal(scale=1.5, size=(n_groups - 1, p))
    if interaction_strength:
        Y += (dummies * log_size[:, None]) @ rng.normal(
            scale=interaction_strength, size=(n_groups - 1, p)
        )

    ones = np.ones((n, 1))
    x_main = np.hstack([ones, log_size[:, None], dummies])
    x_interaction = np.hstack([x_main, dummies * log_size[:, None]])
    return Y, x_main, x_interaction


def test_permuted_null_is_centred_on_the_observed_statistic():
    """With no true interaction, the permuted SS distribution should sit around
    the observed SS. The old bug made the permuted mean ~2.4x the observed."""
    Y, x_main, x_interaction = _interaction_design(interaction_strength=0.0, seed=7)

    # Reproduce the permuted distribution the same way rrpp_test does.
    _c, fitted_reduced = fit_ols(x_main, Y)
    residuals = Y - fitted_reduced
    _c, fitted_full = fit_ols(x_interaction, Y)
    observed = sum_of_squares(fitted_full - fitted_reduced)

    rng = np.random.default_rng(1)
    draws = []
    for _ in range(200):
        y_star = fitted_reduced + residuals[rng.permutation(Y.shape[0])]
        _c, ff = fit_ols(x_interaction, y_star)
        _c, fr = fit_ols(x_main, y_star)
        draws.append(sum_of_squares(ff - fr))

    assert 0.6 < np.mean(draws) / observed < 1.6


def test_null_interaction_p_values_are_not_pinned_at_one():
    """Across null replicates the p-values must spread over (0, 1] rather than
    collapsing onto 1.0, and reject at roughly the nominal rate."""
    p_values = []
    for rep in range(40):
        Y, x_main, x_interaction = _interaction_design(interaction_strength=0.0, seed=100 + rep)
        p_values.append(
            rrpp_test(Y, x_main, x_interaction, "interaction", n_permutations=199, seed=rep).p_value
        )
    p_values = np.array(p_values)

    assert np.mean(p_values == 1.0) < 0.15, "p-values collapsing onto 1.0 - permuted null inflated"
    assert 0.25 < np.median(p_values) < 0.75
    assert np.mean(p_values < 0.05) < 0.20


def test_a_genuine_interaction_is_still_detected_after_the_calibration_fix():
    for rep in range(5):
        Y, x_main, x_interaction = _interaction_design(interaction_strength=2.5, seed=500 + rep)
        result = rrpp_test(Y, x_main, x_interaction, "interaction", n_permutations=199, seed=rep)
        assert result.p_value < 0.05


def test_intercept_only_reduced_model_is_unaffected_by_the_fix():
    """Sanity check on the boundary case: with an intercept-only reduced model
    the permuted residuals stay orthogonal to it, so the old and new statistics
    agree - which is why main effects were never wrong."""
    rng = np.random.default_rng(3)
    n, p = 40, 10
    x = rng.normal(size=n)
    Y = rng.normal(scale=0.5, size=(n, p)) + np.outer(x, rng.normal(size=p))
    ones = np.ones((n, 1))
    full = np.hstack([ones, x[:, None]])

    result = rrpp_test(Y, ones, full, "x", n_permutations=299, seed=11)
    assert result.p_value < 0.05
    assert result.r2 > 0.3


# --- multiple-comparison adjustment -------------------------------------------


def test_holm_and_bh_adjustments_against_known_values():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    # Holm: multiply the k-th smallest by (n - k), then enforce monotonicity.
    holm = adjust_p_values(p, "Holm")
    np.testing.assert_allclose(holm, [0.05, 0.08, 0.09, 0.09, 0.09])
    # BH: multiply the k-th smallest by n/k, then enforce monotonicity from the top.
    bh = adjust_p_values(p, "Benjamini-Hochberg")
    np.testing.assert_allclose(bh, [0.05, 0.05, 0.05, 0.05, 0.05])


def test_adjustments_are_never_smaller_than_the_raw_p_values():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=25)
    for method in ["Holm", "Benjamini-Hochberg"]:
        adjusted = adjust_p_values(p, method)
        assert np.all(adjusted >= p - 1e-12), method
        assert np.all(adjusted <= 1.0), method


def test_holm_is_at_least_as_strict_as_benjamini_hochberg():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=30)
    assert np.all(adjust_p_values(p, "Holm") >= adjust_p_values(p, "Benjamini-Hochberg") - 1e-12)


def test_adjustment_preserves_ordering_and_handles_edges():
    p = np.array([0.2, 0.001, 0.5])
    adjusted = adjust_p_values(p, "Benjamini-Hochberg")
    assert np.argsort(adjusted).tolist() == np.argsort(p).tolist()
    assert adjust_p_values([], "Holm").size == 0
    np.testing.assert_allclose(adjust_p_values(p, "None"), p)


def test_rejects_unknown_adjustment_method():
    with pytest.raises(ValueError, match="Unknown p-value adjustment"):
        adjust_p_values([0.1, 0.2], "bonferroni-ish")
