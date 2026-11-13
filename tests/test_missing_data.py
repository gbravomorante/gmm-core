"""Incomplete specimens: coverage, shared-landmark distances, staged estimation."""

import numpy as np
import pandas as pd
import pytest

from gmm_core.missing_data import (
    coverage_tables,
    estimate_missing_pca,
    estimate_missing_tps,
    greedy_landmark_drop,
    observed_only_test,
    partial_procrustes_distance,
    pattern_validation,
    present_mask,
    reconstruct_batch,
    shared_distance_matrix,
)


def _population(n=30, p=14, seed=1, scale=0.05):
    """A curved base shape with one mode of variation, in varied frames and sizes."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, p, endpoint=False)
    base = np.c_[np.cos(theta), np.sin(theta) * 0.7, np.sin(2 * theta) * 0.3]
    mode = rng.normal(size=base.shape)
    shapes = []
    for _ in range(n):
        q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        s = base + rng.normal() * scale * mode + rng.normal(scale=0.002, size=base.shape)
        shapes.append(s @ q * rng.uniform(0.8, 1.3) + rng.normal(size=3))
    return np.stack(shapes)


def test_coverage_counts_the_right_landmarks_and_specimens():
    shapes = _population(5, 6)
    shapes[0, 2] = np.nan
    shapes[1, 2] = np.nan
    tables = coverage_tables(shapes, list("abcde"))
    assert tables["by_landmark"].iloc[0].to_dict()["Landmark"] == "LM2"
    assert tables["by_landmark"].iloc[0]["N_present"] == 3
    assert tables["by_specimen"].set_index("ID").loc["a", "N_missing"] == 1


def test_a_partly_missing_landmark_counts_as_missing():
    shapes = _population(3, 5)
    shapes[0, 1, 2] = np.nan
    assert not present_mask(shapes)[0, 1]


def test_greedy_drop_removes_the_landmark_that_frees_most_specimens():
    mask = np.ones((6, 5), dtype=bool)
    mask[[0, 1, 2], 3] = False
    mask[4, 1] = False
    steps = greedy_landmark_drop(mask)
    assert steps.iloc[0]["N_complete"] == 2
    assert steps.iloc[1]["Dropped"] == 3 and steps.iloc[1]["N_complete"] == 5
    assert steps.iloc[2]["Dropped"] == 1 and steps.iloc[2]["N_complete"] == 6


def test_distance_ignores_frame_and_size_but_not_reflection():
    shape = _population(1, 10)[0]
    q, _r = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    assert partial_procrustes_distance(shape, shape @ q * 3 + 5) == pytest.approx(0, abs=1e-6)
    mirrored = shape * np.array([-1, 1, 1])
    assert partial_procrustes_distance(shape, mirrored) > 0.05


def test_shared_distances_use_only_shared_landmarks():
    shapes = _population(3, 10)
    expected = partial_procrustes_distance(shapes[0][2:], shapes[1][2:])
    shapes[0, :2] = np.nan
    distances, shared = shared_distance_matrix(shapes)
    assert shared[0, 1] == 8
    assert distances[0, 1] == pytest.approx(expected)
    assert np.allclose(distances, distances.T, equal_nan=True)


def test_too_few_shared_landmarks_gives_nan_not_a_number():
    shapes = _population(2, 6)
    shapes[0, :3] = np.nan
    shapes[1, 3:] = np.nan
    distances, shared = shared_distance_matrix(shapes)
    assert shared[0, 1] == 0 and np.isnan(distances[0, 1])


def test_observed_only_test_separates_a_real_outlier_from_a_member():
    reference = _population(40, 14)
    member = _population(41, 14)[40].copy()
    outlier = member + np.random.default_rng(9).normal(scale=0.2, size=member.shape)
    member[:4] = np.nan
    outlier[:4] = np.nan
    assert observed_only_test(reference, member).upper_tail > 0.05
    result = observed_only_test(reference, outlier)
    assert result.upper_tail < 0.05 and result.n_landmarks == 10


def test_observed_only_null_is_not_biased_tight_for_a_true_member():
    """Left-out reference specimens and a fresh member are scored the same way, so a
    member's percentile should be unremarkable on average, not near 100."""
    population = _population(60, 14, seed=4)
    percentiles = [
        observed_only_test(np.delete(population, k, axis=0), population[k]).percentile
        for k in range(0, 60, 6)
    ]
    assert 25 < np.mean(percentiles) < 75


def test_tps_keeps_observed_values_exactly_and_recovers_hidden_ones():
    shapes = _population(30, 14)
    target = shapes[0].copy()
    target[3:5] = np.nan
    filled = estimate_missing_tps(shapes[1:], target)
    assert np.array_equal(filled[5:], shapes[0][5:])
    naive = np.linalg.norm(shapes[0][3:5] - shapes[0].mean(axis=0), axis=1).mean()
    assert np.linalg.norm(filled[3:5] - shapes[0][3:5], axis=1).mean() < 0.2 * naive


def test_pca_estimate_returns_the_targets_own_frame():
    shapes = _population(30, 14)
    target = shapes[0].copy()
    target[3:5] = np.nan
    filled, converged, _ = estimate_missing_pca(shapes[1:], target, n_pcs=3)
    assert converged
    assert np.linalg.norm(filled[3:5] - shapes[0][3:5], axis=1).mean() < 0.3


def test_batch_runs_in_stages_and_flags_every_estimate():
    shapes = _population(20, 14)
    shapes[18, :2] = np.nan
    shapes[19, 2:4] = np.nan
    ids = [f"s{i}" for i in range(20)]
    result = reconstruct_batch(shapes, ids, stages=[["s18"], ["s19"]])
    assert not np.isnan(result.complete).any()
    assert result.estimated_mask[18, 0] and not result.estimated_mask[18, 5]
    report = result.report.set_index("ID")
    assert report.loc["s18", "Stage"] == 1 and report.loc["s18", "N_reference"] == 18
    assert report.loc["s19", "Stage"] == 2 and report.loc["s19", "N_reference"] == 19


def test_batch_refuses_unknown_ids_and_no_reference():
    shapes = _population(5, 8)
    damaged = shapes.copy()
    damaged[0, 0] = np.nan
    with pytest.raises(ValueError, match="Unknown IDs"):
        reconstruct_batch(damaged, list("abcde"), stages=[["zzz"]])
    empty = shapes.copy()
    empty[:, 0] = np.nan
    with pytest.raises(ValueError, match="No complete specimen"):
        reconstruct_batch(empty, list("abcde"))


def test_batch_reports_convergence():
    shapes = _population(20, 14)
    shapes[19, :9] = np.nan
    result = reconstruct_batch(shapes, [str(i) for i in range(20)], method="pca")
    assert isinstance(result.report, pd.DataFrame) and "Converged" in result.report.columns


def test_pattern_validation_error_grows_with_damage():
    shapes = _population(25, 14)
    light = np.zeros(14, dtype=bool)
    light[3] = True
    heavy = np.zeros(14, dtype=bool)
    heavy[1:9] = True
    a = pattern_validation(shapes, light)
    b = pattern_validation(shapes, heavy)
    assert len(a) == 25
    assert b["Procrustes_error"].mean() > a["Procrustes_error"].mean()
