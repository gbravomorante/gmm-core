import numpy as np
import pandas as pd
import pytest

from gmm_core.analysis.group_tests import run_group_test
from gmm_core.data_model import LABEL_COL, landmark_columns

N_LM = 4
COLS = landmark_columns(N_LM, n_dim=3)
P = N_LM * 3


def _make_gpa_df(shapes_flat, group, extra=None):
    n = shapes_flat.shape[0]
    data = {"ID": [f"s{i}" for i in range(n)], "Group": group}
    if extra:
        data.update(extra)
    for j, col in enumerate(COLS):
        data[col] = shapes_flat[:, j]
    return pd.DataFrame(data)


def test_omnibus_detects_strong_true_group_difference():
    rng = np.random.default_rng(1)
    n_per_group = 15
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group + ["C"] * n_per_group)
    n = len(group)
    shapes_flat = rng.normal(scale=0.2, size=(n, P))
    mean_shift = {
        "A": np.zeros(P),
        "B": rng.normal(scale=5.0, size=P),
        "C": rng.normal(scale=5.0, size=P),
    }
    for i, g in enumerate(group):
        shapes_flat[i] += mean_shift[g]

    gpa_df = _make_gpa_df(shapes_flat, group)
    result = run_group_test(
        gpa_df, COLS, 3, "Group", meta_cols=["Group"], n_permutations=299, seed=42
    )
    assert result.omnibus.p_value < 0.05
    assert result.omnibus.r2 > 0.3


def test_omnibus_null_case_not_significant():
    rng = np.random.default_rng(2)
    n_per_group = 15
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group + ["C"] * n_per_group)
    n = len(group)
    shapes_flat = rng.normal(scale=1.0, size=(n, P))  # no group effect at all

    gpa_df = _make_gpa_df(shapes_flat, group)
    result = run_group_test(
        gpa_df, COLS, 3, "Group", meta_cols=["Group"], n_permutations=299, seed=42
    )
    assert result.omnibus.p_value > 0.05


def test_pairwise_identifies_which_specific_pair_differs():
    rng = np.random.default_rng(3)
    n_per_group = 15
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group + ["C"] * n_per_group)
    n = len(group)
    shapes_flat = rng.normal(scale=0.2, size=(n, P))
    # Only C is shifted; A and B share the same mean shape.
    shift = rng.normal(scale=6.0, size=P)
    shapes_flat[group == "C"] += shift

    gpa_df = _make_gpa_df(shapes_flat, group)
    result = run_group_test(
        gpa_df, COLS, 3, "Group", meta_cols=["Group"], n_permutations=299, seed=7
    )

    pairwise_by_name = {t.term_name: t for t in result.pairwise}
    assert set(pairwise_by_name.keys()) == {"A vs B", "A vs C", "B vs C"}
    assert pairwise_by_name["A vs B"].p_value > 0.05
    assert pairwise_by_name["A vs C"].p_value < 0.05
    assert pairwise_by_name["B vs C"].p_value < 0.05


def test_covariate_rows_with_missing_values_excluded():
    rng = np.random.default_rng(4)
    n = 20
    group = np.array(["A"] * 10 + ["B"] * 10)
    shapes_flat = rng.normal(size=(n, P))
    age = rng.normal(size=n)
    age[0] = np.nan
    age[1] = np.nan

    gpa_df = _make_gpa_df(shapes_flat, group, extra={"Age": age})
    result = run_group_test(
        gpa_df,
        COLS,
        3,
        "Group",
        meta_cols=["Group", "Age"],
        covariate_cols=["Age"],
        n_permutations=99,
        seed=1,
    )
    assert result.n_excluded_missing_covariates == 2
    assert result.n_specimens == 18


def test_scores_include_label_and_distance_column():
    rng = np.random.default_rng(5)
    n = 16
    group = np.array(["A"] * 8 + ["B"] * 8)
    shapes_flat = rng.normal(size=(n, P))
    gpa_df = _make_gpa_df(shapes_flat, group)
    result = run_group_test(
        gpa_df,
        COLS,
        3,
        "Group",
        meta_cols=["Group"],
        n_permutations=49,
        seed=1,
        id_labels={"s0": "First"},
    )
    assert "Distance_to_group_mean" in result.scores.columns
    assert LABEL_COL in result.scores.columns
    assert result.scores.loc[result.scores["ID"] == "s0", LABEL_COL].iloc[0] == "First"
    assert (result.scores["Distance_to_group_mean"] >= 0).all()


def test_distance_to_group_mean_is_zero_for_identical_shapes_in_group():
    group = np.array(["A"] * 5 + ["B"] * 5)
    base_a = np.zeros(P)
    base_b = np.ones(P) * 3.0
    shapes_flat = np.stack([base_a] * 5 + [base_b] * 5)
    gpa_df = _make_gpa_df(shapes_flat, group)
    result = run_group_test(
        gpa_df, COLS, 3, "Group", meta_cols=["Group"], n_permutations=19, seed=1
    )
    assert np.allclose(result.scores["Distance_to_group_mean"], 0.0, atol=1e-8)


def test_rejects_fewer_than_4_specimens():
    shapes_flat = np.zeros((3, P))
    group = np.array(["A", "A", "B"])
    gpa_df = _make_gpa_df(shapes_flat, group)
    with pytest.raises(ValueError, match="Fewer than 4"):
        run_group_test(gpa_df, COLS, 3, "Group", meta_cols=["Group"])


def test_rejects_single_level_group():
    shapes_flat = np.zeros((6, P))
    group = np.array(["A"] * 6)
    gpa_df = _make_gpa_df(shapes_flat, group)
    with pytest.raises(ValueError, match="fewer than 2 levels"):
        run_group_test(gpa_df, COLS, 3, "Group", meta_cols=["Group"])


def test_model_formula_reflects_covariates():
    rng = np.random.default_rng(6)
    n = 20
    group = np.array(["A"] * 10 + ["B"] * 10)
    shapes_flat = rng.normal(size=(n, P))
    age = rng.normal(size=n)
    gpa_df = _make_gpa_df(shapes_flat, group, extra={"Age": age})
    result = run_group_test(
        gpa_df,
        COLS,
        3,
        "Group",
        meta_cols=["Group", "Age"],
        covariate_cols=["Age"],
        n_permutations=19,
        seed=1,
    )
    assert result.model_formula == "Shape ~ Age + Group (Group)"
