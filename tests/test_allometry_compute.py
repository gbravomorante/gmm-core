import numpy as np
import pandas as pd
import pytest

from gmm_core.analysis.allometry import run_allometry, run_pc_allometry, terms_to_dataframe
from gmm_core.data_model import LABEL_COL, landmark_columns

N_LM = 4
COLS = landmark_columns(N_LM, n_dim=3)
P = N_LM * 3


def _make_gpa_df(log_cs, shapes_flat, group=None):
    n = len(log_cs)
    data = {"ID": [f"s{i}" for i in range(n)], "CentroidSize": np.exp(log_cs)}
    if group is not None:
        data["Group"] = group
    for j, col in enumerate(COLS):
        data[col] = shapes_flat[:, j]
    return pd.DataFrame(data)


def test_run_allometry_detects_strong_true_allometry():
    rng = np.random.default_rng(1)
    n = 30
    log_cs = rng.normal(size=n)
    effect = rng.normal(scale=5.0, size=P)
    shapes_flat = rng.normal(scale=0.3, size=(n, P)) + np.outer(log_cs, effect)
    gpa_df = _make_gpa_df(log_cs, shapes_flat)

    result = run_allometry(
        gpa_df,
        COLS,
        3,
        gpa_df["CentroidSize"].to_numpy(),
        meta_cols=[],
        n_permutations=299,
        seed=42,
    )
    size_term = result.terms[0]
    assert size_term.term_name == "log(CentroidSize)"
    assert size_term.p_value < 0.05
    assert size_term.r2 > 0.5
    assert result.model_formula == "Shape ~ log(CentroidSize)"


def test_run_allometry_null_case_not_significant():
    rng = np.random.default_rng(2)
    n = 30
    log_cs = rng.normal(size=n)
    shapes_flat = rng.normal(scale=1.0, size=(n, P))  # no relationship to size
    gpa_df = _make_gpa_df(log_cs, shapes_flat)

    result = run_allometry(
        gpa_df,
        COLS,
        3,
        gpa_df["CentroidSize"].to_numpy(),
        meta_cols=[],
        n_permutations=299,
        seed=42,
    )
    assert result.terms[0].p_value > 0.05


def test_run_allometry_scores_include_label_and_regression_score():
    rng = np.random.default_rng(3)
    n = 20
    log_cs = rng.normal(size=n)
    shapes_flat = rng.normal(size=(n, P)) + np.outer(log_cs, rng.normal(size=P))
    gpa_df = _make_gpa_df(log_cs, shapes_flat)

    result = run_allometry(
        gpa_df,
        COLS,
        3,
        gpa_df["CentroidSize"].to_numpy(),
        meta_cols=[],
        n_permutations=99,
        seed=1,
        id_labels={"s0": "First"},
    )
    assert "RegressionScore" in result.scores.columns
    assert "Log_CentroidSize" in result.scores.columns
    assert LABEL_COL in result.scores.columns
    assert result.scores.loc[result.scores["ID"] == "s0", LABEL_COL].iloc[0] == "First"
    assert result.scores.loc[result.scores["ID"] == "s1", LABEL_COL].iloc[0] == "s1"


def test_run_allometry_group_interaction_detects_divergent_trajectories():
    rng = np.random.default_rng(4)
    n_per_group = 20
    log_cs = np.concatenate([rng.normal(size=n_per_group), rng.normal(size=n_per_group)])
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group)

    slope_effect = rng.normal(scale=4.0, size=P)
    shapes_flat = rng.normal(scale=0.3, size=(2 * n_per_group, P))
    shapes_flat[n_per_group:] += np.outer(
        log_cs[n_per_group:], slope_effect
    )  # only group B has a slope

    gpa_df = _make_gpa_df(log_cs, shapes_flat, group=group)
    result = run_allometry(
        gpa_df,
        COLS,
        3,
        gpa_df["CentroidSize"].to_numpy(),
        meta_cols=["Group"],
        group_col="Group",
        n_permutations=299,
        seed=7,
    )
    assert len(result.terms) == 3
    interaction_term = result.terms[2]
    assert "x Group" in interaction_term.term_name
    assert interaction_term.p_value < 0.05


def test_run_allometry_group_interaction_not_significant_with_common_slope():
    rng = np.random.default_rng(5)
    n_per_group = 20
    log_cs = np.concatenate([rng.normal(size=n_per_group), rng.normal(size=n_per_group)])
    group = np.array(["A"] * n_per_group + ["B"] * n_per_group)

    common_slope = rng.normal(scale=3.0, size=P)
    shapes_flat = rng.normal(scale=0.3, size=(2 * n_per_group, P)) + np.outer(log_cs, common_slope)

    gpa_df = _make_gpa_df(log_cs, shapes_flat, group=group)
    result = run_allometry(
        gpa_df,
        COLS,
        3,
        gpa_df["CentroidSize"].to_numpy(),
        meta_cols=["Group"],
        group_col="Group",
        n_permutations=299,
        seed=7,
    )
    assert result.terms[2].p_value > 0.05


def test_run_allometry_rejects_non_positive_centroid_size():
    rng = np.random.default_rng(6)
    n = 10
    log_cs = rng.normal(size=n)
    shapes_flat = rng.normal(size=(n, P))
    gpa_df = _make_gpa_df(log_cs, shapes_flat)
    bad_sizes = gpa_df["CentroidSize"].to_numpy().copy()
    bad_sizes[0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        run_allometry(gpa_df, COLS, 3, bad_sizes, meta_cols=[])


def test_run_allometry_rejects_single_level_group():
    rng = np.random.default_rng(7)
    n = 10
    log_cs = rng.normal(size=n)
    shapes_flat = rng.normal(size=(n, P))
    gpa_df = _make_gpa_df(log_cs, shapes_flat, group=["A"] * n)
    with pytest.raises(ValueError, match="fewer than 2 levels"):
        run_allometry(
            gpa_df,
            COLS,
            3,
            gpa_df["CentroidSize"].to_numpy(),
            meta_cols=["Group"],
            group_col="Group",
        )


def test_terms_to_dataframe_columns():
    rng = np.random.default_rng(8)
    n = 15
    log_cs = rng.normal(size=n)
    shapes_flat = rng.normal(size=(n, P))
    gpa_df = _make_gpa_df(log_cs, shapes_flat)
    result = run_allometry(
        gpa_df, COLS, 3, gpa_df["CentroidSize"].to_numpy(), meta_cols=[], n_permutations=49, seed=1
    )
    df = terms_to_dataframe(result.terms)
    assert list(df.columns) == [
        "Term",
        "SS_effect",
        "SS_total",
        "R2",
        "P_value",
        "N_permutations",
        "Random_seed",
    ]
    assert len(df) == 1


# --- custom (measured) size variable -----------------------------------------
# Allometry does not have to be against centroid size: a directly measured
# dimension is an equally valid size axis, and is what the dogs live-run needed
# ("Hipbone Length" already present as a metadata column).


def test_custom_size_label_flows_into_terms_formula_and_scores_column():
    rng = np.random.default_rng(11)
    n = 30
    log_size = rng.normal(size=n)
    effect = rng.normal(scale=5.0, size=P)
    shapes_flat = rng.normal(scale=0.3, size=(n, P)) + np.outer(log_size, effect)
    gpa_df = _make_gpa_df(log_size, shapes_flat)
    gpa_df["Hipbone Length"] = np.exp(log_size) * 10.0

    result = run_allometry(
        gpa_df,
        COLS,
        3,
        gpa_df["Hipbone Length"].to_numpy(),
        meta_cols=["Hipbone Length"],
        n_permutations=199,
        seed=7,
        size_label="Hipbone Length",
    )

    assert result.size_label == "Hipbone Length"
    assert result.size_column == "Log_Hipbone Length"
    assert result.size_column in result.scores.columns
    assert result.model_formula == "Shape ~ log(Hipbone Length)"
    assert result.terms[0].term_name == "log(Hipbone Length)"
    assert "Log_CentroidSize" not in result.scores.columns


def test_custom_size_label_appears_in_the_interaction_term_name():
    rng = np.random.default_rng(12)
    n = 40
    log_size = rng.normal(size=n)
    shapes_flat = rng.normal(scale=0.5, size=(n, P))
    group = ["A"] * 20 + ["B"] * 20
    gpa_df = _make_gpa_df(log_size, shapes_flat, group=group)

    result = run_allometry(
        gpa_df,
        COLS,
        3,
        np.exp(log_size),
        meta_cols=["Group"],
        group_col="Group",
        n_permutations=99,
        seed=3,
        size_label="Hipbone Length",
    )
    names = [t.term_name for t in result.terms]
    assert names == ["log(Hipbone Length)", "Group (Group)", "log(Hipbone Length) x Group (Group)"]
    assert result.model_formula == "Shape ~ log(Hipbone Length) * Group"


def test_a_proportional_size_variable_gives_the_same_result_as_centroid_size():
    """log(k * size) = log(k) + log(size), so a size variable that is a constant
    multiple of centroid size differs only in intercept - the allometric term's
    R^2 and p-value must be identical."""
    rng = np.random.default_rng(13)
    n = 35
    log_cs = rng.normal(size=n)
    effect = rng.normal(scale=4.0, size=P)
    shapes_flat = rng.normal(scale=0.3, size=(n, P)) + np.outer(log_cs, effect)
    gpa_df = _make_gpa_df(log_cs, shapes_flat)

    from_cs = run_allometry(
        gpa_df,
        COLS,
        3,
        gpa_df["CentroidSize"].to_numpy(),
        meta_cols=[],
        n_permutations=199,
        seed=5,
    )
    from_scaled = run_allometry(
        gpa_df,
        COLS,
        3,
        gpa_df["CentroidSize"].to_numpy() * 25.4,
        meta_cols=[],
        n_permutations=199,
        seed=5,
        size_label="LengthMM",
    )

    assert from_scaled.terms[0].r2 == pytest.approx(from_cs.terms[0].r2)
    assert from_scaled.terms[0].p_value == pytest.approx(from_cs.terms[0].p_value)


def test_rejects_non_positive_size_values_with_the_variable_named():
    rng = np.random.default_rng(14)
    n = 20
    log_cs = rng.normal(size=n)
    gpa_df = _make_gpa_df(log_cs, rng.normal(size=(n, P)))
    sizes = np.exp(log_cs)
    sizes[3] = 0.0

    with pytest.raises(ValueError, match="Hipbone Length"):
        run_allometry(gpa_df, COLS, 3, sizes, meta_cols=[], size_label="Hipbone Length")


def test_rejects_missing_size_values_with_the_variable_named():
    rng = np.random.default_rng(15)
    n = 20
    log_cs = rng.normal(size=n)
    gpa_df = _make_gpa_df(log_cs, rng.normal(size=(n, P)))
    sizes = np.exp(log_cs)
    sizes[5] = np.nan

    with pytest.raises(ValueError, match="missing or non-numeric"):
        run_allometry(gpa_df, COLS, 3, sizes, meta_cols=[], size_label="Hipbone Length")


# --- per-PC allometry ---------------------------------------------------------


def _pc_fixture(n=60, seed=0, size_on_pc=1):
    """Build shapes where log(size) drives ONE known direction, so we know
    which PC should carry the allometric signal."""
    rng = np.random.default_rng(seed)
    log_size = rng.normal(size=n)

    # Two orthogonal shape axes with very different variances, so PC order is
    # predictable; the size signal is injected into whichever one is chosen.
    axis_big = np.zeros(P)
    axis_big[0] = 1.0
    axis_small = np.zeros(P)
    axis_small[1] = 1.0

    big_scores = rng.normal(scale=3.0, size=n)
    small_scores = rng.normal(scale=0.5, size=n)
    if size_on_pc == 1:
        big_scores = log_size * 3.0
    else:
        small_scores = log_size * 0.5

    shapes_flat = (
        np.outer(big_scores, axis_big)
        + np.outer(small_scores, axis_small)
        + rng.normal(scale=0.02, size=(n, P))
    )
    return _make_gpa_df(log_size, shapes_flat), np.exp(log_size)


def test_finds_allometry_on_the_pc_that_actually_carries_it():
    gpa_df, sizes = _pc_fixture(seed=1, size_on_pc=1)
    result = run_pc_allometry(
        gpa_df, COLS, 3, sizes, meta_cols=[], n_pcs=3, n_permutations=199, seed=42
    )
    t = result.table.set_index("PC")
    assert t.loc["PC1", "P_adjusted"] < 0.05
    assert t.loc["PC1", "R2"] > 0.9
    assert t.loc["PC2", "P_adjusted"] > 0.05


def test_finds_allometry_on_a_minor_pc_when_that_is_where_it_lives():
    """The point of the feature: a low-variance PC can be the allometric one."""
    gpa_df, sizes = _pc_fixture(seed=2, size_on_pc=2)
    result = run_pc_allometry(
        gpa_df, COLS, 3, sizes, meta_cols=[], n_pcs=3, n_permutations=199, seed=42
    )
    t = result.table.set_index("PC")
    assert t.loc["PC2", "P_adjusted"] < 0.05
    assert t.loc["PC1", "P_adjusted"] > 0.05
    # ...and PC2 explains far less variance than PC1, which is exactly why the
    # whole-shape test alone could miss it.
    assert t.loc["PC2", "Percent_variance"] < t.loc["PC1", "Percent_variance"]


def test_table_shape_columns_and_variance_accounting():
    gpa_df, sizes = _pc_fixture(seed=3)
    result = run_pc_allometry(
        gpa_df, COLS, 3, sizes, meta_cols=[], n_pcs=4, n_permutations=99, seed=1
    )
    assert list(result.table["PC"]) == ["PC1", "PC2", "PC3", "PC4"]
    assert set(
        ["PC", "Percent_variance", "Slope_on_log_size", "R2", "P_value", "P_adjusted"]
    ).issubset(result.table.columns)
    assert 0 < result.total_variance_covered <= 100.0 + 1e-9
    assert result.table["Percent_variance"].is_monotonic_decreasing
    for k in range(1, 5):
        assert f"PC{k}" in result.scores.columns
    assert result.size_column in result.scores.columns


def test_group_and_interaction_columns_appear_only_with_a_group_column():
    gpa_df, sizes = _pc_fixture(n=40, seed=4)
    gpa_df["Group"] = ["A"] * 20 + ["B"] * 20

    without = run_pc_allometry(
        gpa_df, COLS, 3, sizes, meta_cols=["Group"], n_pcs=2, n_permutations=99, seed=1
    )
    assert "Group_P_value" not in without.table.columns

    with_group = run_pc_allometry(
        gpa_df,
        COLS,
        3,
        sizes,
        meta_cols=["Group"],
        n_pcs=2,
        group_col="Group",
        n_permutations=99,
        seed=1,
    )
    for col in [
        "Group_R2",
        "Group_P_value",
        "Group_P_adjusted",
        "Interaction_R2",
        "Interaction_P_value",
        "Interaction_P_adjusted",
    ]:
        assert col in with_group.table.columns


def test_n_pcs_capped_to_what_the_sample_supports():
    gpa_df, sizes = _pc_fixture(n=8, seed=5)
    result = run_pc_allometry(
        gpa_df, COLS, 3, sizes, meta_cols=[], n_pcs=99, n_permutations=49, seed=1
    )
    assert result.n_pcs == min(8 - 1, P)
    assert len(result.table) == result.n_pcs


def test_adjusted_p_is_never_below_the_raw_p():
    gpa_df, sizes = _pc_fixture(n=50, seed=6)
    result = run_pc_allometry(
        gpa_df, COLS, 3, sizes, meta_cols=[], n_pcs=5, n_permutations=99, seed=2
    )
    assert (result.table["P_adjusted"] >= result.table["P_value"] - 1e-12).all()


def test_custom_size_label_used_for_the_pc_scores_column():
    gpa_df, sizes = _pc_fixture(n=30, seed=7)
    result = run_pc_allometry(
        gpa_df,
        COLS,
        3,
        sizes,
        meta_cols=[],
        n_pcs=2,
        n_permutations=49,
        seed=1,
        size_label="Hipbone Length",
    )
    assert result.size_column == "Log_Hipbone Length"
    assert "Log_Hipbone Length" in result.scores.columns


def test_rejects_bad_size_values_and_bad_pc_count():
    gpa_df, sizes = _pc_fixture(n=30, seed=8)
    bad = sizes.copy()
    bad[0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        run_pc_allometry(gpa_df, COLS, 3, bad, meta_cols=[], n_pcs=2)
    with pytest.raises(ValueError, match="at least 1 principal component"):
        run_pc_allometry(gpa_df, COLS, 3, sizes, meta_cols=[], n_pcs=0)


# --- is the chosen size variable actually a size measure? -------------------


def test_a_size_variable_that_tracks_centroid_size_is_not_warned_about():
    from gmm_core.analysis.allometry import size_variable_warnings

    rng = np.random.default_rng(0)
    centroid = rng.uniform(50, 120, 200)
    measured = centroid * 0.8 + rng.normal(0, 4, 200)  # a real dimension
    assert size_variable_warnings(measured, centroid, "Hipbone Length") == []


def test_a_numeric_column_that_is_not_a_size_measure_is_warned_about():
    """A disparity score or an age is numeric, and nothing in its NAME says
    it is not size. The correlation with centroid size does."""
    from gmm_core.analysis.allometry import size_variable_warnings

    rng = np.random.default_rng(1)
    centroid = rng.uniform(50, 120, 200)
    unrelated = rng.normal(0, 1, 200) + 0.2 * (centroid - centroid.mean()) / 20
    warnings = size_variable_warnings(unrelated, centroid, "Procrustes_Disparity")
    assert warnings
    assert "Procrustes_Disparity" in warnings[0]
    assert "centroid size" in warnings[0]


def test_centroid_size_never_warns_about_itself():
    from gmm_core.analysis.allometry import size_variable_warnings

    centroid = np.random.default_rng(2).uniform(50, 120, 50)
    assert size_variable_warnings(centroid, centroid, "CentroidSize") == []


def test_the_size_check_is_silent_when_there_is_nothing_to_compare_against():
    from gmm_core.analysis.allometry import size_variable_warnings

    values = np.arange(10.0) + 1
    assert size_variable_warnings(values, None, "Whatever") == []
    assert size_variable_warnings(values[:2], values[:2], "Whatever") == []
    assert size_variable_warnings(np.ones(10), np.arange(10.0), "Constant") == []


def test_a_negatively_correlated_size_variable_is_warned_about():
    from gmm_core.analysis.allometry import size_variable_warnings

    centroid = np.linspace(50, 120, 60)
    warnings = size_variable_warnings(-centroid, centroid, "Backwards")
    assert warnings and "runs opposite to" in warnings[0]
