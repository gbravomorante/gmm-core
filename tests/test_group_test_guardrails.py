"""Design guardrails on group tests.

A typical trap: several observers each landmark mostly one group. The apparent
group difference is then largely an observer effect, and an unguarded test
reports it as biology. In one real case, an apparent population effect of
R2 = 0.080 fell to R2 = 0.027 once the observer was accounted for.

These are warnings, never blocks. The point is to make an analyst look.
"""

import numpy as np
import pandas as pd
import pytest

from gmm_core.analysis.group_tests import (
    SIZE_RATIO_WARN,
    _cramers_v,
    group_test_warnings,
)


def frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


# --- Cramer's V ------------------------------------------------------------


def test_independent_columns_score_near_zero():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.choice(["x", "y"], 400))
    b = pd.Series(rng.choice(["p", "q"], 400))
    assert _cramers_v(a, b) < 0.2


def test_identical_columns_score_one():
    a = pd.Series(["x", "x", "y", "y"] * 20)
    assert _cramers_v(a, a.copy()) == pytest.approx(1.0, abs=1e-9)


def test_a_single_level_cannot_be_associated_with_anything():
    a = pd.Series(["only"] * 50)
    b = pd.Series(["p", "q"] * 25)
    assert _cramers_v(a, b) == 0.0


# --- the confound warning, which is the point ------------------------------


def test_a_group_perfectly_confounded_with_an_observer_is_flagged():
    """The real case: each population digitised by one person."""
    df = frame(
        Group=["A"] * 30 + ["B"] * 30,
        Student=["ann"] * 30 + ["bob"] * 30,
        CentroidSize=[10.0] * 60,
    )
    warnings = group_test_warnings(df, "Group", [], ["Student", "CentroidSize"])
    assert any("entangled with 'Student'" in w for w in warnings)
    assert any("inter-observer" in w for w in warnings)


def test_a_balanced_design_is_not_flagged_as_confounded():
    """Every observer measured every group - nothing to warn about."""
    df = frame(
        Group=["A", "B"] * 30,
        Student=["ann", "ann", "bob", "bob"] * 15,
        CentroidSize=[10.0] * 60,
    )
    warnings = group_test_warnings(df, "Group", [], ["Student", "CentroidSize"])
    assert not any("entangled" in w for w in warnings)


def test_a_column_already_used_as_a_covariate_is_not_flagged():
    """If the analyst already accounted for it, saying so again is noise."""
    df = frame(Group=["A"] * 30 + ["B"] * 30, Student=["ann"] * 30 + ["bob"] * 30)
    warnings = group_test_warnings(df, "Group", ["Student"], ["Student"])
    assert not any("entangled" in w for w in warnings)


def test_an_id_like_column_is_not_treated_as_a_confound():
    """A column with a distinct value per specimen is an identifier, not a
    design variable, and would otherwise fire on every dataset."""
    df = frame(Group=["A"] * 30 + ["B"] * 30, ID=[f"s{i}" for i in range(60)])
    warnings = group_test_warnings(df, "Group", [], ["ID"])
    assert not any("entangled" in w for w in warnings)


# --- the size warning ------------------------------------------------------


def test_groups_that_differ_in_size_are_flagged():
    df = frame(Group=["small"] * 30 + ["big"] * 30, CentroidSize=[10.0] * 30 + [14.0] * 30)
    warnings = group_test_warnings(df, "Group", [], ["CentroidSize"])
    assert any("differ in SIZE" in w for w in warnings)
    assert any("allometric" in w for w in warnings)


def test_groups_of_equal_size_are_not_flagged():
    df = frame(Group=["a"] * 30 + ["b"] * 30, CentroidSize=[10.0] * 60)
    warnings = group_test_warnings(df, "Group", [], ["CentroidSize"])
    assert not any("differ in SIZE" in w for w in warnings)


def test_the_size_threshold_is_the_documented_one():
    just_under = frame(
        Group=["a"] * 10 + ["b"] * 10,
        CentroidSize=[10.0] * 10 + [10.0 * (SIZE_RATIO_WARN - 0.02)] * 10,
    )
    assert not any(
        "differ in SIZE" in w
        for w in group_test_warnings(just_under, "Group", [], ["CentroidSize"])
    )


# --- sample-size warnings --------------------------------------------------


def test_a_group_of_two_is_flagged_as_too_small():
    df = frame(Group=["a"] * 30 + ["rare", "rare"])
    warnings = group_test_warnings(df, "Group", [], [])
    assert any("fewer than" in w and "rare" in w for w in warnings)


def test_severe_imbalance_is_flagged():
    df = frame(Group=["big"] * 200 + ["small"] * 5)
    warnings = group_test_warnings(df, "Group", [], [])
    assert any("unbalanced" in w for w in warnings)


def test_a_clean_balanced_design_produces_no_warnings_at_all():
    """The guardrails must stay quiet on a good design, or they become noise
    that gets ignored on a bad one."""
    df = frame(Group=["a"] * 40 + ["b"] * 40, Student=["ann", "bob"] * 40, CentroidSize=[10.0] * 80)
    assert group_test_warnings(df, "Group", [], ["Student", "CentroidSize"]) == []


def test_an_unknown_group_column_is_survivable():
    assert group_test_warnings(frame(A=[1, 2]), "NotThere", [], []) == []


# --- reaching the caller ---------------------------------------------------


def test_the_warnings_reach_the_result_object():
    from gmm_core.analysis.group_tests import run_group_test
    from gmm_core.data_model import landmarks_to_row_dict
    from gmm_core.gpa import generalized_procrustes

    rng = np.random.default_rng(3)
    rows, coords = [], []
    for i in range(24):
        group = "A" if i < 12 else "B"
        base = rng.normal(size=(6, 3)) * 0.4 + np.arange(6)[:, None]
        if group == "B":
            base = base + 0.8
        rows.append({"ID": f"s{i}", "Group": group, "Student": "ann" if group == "A" else "bob"})
        coords.append(base)

    aligned = generalized_procrustes(np.stack(coords), scale=True)
    frame_rows = []
    for i, row in enumerate(rows):
        r = dict(row)
        r.update(landmarks_to_row_dict(aligned.aligned[i], 3))
        r["CentroidSize"] = float(aligned.centroid_sizes[i])
        frame_rows.append(r)
    df = pd.DataFrame(frame_rows)
    lm = [c for c in df.columns if c.startswith("LM")]

    result = run_group_test(
        df,
        lm,
        3,
        group_col="Group",
        meta_cols=["Student", "CentroidSize"],
        n_permutations=99,
        seed=1,
    )
    assert any("entangled with 'Student'" in w for w in result.warnings)
