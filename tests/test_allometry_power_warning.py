"""A non-significant interaction is not evidence of a shared trajectory.

If one group's span of log(size) is a small fraction of another's, the null
of a common slope is uninformative. In a comparison of domestic dogs (2.96x
range in centroid size) with wolves (1.42x), a power simulation on that exact
design detected an injected slope difference of 10% or 25% in 0 of 20
replicates. The correct wording is "no detectable difference, and the design
has little power to find one", and the warning tested here says so.
"""

import numpy as np

from gmm_core.analysis.allometry import SPAN_RATIO_WARN, allometry_power_warnings
from gmm_core.regression import RRPPResult


def term(name: str, p: float) -> RRPPResult:
    return RRPPResult(
        term_name=name, ss_effect=1.0, ss_total=10.0, r2=0.1, p_value=p, n_permutations=999, seed=1
    )


def sizes(narrow_span: float, wide_span: float, n: int = 30):
    """Two groups, one measured over a much narrower size range than the other."""
    narrow = np.linspace(0.0, np.log(narrow_span), n)
    wide = np.linspace(0.0, np.log(wide_span), n)
    log_size = np.concatenate([narrow, wide])
    groups = np.array(["narrow"] * n + ["wide"] * n)
    return log_size, groups


INTERACTION = "log(CentroidSize) x Group (Species)"


def test_unequal_spans_with_a_null_interaction_are_flagged():
    log_size, groups = sizes(1.42, 2.96)  # the real dogs numbers
    warnings = allometry_power_warnings(
        log_size, groups, [term(INTERACTION, 0.748)], "CentroidSize"
    )
    assert len(warnings) == 1
    assert "little power" in warnings[0]
    assert "NOT as evidence" in warnings[0]


def test_the_measured_spans_are_reported_so_they_can_be_checked():
    log_size, groups = sizes(1.42, 2.96)
    warnings = allometry_power_warnings(
        log_size, groups, [term(INTERACTION, 0.748)], "CentroidSize"
    )
    assert "1.42x" in warnings[0]
    assert "2.96x" in warnings[0]


def test_a_SIGNIFICANT_interaction_is_never_flagged():
    """Low power cannot manufacture a positive result, so a significant term
    stands on its own and a warning there would be pure noise."""
    log_size, groups = sizes(1.42, 2.96)
    assert (
        allometry_power_warnings(log_size, groups, [term(INTERACTION, 0.004)], "CentroidSize") == []
    )


def test_comparable_spans_are_not_flagged():
    log_size, groups = sizes(2.80, 2.96)
    assert (
        allometry_power_warnings(log_size, groups, [term(INTERACTION, 0.748)], "CentroidSize") == []
    )


def test_no_group_term_means_nothing_to_warn_about():
    log_size, _ = sizes(1.42, 2.96)
    assert (
        allometry_power_warnings(log_size, None, [term("log(CentroidSize)", 0.001)], "CentroidSize")
        == []
    )


def test_no_interaction_term_means_nothing_to_warn_about():
    log_size, groups = sizes(1.42, 2.96)
    assert (
        allometry_power_warnings(log_size, groups, [term("Group (Species)", 0.9)], "CentroidSize")
        == []
    )


def test_a_group_with_no_size_range_at_all_is_called_out_separately():
    n = 20
    log_size = np.concatenate([np.zeros(n), np.linspace(0, 1.0, n)])
    groups = np.array(["flat"] * n + ["wide"] * n)
    warnings = allometry_power_warnings(log_size, groups, [term(INTERACTION, 0.5)], "CentroidSize")
    assert len(warnings) == 1
    assert "spans no range" in warnings[0]


def test_the_threshold_is_the_documented_one():
    just_under = np.exp(np.log(2.96) / (SPAN_RATIO_WARN - 0.3))
    log_size, groups = sizes(just_under, 2.96)
    assert (
        allometry_power_warnings(log_size, groups, [term(INTERACTION, 0.7)], "CentroidSize") == []
    )


def test_the_warning_reaches_the_result_object():
    """Guarding the whole path: compute -> AllometryResult.warnings."""
    import pandas as pd

    from gmm_core.analysis.allometry import run_allometry
    from gmm_core.data_model import landmarks_to_row_dict
    from gmm_core.gpa import generalized_procrustes

    rng = np.random.default_rng(5)
    coords, rows, size = [], [], []
    for i in range(60):
        group = "narrow" if i < 30 else "wide"
        scale = (
            (1.0 + 0.4 * (i % 30) / 29.0) if group == "narrow" else (1.0 + 2.0 * (i % 30) / 29.0)
        )
        base = (rng.normal(size=(6, 3)) * 0.25 + np.arange(6)[:, None]) * scale
        coords.append(base)
        rows.append({"ID": f"s{i}", "Species": group})
        size.append(scale)

    aligned = generalized_procrustes(np.stack(coords), scale=True)
    frame_rows = []
    for i, row in enumerate(rows):
        r = dict(row)
        r.update(landmarks_to_row_dict(aligned.aligned[i], 3))
        r["CentroidSize"] = float(aligned.centroid_sizes[i])
        frame_rows.append(r)
    df = pd.DataFrame(frame_rows)
    lm = [c for c in df.columns if c.startswith("LM")]

    result = run_allometry(
        df,
        lm,
        3,
        size_values=df["CentroidSize"].to_numpy(float),
        meta_cols=["Species"],
        group_col="Species",
        n_permutations=99,
        seed=1,
    )
    interaction = next(t for t in result.terms if " x " in t.term_name)
    if interaction.p_value >= 0.05:
        assert any("little power" in w for w in result.warnings)
