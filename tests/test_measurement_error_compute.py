import numpy as np
import pandas as pd
import pytest

from gmm_core.analysis.measurement_error import (
    measurement_error_summary_to_dataframe,
    run_measurement_error_analysis,
)
from gmm_core.data_model import landmark_columns

N_LM = 6
COLS = landmark_columns(N_LM, n_dim=3)
P = N_LM * 3


def _make_df(shapes_flat, ids, group, observer=None):
    data = {"ID": ids, "Specimen": group}
    if observer is not None:
        data["Observer"] = observer
    for j, col in enumerate(COLS):
        data[col] = shapes_flat[:, j]
    return pd.DataFrame(data)


def test_perfect_repeatability_gives_zero_tem_and_full_reliability():
    rng = np.random.default_rng(1)
    n_specimens = 10
    base_shapes = rng.normal(scale=1.0, size=(n_specimens, N_LM, 3))
    # Each specimen digitized twice, with an EXACT duplicate (zero measurement error).
    shapes = np.repeat(base_shapes, 2, axis=0)
    ids = [f"s{i}_{'a' if r == 0 else 'b'}" for i in range(n_specimens) for r in range(2)]
    group = [f"s{i}" for i in range(n_specimens) for _ in range(2)]
    shapes_flat = shapes.reshape(2 * n_specimens, P)
    raw_df = _make_df(shapes_flat, ids, group)

    result = run_measurement_error_analysis(raw_df, COLS, 3, "Specimen")
    assert result.overall_tem == pytest.approx(0.0, abs=1e-6)
    assert result.overall_r == pytest.approx(1.0, abs=1e-6)
    assert len(result.pairs) == n_specimens


def test_known_tem_for_a_single_pair_group():
    # A single group with 2 replicates and a hand-computable Procrustes distance,
    # plus other single-replicate groups to establish "total variance".
    rng = np.random.default_rng(2)
    other_shapes = rng.normal(scale=1.0, size=(8, N_LM, 3))
    # Two replicates of the "test" specimen with a small, known offset in one landmark.
    rep_a = rng.normal(scale=1.0, size=(N_LM, 3))
    rep_b = rep_a.copy()
    rep_b[0, 0] += 5.0  # a real (large, easily detectable) digitizing discrepancy

    ids = [f"o{i}" for i in range(8)] + ["t_a", "t_b"]
    group = [f"o{i}" for i in range(8)] + ["test", "test"]
    shapes = np.concatenate([other_shapes, rep_a[None, :, :], rep_b[None, :, :]], axis=0)
    shapes_flat = shapes.reshape(10, P)
    raw_df = _make_df(shapes_flat, ids, group)

    result = run_measurement_error_analysis(raw_df, COLS, 3, "Specimen")
    assert len(result.pairs) == 1
    assert result.pairs[0].procrustes_distance > 0
    # TEM for a single pair reduces to distance / sqrt(2).
    expected_tem = result.pairs[0].procrustes_distance / np.sqrt(2)
    assert result.overall_tem == pytest.approx(expected_tem, rel=1e-6)


def test_intra_observer_error_smaller_than_inter_observer_error():
    rng = np.random.default_rng(3)
    n_specimens = 12
    base_shapes = rng.normal(scale=1.0, size=(n_specimens, N_LM, 3))

    ids, group, observer, rows = [], [], [], []
    for i in range(n_specimens):
        base = base_shapes[i]
        # Two intra-observer replicates from Observer A (tiny noise).
        rep_a1 = base + rng.normal(scale=0.02, size=(N_LM, 3))
        rep_a2 = base + rng.normal(scale=0.02, size=(N_LM, 3))
        # One replicate from Observer B with much larger systematic difference.
        rep_b1 = base + rng.normal(scale=2.0, size=(N_LM, 3))
        for suffix, rep, obs in [("A1", rep_a1, "A"), ("A2", rep_a2, "A"), ("B1", rep_b1, "B")]:
            ids.append(f"s{i}_{suffix}")
            group.append(f"s{i}")
            observer.append(obs)
            rows.append(rep)

    shapes_flat = np.stack(rows).reshape(len(rows), P)
    raw_df = _make_df(shapes_flat, ids, group, observer=observer)

    result = run_measurement_error_analysis(raw_df, COLS, 3, "Specimen", observer_col="Observer")
    assert result.intra_tem is not None and result.inter_tem is not None
    assert result.intra_tem < result.inter_tem
    assert result.intra_n_pairs == n_specimens  # exactly one A-A pair per specimen
    assert result.inter_n_pairs == 2 * n_specimens  # A1-B1 and A2-B1 per specimen


def test_single_replicate_groups_are_skipped_not_errored():
    rng = np.random.default_rng(4)
    shapes_flat = rng.normal(size=(6, P))
    ids = [f"s{i}" for i in range(6)]
    group = ["g1", "g1", "g2", "g3", "g3", "g3"]
    raw_df = _make_df(shapes_flat, ids, group)

    result = run_measurement_error_analysis(raw_df, COLS, 3, "Specimen")
    assert result.n_groups_with_replicates == 2  # g1 (2 reps) and g3 (3 reps -> 3 pairs)
    assert result.n_groups_skipped_single == 1  # g2
    assert len(result.pairs) == 1 + 3  # g1: 1 pair, g3: C(3,2)=3 pairs


def test_rejects_unknown_group_column():
    shapes_flat = np.zeros((6, P))
    ids = [f"s{i}" for i in range(6)]
    group = ["g1"] * 6
    raw_df = _make_df(shapes_flat, ids, group)
    with pytest.raises(ValueError, match="not found"):
        run_measurement_error_analysis(raw_df, COLS, 3, "Ghost")


def test_rejects_unknown_observer_column():
    shapes_flat = np.zeros((6, P))
    ids = [f"s{i}" for i in range(6)]
    group = ["g1"] * 6
    raw_df = _make_df(shapes_flat, ids, group)
    with pytest.raises(ValueError, match="not found"):
        run_measurement_error_analysis(raw_df, COLS, 3, "Specimen", observer_col="Ghost")


def test_rejects_no_repeated_groups():
    rng = np.random.default_rng(5)
    shapes_flat = rng.normal(size=(6, P))
    ids = [f"s{i}" for i in range(6)]
    group = [f"g{i}" for i in range(6)]  # every specimen its own group, no repeats
    raw_df = _make_df(shapes_flat, ids, group)
    with pytest.raises(ValueError, match="2 or more repeat digitizations"):
        run_measurement_error_analysis(raw_df, COLS, 3, "Specimen")


def test_rows_with_missing_group_value_are_excluded_not_crashed():
    # Regression test: a genuinely missing (NaN) group_col cell must be excluded from this
    # analysis and reported, not crash it - see the matching test in test_symmetry_compute.py
    # for the full explanation (pandas' newer default string dtype keeps a missing cell as a
    # real float NaN even in an already-str column).
    rng = np.random.default_rng(9)
    n_specimens = 6
    base_shapes = rng.normal(size=(n_specimens, N_LM, 3))
    shapes = np.repeat(base_shapes, 2, axis=0)
    ids = [f"s{i}_{r}" for i in range(n_specimens) for r in range(2)]
    group = [f"s{i}" for i in range(n_specimens) for _ in range(2)]
    shapes_flat = shapes.reshape(2 * n_specimens, P)
    raw_df = _make_df(shapes_flat, ids, group)

    extra_shapes = rng.normal(size=(2, P))
    extra_df = _make_df(extra_shapes, ["X0", "X1"], [np.nan, np.nan])
    raw_df = pd.concat([raw_df, extra_df], ignore_index=True)

    result = run_measurement_error_analysis(raw_df, COLS, 3, "Specimen")
    assert result.n_excluded_missing_group == 2
    assert len(result.pairs) == n_specimens


def test_missing_observer_value_marks_pair_unspecified_not_wrongly_inter():
    # Regression test: comparing a real observer string against a missing (NaN) observer value
    # with == is always False, so without an explicit missing-value check a replicate pair
    # would be silently mislabelled "inter" instead of "unspecified".
    rng = np.random.default_rng(10)
    base = rng.normal(size=(N_LM, 3))
    rep_a = base + rng.normal(scale=0.02, size=(N_LM, 3))
    rep_b = base + rng.normal(scale=0.02, size=(N_LM, 3))
    other_shapes = rng.normal(size=(4, N_LM, 3))

    ids = ["t_a", "t_b"] + [f"o{i}" for i in range(4)]
    group = ["test", "test"] + [f"o{i}" for i in range(4)]
    observer = [np.nan, "A"] + ["A"] * 4
    shapes = np.concatenate([rep_a[None], rep_b[None], other_shapes], axis=0)
    shapes_flat = shapes.reshape(6, P)
    raw_df = _make_df(shapes_flat, ids, group, observer=observer)

    result = run_measurement_error_analysis(raw_df, COLS, 3, "Specimen", observer_col="Observer")
    assert len(result.pairs) == 1
    assert result.pairs[0].kind == "unspecified"


def test_rejects_missing_landmarks():
    rng = np.random.default_rng(6)
    shapes_flat = rng.normal(size=(6, P))
    shapes_flat[0, 0] = np.nan
    ids = [f"s{i}" for i in range(6)]
    group = ["g1", "g1", "g2", "g2", "g3", "g3"]
    raw_df = _make_df(shapes_flat, ids, group)
    with pytest.raises(ValueError, match="missing landmark coordinates"):
        run_measurement_error_analysis(raw_df, COLS, 3, "Specimen")


def test_summary_dataframe_without_observer_col():
    rng = np.random.default_rng(7)
    n_specimens = 8
    base_shapes = rng.normal(size=(n_specimens, N_LM, 3))
    shapes = np.repeat(base_shapes, 2, axis=0)
    ids = [f"s{i}_{r}" for i in range(n_specimens) for r in range(2)]
    group = [f"s{i}" for i in range(n_specimens) for _ in range(2)]
    shapes_flat = shapes.reshape(2 * n_specimens, P)
    raw_df = _make_df(shapes_flat, ids, group)

    result = run_measurement_error_analysis(raw_df, COLS, 3, "Specimen")
    summary = measurement_error_summary_to_dataframe(result)
    assert list(summary["Category"]) == ["Overall"]


def test_summary_dataframe_with_observer_col():
    rng = np.random.default_rng(8)
    n_specimens = 8
    ids, group, observer, rows = [], [], [], []
    for i in range(n_specimens):
        base = rng.normal(size=(N_LM, 3))
        rows.append(base)
        ids.append(f"s{i}_a")
        group.append(f"s{i}")
        observer.append("A")
        rows.append(base + rng.normal(scale=0.1, size=(N_LM, 3)))
        ids.append(f"s{i}_b")
        group.append(f"s{i}")
        observer.append("B")

    shapes_flat = np.stack(rows).reshape(len(rows), P)
    raw_df = _make_df(shapes_flat, ids, group, observer=observer)
    result = run_measurement_error_analysis(raw_df, COLS, 3, "Specimen", observer_col="Observer")
    summary = measurement_error_summary_to_dataframe(result)
    assert list(summary["Category"]) == ["Overall", "Intra-observer", "Inter-observer"]
