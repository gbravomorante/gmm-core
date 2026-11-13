import numpy as np

from gmm_core.gpa import generalized_procrustes
from gmm_core.qc import (
    centroid_size_outliers,
    procrustes_distance_outliers,
    single_landmark_outliers,
)


def _sample_shapes(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(6, 3))
    return np.stack([base + rng.normal(scale=0.02, size=base.shape) for _ in range(n)])


def test_centroid_size_outlier_detected():
    ids = [f"s{i}" for i in range(10)]
    # Small realistic spread among 9 specimens (MAD stays nonzero) plus one clear outlier.
    sizes = np.array([1.0, 1.02, 0.98, 1.01, 0.99, 1.03, 0.97, 1.0, 1.02, 10.0])
    result = centroid_size_outliers(ids, sizes)
    assert "s9" in set(result["ID"])
    assert "s0" not in set(result["ID"])


def test_no_false_positives_on_uniform_sample():
    shapes = _sample_shapes(n=10, seed=1)
    gpa_result = generalized_procrustes(shapes, scale=True)
    ids = [f"s{i}" for i in range(10)]
    result = centroid_size_outliers(ids, gpa_result.centroid_sizes)
    assert result.empty


def test_procrustes_distance_outlier_detected():
    shapes = _sample_shapes(n=10)
    gpa_result = generalized_procrustes(shapes, scale=True)
    distances = gpa_result.procrustes_distances.copy()
    distances[-1] = distances[:-1].max() * 20 + 1.0
    ids = [f"s{i}" for i in range(10)]
    result = procrustes_distance_outliers(ids, distances)
    assert "s9" in set(result["ID"])


def test_single_landmark_outlier_detected():
    shapes = _sample_shapes(n=10)
    gpa_result = generalized_procrustes(shapes, scale=True)
    aligned = gpa_result.aligned.copy()
    aligned[-1, 0, :] += 5.0  # displace one landmark on one specimen far from consensus
    ids = [f"s{i}" for i in range(10)]
    result = single_landmark_outliers(ids, aligned, gpa_result.consensus)
    assert "s9" in set(result["ID"])
    assert any("LM0" in reason for reason in result["Reason"])


def test_qc_checks_never_remove_rows_only_flag():
    # Flag functions must never mutate their inputs or "drop" a specimen -
    # they only ever return additional rows to review.
    ids = [f"s{i}" for i in range(10)]
    sizes = np.array([1.0] * 9 + [10.0])
    original_ids = list(ids)
    centroid_size_outliers(ids, sizes)
    assert ids == original_ids


# --- mixed units masquerading as biological size ----------------------------


def test_a_continuous_biological_size_range_is_never_flagged():
    """A comparative sample really does span shrew to elephant. Range alone
    must not trigger this - only an empty gap does."""
    from gmm_core.qc import scale_anomaly

    sizes = np.geomspace(70, 3600, 200)  # 50x, continuously filled
    ids = [f"S{i}" for i in range(len(sizes))]
    assert scale_anomaly(ids, sizes) == {}


def test_two_bands_with_an_empty_gap_are_flagged_as_probable_unit_mismatch():
    from gmm_core.qc import scale_anomaly, scale_anomaly_message

    sizes = np.concatenate([np.geomspace(70, 3600, 190), np.geomspace(7.0e7, 1.3e9, 18)])
    ids = [f"S{i}" for i in range(len(sizes))]
    found = scale_anomaly(ids, sizes)
    assert found
    assert found["n_minority"] == 18
    assert found["n_majority"] == 190
    assert found["minority_is_larger"] is True
    message = scale_anomaly_message(found)
    assert "UNIT MISMATCH" in message
    assert "cannot leave a gap" in message


def test_the_flagged_specimens_are_named_so_they_can_be_fixed():
    from gmm_core.qc import scale_anomaly

    sizes = np.array([100.0, 110.0, 120.0, 130.0, 1.2e8, 1.3e8])
    ids = ["a", "b", "c", "d", "wrong_units_1", "wrong_units_2"]
    found = scale_anomaly(ids, sizes)
    assert set(found["minority_ids"]) == {"wrong_units_1", "wrong_units_2"}


def test_the_smaller_band_can_be_the_odd_one_out():
    """The minority is whichever band has fewer specimens, not whichever is
    smaller - the correct unit is not necessarily the commoner one, but the
    unusual one is what needs checking."""
    from gmm_core.qc import scale_anomaly

    sizes = np.concatenate([np.array([0.001, 0.0012]), np.geomspace(100, 400, 30)])
    ids = [f"S{i}" for i in range(len(sizes))]
    found = scale_anomaly(ids, sizes)
    assert found and found["n_minority"] == 2
    assert found["minority_is_larger"] is False


def test_a_recognised_conversion_factor_is_named():
    from gmm_core.qc import scale_anomaly, scale_anomaly_message

    band = np.geomspace(100, 400, 20)
    sizes = np.concatenate([band, band[:4] * 1e6])  # a genuine mm-to-nm mixture
    ids = [f"S{i}" for i in range(len(sizes))]
    found = scale_anomaly(ids, sizes)
    assert 0.3 <= found["factor"] / 1e6 <= 3.0
    assert "conversion" in scale_anomaly_message(found)


def test_the_offset_is_measured_between_band_MEDIANS_not_the_gap():
    """Each band spans a real size range of its own, so the empty gap
    understates how far apart the two scales are."""
    from gmm_core.qc import scale_anomaly

    band = np.geomspace(100, 400, 20)
    found = scale_anomaly([f"S{i}" for i in range(24)], np.concatenate([band, band[:4] * 1e6]))
    assert found["factor"] > 10 ** found["gap_decades"]


def test_too_few_specimens_to_judge_is_silent():
    from gmm_core.qc import scale_anomaly

    assert scale_anomaly(["a", "b"], np.array([1.0, 1e9])) == {}


def test_non_positive_and_missing_sizes_do_not_crash_the_check():
    from gmm_core.qc import scale_anomaly

    sizes = np.array([np.nan, 0.0, -3.0, 100.0, 110.0, 120.0, 130.0])
    assert isinstance(scale_anomaly([f"S{i}" for i in range(7)], sizes), dict)


def test_the_finding_becomes_validation_rows_one_per_odd_specimen():
    from gmm_core.qc import scale_anomaly_rows
    from gmm_core.validation import STATUS_WARNING, VALIDATION_COLUMNS

    sizes = np.array([100.0, 110.0, 120.0, 130.0, 1.2e8, 1.3e8])
    ids = ["a", "b", "c", "d", "odd1", "odd2"]
    rows = scale_anomaly_rows(ids, sizes)
    assert list(rows.columns) == list(VALIDATION_COLUMNS)
    assert len(rows) == 2
    assert set(rows["ID"]) == {"odd1", "odd2"}
    assert set(rows["Status"]) == {STATUS_WARNING}
    # Flags, never removes - the other band may be the wrong one.
    assert "Do not exclude" in rows.iloc[0]["Suggested_action"]


def test_a_clean_sample_produces_no_scale_rows():
    from gmm_core.qc import scale_anomaly_rows

    sizes = np.geomspace(70, 3600, 100)
    assert scale_anomaly_rows([f"S{i}" for i in range(100)], sizes).empty


# --- the per-landmark covariance check for samples of several taxa ----------------------


def _two_taxa(n_per=40, seed=0):
    """Two taxa with clearly different mean shapes, each with its own spread."""
    rng = np.random.default_rng(seed)
    base_a = rng.normal(size=(8, 3))
    base_b = base_a + rng.normal(scale=0.6, size=(8, 3))
    shapes = [base_a + rng.normal(scale=0.03, size=(8, 3)) for _ in range(n_per)]
    shapes += [base_b + rng.normal(scale=0.03, size=(8, 3)) for _ in range(n_per)]
    return np.stack(shapes), ["A"] * n_per + ["B"] * n_per


def test_covariance_check_finds_a_swapped_landmark_within_its_taxon():
    from gmm_core.qc import comparative_qc_rows

    shapes, groups = _two_taxa()
    shapes[5, [2, 3]] = shapes[5, [3, 2]]  # swap two landmarks on one specimen
    gpa = generalized_procrustes(shapes, scale=True)
    ids = [f"s{i}" for i in range(len(shapes))]
    rows = comparative_qc_rows(ids, gpa.centroid_sizes, gpa.aligned, groups, "Species")
    flagged = rows[rows["Reason"].str.startswith("Landmark")]
    assert "s5" in set(flagged["ID"])
    assert set(flagged["ID"]) <= {"s5"}, flagged[["ID", "Reason"]]
    assert flagged["Reason"].str.contains("for its Species 'A'").all()


def test_covariance_check_is_quiet_on_a_clean_two_taxon_sample():
    from gmm_core.qc import comparative_qc_rows

    shapes, groups = _two_taxa(seed=3)
    gpa = generalized_procrustes(shapes, scale=True)
    ids = [f"s{i}" for i in range(len(shapes))]
    rows = comparative_qc_rows(ids, gpa.centroid_sizes, gpa.aligned, groups, "Species")
    landmark_rows = rows[rows["Reason"].str.startswith("Landmark")]
    assert len(landmark_rows) <= 1  # family-wise alpha 0.1 over 640 tests


def test_small_groups_are_compared_with_the_whole_sample_and_say_so():
    from gmm_core.qc import comparative_qc_rows

    shapes, groups = _two_taxa()
    groups[0] = "Fossil"  # a group of one
    shapes[0, 2] = shapes[0, 2] + 3.0  # and unusual (a shift, not a translation)
    gpa = generalized_procrustes(shapes, scale=True)
    rows = comparative_qc_rows(
        [f"s{i}" for i in range(len(shapes))], gpa.centroid_sizes, gpa.aligned, groups, "Species"
    )
    fossil = rows[rows["ID"] == "s0"]
    assert not fossil.empty
    assert fossil["Reason"].str.contains("fewer than 8 specimens").all()


def test_rerunning_qc_replaces_rather_than_duplicates():
    import pandas as pd

    from gmm_core.qc import run_all_qc_checks, without_qc_rows
    from gmm_core.validation import combine_validation_tables

    shapes, _ = _two_taxa()
    shapes[5, [2, 3]] = shapes[5, [3, 2]]
    gpa = generalized_procrustes(shapes, scale=True)
    ids = [f"s{i}" for i in range(len(shapes))]
    imported = pd.DataFrame(
        [
            {
                "Row": 1,
                "ID": "s1",
                "Status": "Warning",
                "Reason": "2 of 24 landmark coordinate values are missing.",
                "Missing_landmarks": "",
                "Suggested_action": "",
            }
        ]
    )
    qc = run_all_qc_checks(
        ids, gpa.centroid_sizes, gpa.procrustes_distances, gpa.aligned, gpa.consensus
    )
    once = combine_validation_tables([without_qc_rows(imported), qc])
    twice = combine_validation_tables([without_qc_rows(once), qc])
    assert len(twice) == len(once)
    assert "2 of 24 landmark coordinate values are missing." in set(twice["Reason"])


def test_unknown_landmark_method_is_refused():
    import pytest

    from gmm_core.qc import run_all_qc_checks

    shapes, _ = _two_taxa()
    gpa = generalized_procrustes(shapes, scale=True)
    with pytest.raises(ValueError, match="Unknown landmark_method"):
        run_all_qc_checks(
            ["a"] * len(shapes),
            gpa.centroid_sizes,
            gpa.procrustes_distances,
            gpa.aligned,
            gpa.consensus,
            landmark_method="nope",
        )
