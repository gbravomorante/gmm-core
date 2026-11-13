"""The tabular importer, against the small files in tests/data/tabular."""

from pathlib import Path

import numpy as np
import pytest

from gmm_core.data_model import extract_landmark_array
from gmm_core.io.tabular import (
    FORMAT_LONG,
    FORMAT_TPS,
    FORMAT_WIDE,
    TabularOptions,
    import_tabular_files,
    landmark_names_frame,
    sniff_delimiter,
    wide_coordinate_columns,
)
from gmm_core.validation import STATUS_FATAL, STATUS_WARNING

FIX = Path(__file__).resolve().parent / "data" / "tabular"


def _row(result, spec_id):
    return result.raw.loc[result.raw["ID"] == spec_id].iloc[0]


def _coords(result, spec_id):
    row = result.raw.loc[result.raw["ID"] == spec_id]
    return extract_landmark_array(row, result.landmark_cols, result.n_dim)[0]


def _reasons(result, status=STATUS_FATAL):
    return " | ".join(result.validation.loc[result.validation["Status"] == status, "Reason"])


# --- long ------------------------------------------------------------------------------


def test_long_detected_with_names_and_metadata():
    r = import_tabular_files([FIX / "long_with_extras.csv"])
    assert (r.n_landmarks, r.n_dim) == (4, 3)
    assert r.landmark_names == ["nasion", "bregma", "lambda", "inion"]
    assert r.source_format == FORMAT_LONG and r.detected["long_with_extras.csv"]["layout"] == "long"
    assert _row(r, "A1")["Species"] == "Homo"
    assert _row(r, "A2")["Length"] == 12.0
    assert _row(r, "A1")["SourceFile"] == "long_with_extras.csv"


def test_long_matches_landmarks_by_name_not_by_row_order():
    r = import_tabular_files([FIX / "long_with_extras.csv"])
    # A2 lists its landmarks backwards in the file.
    np.testing.assert_array_equal(
        _coords(r, "A2"), [[0.5, 0.5, 0.5], [2, 0, 0], [0, 2, 0], [0, 0, 2]]
    )
    assert "matched by landmark name" in _reasons(r, "Informational")


def test_long_missing_codes_become_nan():
    r = import_tabular_files([FIX / "long_with_extras.csv"])
    c = _coords(r, "A3")
    assert np.isnan(c[0]).all() and np.isnan(c[2]).all()  # blank; -999
    assert np.isnan(c[1, 1]) and c[1, 0] == 1.1  # NA in one axis only
    assert not np.isnan(c[3]).any()


def test_long_specimen_with_no_row_for_a_landmark_is_excluded_not_padded():
    r = import_tabular_files([FIX / "long_with_extras.csv"])
    assert "A4" not in set(r.raw["ID"])
    assert "A4" in _reasons(r) and "excluded rather than padded" in _reasons(r)


# --- wide ------------------------------------------------------------------------------


def test_wide_named_columns_are_ordered_numerically_and_metadata_kept():
    r = import_tabular_files([FIX / "wide_named.csv"])
    assert r.source_format == FORMAT_WIDE and (r.n_landmarks, r.n_dim) == (3, 3)
    assert r.landmark_names == ["1", "2", "10"]  # LM10 third, not second
    np.testing.assert_array_equal(_coords(r, "S1"), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert _row(r, "S2")["Sex"] == "M" and _row(r, "S2")["Age"] == 41.0
    assert np.isnan(_coords(r, "S3")[0]).all()  # 9999
    assert str(r.raw["Age"].dtype) != "object" or isinstance(_row(r, "S1")["Age"], float)


def test_wide_sequential_v_columns_read_as_xyz_triplets():
    r = import_tabular_files([FIX / "wide_sequential.txt"])
    assert r.detected["wide_sequential.txt"]["delimiter"] == repr("\t")
    assert (r.n_landmarks, r.n_dim) == (3, 3)
    np.testing.assert_array_equal(_coords(r, "b")[2], [7, 8, 9.5])
    assert _row(r, "a")["spp"] == "Hsap"


def test_decimal_comma_with_semicolon_delimiter():
    r = import_tabular_files([FIX / "wide_decimal_comma.csv"])
    assert r.n_dim == 2 and r.detected["wide_decimal_comma.csv"]["decimal"] == ","
    assert _coords(r, "p2")[0, 0] == 1.25


def test_wide_duplicate_ids_and_non_numeric_cells_are_reported_not_dropped_silently():
    r = import_tabular_files([FIX / "duplicate_ids.csv"])
    assert list(r.raw["ID"]) == ["ok"]
    assert "'dup' appears more than once" in _reasons(r)
    assert "'bad' has a non-numeric" in _reasons(r)


def test_numeric_metadata_is_not_mistaken_for_coordinates():
    columns, names, dim = wide_coordinate_columns(["ID", "Age", "Length2", "V1", "V2", "V3"], None)
    assert columns == ["V1", "V2", "V3"] and names == ["1"] and dim == 3


def test_sequential_columns_that_could_be_2d_or_3d_must_be_chosen():
    columns = [f"V{i}" for i in range(1, 13)]
    with pytest.raises(ValueError, match="2D or 3D"):
        wide_coordinate_columns(columns, None)
    assert wide_coordinate_columns(columns, 2)[2] == 2


def test_incomplete_axes_are_rejected():
    with pytest.raises(ValueError, match="do not have all"):
        wide_coordinate_columns(["LM1_X", "LM1_Y", "LM1_Z", "LM2_X", "LM2_Y"], None)


def test_explicit_overrides_win():
    options = TabularOptions(
        layout="wide", id_col="spp", n_dim=3, coord_cols=["V1", "V2", "V3", "V4", "V5", "V6"]
    )
    r = import_tabular_files([FIX / "wide_sequential.txt"], options)
    assert set(r.raw["ID"]) == {"Hsap", "Here"} and r.n_landmarks == 2
    assert "V7" in r.raw.columns and "indiv" in r.raw.columns  # the rest is metadata


def test_custom_missing_codes():
    options = TabularOptions(missing_codes=("", "7"))
    assert np.isnan(_coords(import_tabular_files([FIX / "wide_named.csv"], options), "S1")[2, 0])


def test_sniff_prefers_a_consistent_delimiter():
    assert sniff_delimiter(["a,b;c", "1,2;3"]) == ","
    assert sniff_delimiter(["a b c", "1 2 3"]) == " "
    with pytest.raises(ValueError):
        sniff_delimiter(["a b", "1 2 3"])


# --- TPS -------------------------------------------------------------------------------


def test_tps_scale_is_applied_and_recorded():
    r = import_tabular_files([FIX / "specimens.tps"])
    assert r.source_format == FORMAT_TPS and r.n_dim == 2
    np.testing.assert_array_equal(_coords(r, "one"), [[0.5, 1], [1.5, 2], [2.5, 3]])
    assert _row(r, "one")["Image"] == "photo_one.jpg" and _row(r, "one")["Scale"] == 0.5
    assert np.isnan(_coords(r, "two")[1:]).all()


def test_tps_unscaled_specimens_are_warned_about_and_scale_can_be_turned_off():
    r = import_tabular_files([FIX / "specimens.tps"])
    assert "Only 2 of 3 specimens have a SCALE=" in _reasons(r, STATUS_WARNING)
    off = import_tabular_files([FIX / "specimens.tps"], TabularOptions(apply_tps_scale=False))
    assert _coords(off, "one")[0, 0] == 1.0


def test_tps_curve_points_are_not_read_as_landmarks_and_short_block_is_reported():
    r = import_tabular_files([FIX / "specimens.tps"])
    np.testing.assert_array_equal(_coords(r, "three"), [[1, 1], [2, 2], [3, 3]])
    assert "short" not in set(r.raw["ID"]) and "only 2 of 3" in _reasons(r)


def test_tps_3d_with_na():
    r = import_tabular_files([FIX / "three_d.tps"])
    assert (r.n_landmarks, r.n_dim) == (2, 3)
    assert np.isnan(_coords(r, "x2")[1, 0]) and _coords(r, "x2")[1, 1] == 1.0


# --- batches -----------------------------------------------------------------------------


def test_batch_rejects_mixed_dimensions_and_repeated_ids(tmp_path):
    r = import_tabular_files([FIX / "wide_named.csv", FIX / "wide_decimal_comma.csv"])
    assert "2D" in _reasons(r) and set(r.raw["ID"]) == {"S1", "S2", "S3", "S3b"}
    r = import_tabular_files([FIX / "three_d.tps", FIX / "three_d.tps"])
    assert "more than one file" in _reasons(r) and len(r.raw) == 2


def test_an_unreadable_file_is_reported_and_the_rest_still_import(tmp_path):
    junk = tmp_path / "junk.csv"
    junk.write_text("no landmarks here\njust text\n", encoding="utf-8")
    r = import_tabular_files([junk, FIX / "three_d.tps"])
    assert len(r.raw) == 2 and "junk.csv" in _reasons(r)


def test_sex_codes_in_imported_metadata_are_validated(tmp_path):
    path = tmp_path / "sex.csv"
    path.write_text("ID,Sex,LM1_X,LM1_Y\na,Q,1,2\nb,F,3,4\n", encoding="utf-8")
    r = import_tabular_files([path])
    assert "Q" in _reasons(r, "Informational")


def test_landmark_names_frame():
    frame = landmark_names_frame(["nasion", "bregma"])
    assert frame.to_dict("records")[1] == {"Landmark": "LM1", "Source_name": "bregma"}
