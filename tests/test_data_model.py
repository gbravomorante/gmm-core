import numpy as np
import pandas as pd
import pytest

from gmm_core.data_model import (
    AGE_CLASS_COL,
    AGE_COL,
    ID_COL,
    LABEL_COL,
    add_display_label_column,
    add_virtual_metadata_columns,
    classify_age,
    compute_label_column,
    detect_dimensionality,
    find_landmark_structure_issues,
    get_landmark_indices,
    landmark_columns,
    landmark_count,
    natural_sort_landmark_columns,
    normalize_sex,
    numeric_coercible_columns,
    resolve_display_label,
    virtual_metadata_column_names,
)


def test_natural_sort_numeric_not_alphabetical():
    columns = ["LM10_X", "LM10_Y", "LM10_Z", "LM2_X", "LM2_Y", "LM2_Z", "ID"]
    sorted_cols = natural_sort_landmark_columns(columns)
    assert sorted_cols == ["LM2_X", "LM2_Y", "LM2_Z", "LM10_X", "LM10_Y", "LM10_Z"]


def test_detect_dimensionality_3d():
    assert detect_dimensionality(["LM0_X", "LM0_Y", "LM0_Z"]) == 3


def test_detect_dimensionality_2d():
    assert detect_dimensionality(["LM0_X", "LM0_Y", "LM1_X", "LM1_Y"]) == 2


def test_landmark_count_and_indices():
    cols = landmark_columns(5, n_dim=3)
    assert landmark_count(cols) == 5
    assert get_landmark_indices(cols) == [0, 1, 2, 3, 4]


def test_structure_issues_detects_gap():
    cols = ["LM0_X", "LM0_Y", "LM0_Z", "LM2_X", "LM2_Y", "LM2_Z"]  # missing LM1
    issues = find_landmark_structure_issues(cols)
    assert any("gap" in issue.lower() for issue in issues)


def test_structure_issues_detects_nonzero_start():
    cols = ["LM1_X", "LM1_Y", "LM1_Z"]
    issues = find_landmark_structure_issues(cols)
    assert any("start at 0" in issue for issue in issues)


def test_structure_issues_detects_missing_axis():
    cols = ["LM0_X", "LM0_Y", "LM0_Z", "LM1_X", "LM1_Y"]  # LM1 missing Z
    issues = find_landmark_structure_issues(cols)
    assert any("LM1" in issue for issue in issues)


def test_structure_clean_has_no_issues():
    cols = landmark_columns(10, n_dim=3)
    assert find_landmark_structure_issues(cols) == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("M", "M"),
        ("male", "M"),
        ("F", "F"),
        ("Female", "F"),
        ("X", "X"),
        ("", "X"),
        (None, "X"),
        ("unknown", "X"),
    ],
)
def test_normalize_sex_known_codes(raw, expected):
    assert normalize_sex(raw) == expected


def test_normalize_sex_preserves_unknown_codes_verbatim():
    # Non-blank, non-standard codes are kept as-is, not collapsed to X, so
    # they surface in validation instead of silently becoming "unknown".
    assert normalize_sex("JUV") == "JUV"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("subadult", "Subadult"),
        ("Subadult", "Subadult"),
        ("SUBADULT", "Subadult"),
        ("juvenile", "Subadult"),
        ("adult", "Adult"),
        ("Adult", "Adult"),
        (15, "Subadult"),
        (15.9, "Subadult"),
        ("15", "Subadult"),
        (16, "Adult"),
        (16.0, "Adult"),
        (40, "Adult"),
        (0, "Subadult"),
    ],
)
def test_classify_age_known_values(raw, expected):
    assert classify_age(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "  ", np.nan])
def test_classify_age_missing_is_unknown_not_adult(raw):
    # Missing data should never be silently assumed Adult.
    assert classify_age(raw) == "Unknown"


def test_classify_age_unrecognized_text_is_unknown():
    assert classify_age("juvenile-ish maybe?") == "Unknown"


def test_classify_age_custom_threshold():
    assert classify_age(17, threshold=18.0) == "Subadult"
    assert classify_age(18, threshold=18.0) == "Adult"


def test_virtual_metadata_column_names_only_when_age_present():
    assert virtual_metadata_column_names(["ID", "Age", "Sex"]) == [AGE_CLASS_COL]
    assert virtual_metadata_column_names(["ID", "Sex"]) == []
    # Already-materialized AgeClass shouldn't be listed as "virtual" again.
    assert virtual_metadata_column_names(["ID", "Age", AGE_CLASS_COL]) == []


def test_add_virtual_metadata_columns_populates_age_class():
    df = pd.DataFrame({"ID": ["a", "b", "c"], AGE_COL: ["subadult", 40, None]})
    out = add_virtual_metadata_columns(df)
    assert list(out[AGE_CLASS_COL]) == ["Subadult", "Adult", "Unknown"]
    # Original frame is untouched.
    assert AGE_CLASS_COL not in df.columns


def test_add_virtual_metadata_columns_noop_without_age():
    df = pd.DataFrame({"ID": ["a", "b"], "Sex": ["M", "F"]})
    out = add_virtual_metadata_columns(df)
    assert AGE_CLASS_COL not in out.columns


def test_numeric_coercible_columns_mixed_age():
    df = pd.DataFrame({"Age": ["subadult", "40", "22.5", None], "Sex": ["M", "F", "M", "F"]})
    result = numeric_coercible_columns(df, ["Age", "Sex"])
    assert result == ["Age"]


def test_resolve_display_label_falls_back_to_id_when_no_alias():
    assert resolve_display_label("KNM-WT-15000-hipboneR-1007", {}) == "KNM-WT-15000-hipboneR-1007"


def test_resolve_display_label_uses_alias_when_present():
    labels = {"KNM-WT-15000-hipboneR-1007": "WT15000 R"}
    assert resolve_display_label("KNM-WT-15000-hipboneR-1007", labels) == "WT15000 R"
    assert resolve_display_label("some_other_id", labels) == "some_other_id"


def test_compute_label_column_mixed_overrides():
    df = pd.DataFrame({ID_COL: ["a", "b", "c"]})
    labels = {"b": "Specimen B"}
    result = compute_label_column(df, labels)
    assert list(result) == ["a", "Specimen B", "c"]


def test_add_display_label_column_defaults_to_id_and_does_not_mutate_original():
    df = pd.DataFrame({ID_COL: ["a", "b"]})
    out = add_display_label_column(df, {"a": "Alpha"})
    assert list(out[LABEL_COL]) == ["Alpha", "b"]
    assert LABEL_COL not in df.columns  # original untouched


def test_add_display_label_column_with_no_labels_still_adds_column():
    df = pd.DataFrame({ID_COL: ["a", "b"]})
    out = add_display_label_column(df, {})
    assert list(out[LABEL_COL]) == ["a", "b"]


def test_numeric_coercible_columns_all_text_excluded():
    df = pd.DataFrame({"Species": ["a", "b", "c"]})
    assert numeric_coercible_columns(df, ["Species"]) == []


# --- landmark column capitalisation ---------------------------------------
# Real datasets write `LM0_x` as often as `LM0_X`. Rejecting the lower-case
# form would stop whole files from loading, with an error that never mentions
# capitalisation.


def test_lower_case_axis_letters_are_recognised():
    from gmm_core.data_model import is_landmark_column

    assert is_landmark_column("LM0_x")
    assert is_landmark_column("LM10_z")
    assert is_landmark_column("lm2_Y")
    assert not is_landmark_column("LM0 X")
    assert not is_landmark_column("Sex")


def test_canonical_name_is_always_upper_case():
    from gmm_core.data_model import canonical_landmark_name

    assert canonical_landmark_name("LM0_x") == "LM0_X"
    assert canonical_landmark_name("LM10_z") == "LM10_Z"
    assert canonical_landmark_name("LM0_X") == "LM0_X"
    assert canonical_landmark_name("Sex") == "Sex"


def test_canonicalising_renames_only_landmark_columns():
    import pandas as pd

    from gmm_core.data_model import canonicalise_landmark_columns

    df = pd.DataFrame(columns=["ID", "Sex", "LM0_x", "LM0_y", "LM0_z"])
    out = canonicalise_landmark_columns(df)
    assert list(out.columns) == ["ID", "Sex", "LM0_X", "LM0_Y", "LM0_Z"]


def test_mixed_case_columns_still_sort_numerically():
    from gmm_core.data_model import detect_dimensionality, natural_sort_landmark_columns

    cols = ["LM10_x", "LM2_y", "LM0_z", "LM0_x", "Sex"]
    ordered = natural_sort_landmark_columns(cols)
    assert ordered[0] == "LM0_x"  # LM0 before LM2 before LM10, never alphabetical
    assert ordered[-1] == "LM10_x"
    assert detect_dimensionality(cols) == 3
