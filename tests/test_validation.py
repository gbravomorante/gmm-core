import numpy as np
import pandas as pd

from gmm_core.validation import (
    STATUS_FATAL,
    STATUS_INFO,
    STATUS_WARNING,
    check_duplicate_ids,
    check_landmark_completeness,
    check_sex_codes,
    flag_landmark_count_mismatch,
)


def test_check_duplicate_ids():
    df = pd.DataFrame({"ID": ["a", "b", "a", "c"]})
    result = check_duplicate_ids(df)
    assert set(result["ID"]) == {"a"}
    assert (result["Status"] == STATUS_WARNING).all()
    assert len(result) == 2  # both occurrences of "a" are flagged, neither dropped


def test_check_landmark_completeness_all_missing_is_fatal():
    df = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "LM0_X": [np.nan, 1.0],
            "LM0_Y": [np.nan, 2.0],
            "LM0_Z": [np.nan, 3.0],
        }
    )
    result = check_landmark_completeness(df, ["LM0_X", "LM0_Y", "LM0_Z"])
    row = result.loc[result["ID"] == "a"].iloc[0]
    assert row["Status"] == STATUS_FATAL
    assert "b" not in set(result["ID"])


def test_check_landmark_completeness_partial_is_warning():
    df = pd.DataFrame({"ID": ["a"], "LM0_X": [np.nan], "LM0_Y": [2.0], "LM0_Z": [3.0]})
    result = check_landmark_completeness(df, ["LM0_X", "LM0_Y", "LM0_Z"])
    assert result.iloc[0]["Status"] == STATUS_WARNING
    assert "LM0_X" in result.iloc[0]["Missing_landmarks"]


def test_check_sex_codes_flags_unrecognized_but_does_not_alter_data():
    df = pd.DataFrame({"ID": ["a", "b"], "Sex": ["M", "juvenile"]})
    result = check_sex_codes(df)
    assert len(result) == 1
    assert result.iloc[0]["ID"] == "b"
    assert result.iloc[0]["Status"] == STATUS_INFO
    assert df.loc[1, "Sex"] == "juvenile"  # original value untouched


def test_flag_landmark_count_mismatch():
    counts = {"file_a": 20, "file_b": 20, "file_c": 18}
    expected, mismatches = flag_landmark_count_mismatch(counts)
    assert expected == 20
    assert list(mismatches["ID"]) == ["file_c"]
    assert (mismatches["Status"] == STATUS_FATAL).all()


def test_flag_landmark_count_mismatch_no_mismatch():
    counts = {"file_a": 10, "file_b": 10}
    expected, mismatches = flag_landmark_count_mismatch(counts)
    assert expected == 10
    assert mismatches.empty
