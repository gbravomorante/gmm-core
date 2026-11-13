import numpy as np
import pandas as pd
import pytest

from gmm_core.io.excel import import_excel_sheet, list_sheet_names
from gmm_core.validation import STATUS_WARNING


def _make_workbook(tmp_path, df, sheet_name="RAW", filename="test.xlsx"):
    path = tmp_path / filename
    df.to_excel(path, sheet_name=sheet_name, index=False)
    return path


def _valid_dataframe(n=6, n_landmarks=4):
    rng = np.random.default_rng(0)
    data = {
        "ID": [f"spec_{i:03d}" for i in range(n)],
        "Sex": (["M", "F", "X", "M", "F", "M"] * ((n // 6) + 1))[:n],
        "Species": ["Homo sapiens"] * n,
    }
    for lm in range(n_landmarks):
        for axis in "XYZ":
            data[f"LM{lm}_{axis}"] = rng.normal(size=n)
    return pd.DataFrame(data)


def test_list_sheet_names(tmp_path):
    df = _valid_dataframe()
    path = _make_workbook(tmp_path, df, sheet_name="RAW")
    assert list_sheet_names(path) == ["RAW"]


def test_import_valid_sheet(tmp_path):
    df = _valid_dataframe()
    path = _make_workbook(tmp_path, df)
    result = import_excel_sheet(path, "RAW")
    assert result.n_landmarks == 4
    assert result.n_dim == 3
    assert list(result.raw["ID"]) == list(df["ID"])
    assert "SourceFormat" in result.raw.columns
    assert (result.raw["SourceFormat"] == "Excel").all()
    assert (result.raw["SourceFile"] == path.name).all()


def test_import_preserves_leading_zero_ids(tmp_path):
    df = _valid_dataframe()
    df["ID"] = [f"{i:04d}" for i in range(len(df))]
    path = _make_workbook(tmp_path, df)
    result = import_excel_sheet(path, "RAW")
    assert list(result.raw["ID"]) == list(df["ID"])
    assert result.raw["ID"].iloc[0] == "0000"


def test_missing_id_column_raises(tmp_path):
    df = _valid_dataframe().drop(columns=["ID"])
    path = _make_workbook(tmp_path, df)
    with pytest.raises(ValueError, match="ID"):
        import_excel_sheet(path, "RAW")


def test_landmark_gap_raises(tmp_path):
    df = _valid_dataframe(n_landmarks=3)
    df = df.drop(columns=["LM1_X", "LM1_Y", "LM1_Z"])  # LM0 and LM2 present, LM1 missing
    path = _make_workbook(tmp_path, df)
    with pytest.raises(ValueError, match="landmark column structure"):
        import_excel_sheet(path, "RAW")


def test_duplicate_ids_flagged_not_dropped(tmp_path):
    df = _valid_dataframe()
    df.loc[1, "ID"] = df.loc[0, "ID"]
    path = _make_workbook(tmp_path, df)
    result = import_excel_sheet(path, "RAW")
    assert len(result.raw) == len(df)
    assert (result.validation["Status"] == STATUS_WARNING).any()
    assert "Duplicate ID" in " ".join(result.validation["Reason"])


def test_missing_landmark_values_flagged_not_dropped(tmp_path):
    df = _valid_dataframe()
    df.loc[2, "LM1_X"] = np.nan
    path = _make_workbook(tmp_path, df)
    result = import_excel_sheet(path, "RAW")
    assert len(result.raw) == len(df)
    flagged_ids = set(result.validation["ID"])
    assert df.loc[2, "ID"] in flagged_ids


def test_metadata_columns_preserved_as_found(tmp_path):
    df = _valid_dataframe()
    df["CustomNote"] = "kept"
    path = _make_workbook(tmp_path, df)
    result = import_excel_sheet(path, "RAW")
    assert "CustomNote" in result.raw.columns
    assert (result.raw["CustomNote"] == "kept").all()


def test_landmark_columns_are_naturally_sorted(tmp_path):
    df = _valid_dataframe(n_landmarks=11)  # includes LM10, exercises numeric-not-alphabetic sort
    path = _make_workbook(tmp_path, df)
    result = import_excel_sheet(path, "RAW")
    assert result.landmark_cols[-3:] == ["LM10_X", "LM10_Y", "LM10_Z"]


def test_blank_id_gets_a_flagged_placeholder_not_nan(tmp_path):
    """A blank ID must not import as an unflagged NaN, which breaks sorting and joining by ID."""
    from gmm_core.validation import MISSING_ID_PREFIX, STATUS_FATAL

    df = _valid_dataframe()
    df.loc[3, "ID"] = None
    df.loc[4, "ID"] = "   "
    result = import_excel_sheet(_make_workbook(tmp_path, df), "RAW")

    ids = list(result.raw["ID"])
    assert all(isinstance(v, str) for v in ids)
    assert sorted(ids)  # sortable: this is what crashed
    assert ids[3] == f"{MISSING_ID_PREFIX}, spreadsheet row 5)"
    assert ids[4] == f"{MISSING_ID_PREFIX}, spreadsheet row 6)"
    fatal = result.validation[result.validation["Status"] == STATUS_FATAL]
    assert set(fatal["ID"]) == {ids[3], ids[4]}
    assert not result.validation["Reason"].str.contains("Duplicate").any()


def test_completely_empty_rows_are_skipped_and_reported(tmp_path):
    from gmm_core.validation import MISSING_ID_PREFIX, STATUS_INFO

    df = _valid_dataframe()
    blank = pd.DataFrame([[np.nan] * df.shape[1]] * 2, columns=df.columns)
    df = pd.concat([df.iloc[:3], blank, df.iloc[3:]], ignore_index=True)
    df.loc[6, "ID"] = None  # a real row with no ID, below the gap
    result = import_excel_sheet(_make_workbook(tmp_path, df), "RAW")

    assert len(result.raw) == 6
    assert result.raw["ID"].iloc[4] == f"{MISSING_ID_PREFIX}, spreadsheet row 8)"
    info = result.validation[result.validation["Status"] == STATUS_INFO]
    assert info["Reason"].str.contains("2 completely empty row").any()


def test_gpa_sheet_reuse_still_matches_when_ids_are_blank(tmp_path):
    """RAW and a reused GPA sheet are tidied the same way, so their IDs still line up."""
    from gmm_core.io.excel import load_and_augment_gpa_sheet

    df = _valid_dataframe()
    df.loc[2, "ID"] = None
    path = tmp_path / "both.xlsx"
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="RAW", index=False)
        df.to_excel(writer, sheet_name="GPA", index=False)
    raw = import_excel_sheet(path, "RAW")
    gpa = load_and_augment_gpa_sheet(path, "GPA", raw.landmark_cols, raw.n_dim)
    assert list(gpa["ID"]) == list(raw.raw["ID"])
