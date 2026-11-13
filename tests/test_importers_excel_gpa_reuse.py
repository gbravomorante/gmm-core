import numpy as np
import pandas as pd

from gmm_core.data_model import ID_COL, landmark_columns, landmarks_to_row_dict
from gmm_core.gpa import generalized_procrustes
from gmm_core.io.excel import (
    find_existing_gpa_sheets,
    load_and_augment_gpa_sheet,
    validate_existing_gpa_sheet,
)


def _raw_and_gpa_dfs(n=8, n_lm=5, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_lm, 3))
    raw_shapes = np.stack([base + rng.normal(scale=0.1, size=base.shape) for _ in range(n)])
    ids = [f"s{i}" for i in range(n)]
    cols = landmark_columns(n_lm, n_dim=3)

    raw_rows = []
    for i, spec_id in enumerate(ids):
        row = {ID_COL: spec_id}
        row.update(landmarks_to_row_dict(raw_shapes[i], 3))
        raw_rows.append(row)
    raw_df = pd.DataFrame(raw_rows)

    gpa_result = generalized_procrustes(raw_shapes, scale=True)
    gpa_rows = []
    for i, spec_id in enumerate(ids):
        row = {ID_COL: spec_id}
        row.update(landmarks_to_row_dict(gpa_result.aligned[i], 3))
        gpa_rows.append(row)
    gpa_df = pd.DataFrame(gpa_rows)

    return raw_df, gpa_df, cols


def test_find_existing_gpa_sheets_case_insensitive():
    sheets = ["RAW", "gpa", "GPA_NOSCALE", "Notes"]
    found = find_existing_gpa_sheets(sheets)
    assert found == {"GPA": "gpa", "GPA_noscale": "GPA_NOSCALE"}


def test_find_existing_gpa_sheets_none_present():
    assert find_existing_gpa_sheets(["RAW", "Notes"]) == {}


def test_validate_existing_gpa_sheet_accepts_genuine_gpa_output():
    raw_df, gpa_df, cols = _raw_and_gpa_dfs()
    issues = validate_existing_gpa_sheet(gpa_df, raw_df, cols, n_dim=3, expect_unit_scale=True)
    assert issues == []


def test_validate_existing_gpa_sheet_rejects_uncentered_sheet():
    raw_df, gpa_df, cols = _raw_and_gpa_dfs()
    shifted = gpa_df.copy()
    for lm in range(5):
        shifted[f"LM{lm}_X"] += 5.0  # break centering
    issues = validate_existing_gpa_sheet(shifted, raw_df, cols, n_dim=3, expect_unit_scale=True)
    assert any("centered" in issue.lower() for issue in issues)


def test_validate_existing_gpa_sheet_rejects_non_unit_scale_when_expected():
    raw_df, gpa_df, cols = _raw_and_gpa_dfs()
    scaled = gpa_df.copy()
    for lm in range(5):
        for axis in "XYZ":
            scaled[f"LM{lm}_{axis}"] *= 3.0  # break unit centroid size
    issues = validate_existing_gpa_sheet(scaled, raw_df, cols, n_dim=3, expect_unit_scale=True)
    assert any("1.0" in issue for issue in issues)


def test_validate_existing_gpa_sheet_noscale_does_not_check_unit_scale():
    # GPA_noscale retains original size on purpose - scale check must be skipped.
    raw_df, gpa_df, cols = _raw_and_gpa_dfs()
    scaled = gpa_df.copy()
    for lm in range(5):
        for axis in "XYZ":
            scaled[f"LM{lm}_{axis}"] *= 3.0
    issues = validate_existing_gpa_sheet(scaled, raw_df, cols, n_dim=3, expect_unit_scale=False)
    assert not any("1.0" in issue for issue in issues)


def test_validate_existing_gpa_sheet_rejects_landmark_structure_mismatch():
    raw_df, gpa_df, cols = _raw_and_gpa_dfs()
    truncated = gpa_df.drop(columns=["LM4_X", "LM4_Y", "LM4_Z"])
    issues = validate_existing_gpa_sheet(truncated, raw_df, cols, n_dim=3, expect_unit_scale=True)
    assert any("landmark columns" in issue.lower() for issue in issues)


def test_validate_existing_gpa_sheet_rejects_row_count_mismatch():
    raw_df, gpa_df, cols = _raw_and_gpa_dfs()
    fewer_rows = gpa_df.iloc[:-2].reset_index(drop=True)
    issues = validate_existing_gpa_sheet(fewer_rows, raw_df, cols, n_dim=3, expect_unit_scale=True)
    assert any("row count" in issue.lower() for issue in issues)


def test_validate_existing_gpa_sheet_rejects_id_mismatch():
    raw_df, gpa_df, cols = _raw_and_gpa_dfs()
    reordered = gpa_df.iloc[::-1].reset_index(drop=True)
    issues = validate_existing_gpa_sheet(reordered, raw_df, cols, n_dim=3, expect_unit_scale=True)
    assert any("id" in issue.lower() for issue in issues)


def test_load_and_augment_gpa_sheet_adds_missing_metrics(tmp_path):
    raw_df, gpa_df, cols = _raw_and_gpa_dfs()
    path = tmp_path / "wb.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="RAW", index=False)
        gpa_df.to_excel(writer, sheet_name="GPA", index=False)

    augmented = load_and_augment_gpa_sheet(path, "GPA", cols, n_dim=3)
    assert "CentroidSize" in augmented.columns
    assert "ProcrustesDistance_to_consensus" in augmented.columns
    assert np.allclose(augmented["CentroidSize"], 1.0, atol=1e-6)


def test_load_and_augment_gpa_sheet_preserves_existing_centroid_size(tmp_path):
    raw_df, gpa_df, cols = _raw_and_gpa_dfs()
    gpa_df = gpa_df.copy()
    gpa_df["CentroidSize"] = 42.0  # deliberately fake value to prove it's preserved, not recomputed
    path = tmp_path / "wb.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="RAW", index=False)
        gpa_df.to_excel(writer, sheet_name="GPA", index=False)

    augmented = load_and_augment_gpa_sheet(path, "GPA", cols, n_dim=3)
    assert (augmented["CentroidSize"] == 42.0).all()
