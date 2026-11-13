"""Excel importer.

Converts one worksheet of an ``.xlsx`` workbook into the canonical table
(see :mod:`gmm_core.data_model`). The sheet is expected to use the
``LM<n>_X/Y/Z`` column convention already; the Amira, Slicer and Morphologika
importers build that convention from unlabelled coordinate lists instead.

Also detects existing GPA sheets. A sheet named ``GPA`` or ``GPA_noscale`` is
never trusted because of its name: :func:`validate_existing_gpa_sheet` checks
its structure, row count, centring and scale before it can be reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gmm_core.data_model import (
    ID_COL,
    SOURCE_FILE_COL,
    SOURCE_FORMAT_COL,
    canonicalise_landmark_columns,
    detect_dimensionality,
    extract_landmark_array,
    find_landmark_structure_issues,
    landmark_count,
    natural_sort_landmark_columns,
)
from gmm_core.gpa import center_shape, centroid_size
from gmm_core.validation import (
    STATUS_INFO,
    VALIDATION_COLUMNS,
    combine_validation_tables,
    fill_missing_ids,
    validate_dataset,
)

SOURCE_FORMAT_NAME = "Excel"
GPA_SHEET_NAMES = {"gpa": "GPA", "gpa_noscale": "GPA_noscale"}


@dataclass
class ExcelImportResult:
    raw: pd.DataFrame
    landmark_cols: list[str]
    n_landmarks: int
    n_dim: int
    validation: pd.DataFrame
    sheet_name: str
    source_file: str


def list_sheet_names(path: str | Path) -> list[str]:
    with pd.ExcelFile(Path(path), engine="openpyxl") as workbook:
        return list(workbook.sheet_names)


def _tidy_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Skip completely empty rows and give blank IDs a flagged placeholder.

    Excel often carries formatted-but-empty rows below the data; they hold no
    specimen. A blank ID in a real row is kept, labelled with its spreadsheet
    row, and flagged Fatal by validation - never left as NaN, which is not text
    and breaks anything that sorts or joins by ID. Returns (frame, n_empty).
    """
    empty = df.isna().all(axis=1)
    df = df.loc[~empty]
    df = fill_missing_ids(df)
    df = df.reset_index(drop=True)
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    return df, int(empty.sum())


def import_excel_sheet(path: str | Path, sheet_name: str) -> ExcelImportResult:
    path = Path(path)

    header_only = pd.read_excel(path, sheet_name=sheet_name, nrows=0, engine="openpyxl")
    if ID_COL not in header_only.columns:
        raise ValueError(
            f"Sheet '{sheet_name}' has no '{ID_COL}' column. Found columns: "
            f"{list(header_only.columns)}. The canonical schema requires an exact 'ID' column."
        )

    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl", dtype={ID_COL: str})

    # Real datasets write `LM0_x` as often as `LM0_X`. Rename to the canonical
    # spelling immediately, so every later step - validation, GPA, export - sees
    # exactly one form. Otherwise a whole file can be rejected for a reason the
    # error message does not reveal.
    df = canonicalise_landmark_columns(df)

    structure_issues = find_landmark_structure_issues(df.columns)
    if structure_issues:
        raise ValueError(
            f"Sheet '{sheet_name}' has an inconsistent landmark column structure:\n- "
            + "\n- ".join(structure_issues)
        )

    landmark_cols = natural_sort_landmark_columns(df.columns)
    n_dim = detect_dimensionality(landmark_cols)
    n_landmarks = landmark_count(landmark_cols)

    df, n_empty_rows = _tidy_rows(df)
    df[SOURCE_FORMAT_COL] = SOURCE_FORMAT_NAME
    df[SOURCE_FILE_COL] = path.name

    # Column order: ID, metadata (as found), landmarks (canonical order), source tags.
    meta_cols = [
        c
        for c in df.columns
        if c not in set(landmark_cols) | {ID_COL, SOURCE_FORMAT_COL, SOURCE_FILE_COL}
    ]
    ordered_cols = [ID_COL] + meta_cols + landmark_cols + [SOURCE_FORMAT_COL, SOURCE_FILE_COL]
    df = df.loc[:, ordered_cols]

    validation = validate_dataset(df, landmark_cols)
    if n_empty_rows:
        note = pd.DataFrame(
            [
                {
                    "Row": "",
                    "ID": "",
                    "Status": STATUS_INFO,
                    "Reason": f"{n_empty_rows} completely empty row(s) in the sheet were skipped.",
                    "Missing_landmarks": "",
                    "Suggested_action": "None needed - an empty row holds no specimen.",
                }
            ],
            columns=VALIDATION_COLUMNS,
        )
        validation = combine_validation_tables([validation, note])

    return ExcelImportResult(
        raw=df,
        landmark_cols=landmark_cols,
        n_landmarks=n_landmarks,
        n_dim=n_dim,
        validation=validation,
        sheet_name=sheet_name,
        source_file=path.name,
    )


def find_existing_gpa_sheets(all_sheet_names: list[str]) -> dict[str, str]:
    """Case-insensitive match of workbook sheet names to the canonical
    "GPA" / "GPA_noscale" names. Returns {canonical_name: actual_sheet_name}.
    """
    lower_map = {name.lower(): name for name in all_sheet_names}
    return {
        canonical: lower_map[key] for key, canonical in GPA_SHEET_NAMES.items() if key in lower_map
    }


def validate_existing_gpa_sheet(
    candidate_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    landmark_cols: list[str],
    n_dim: int,
    expect_unit_scale: bool,
    centroid_tolerance: float = 1e-3,
    scale_tolerance: float = 1e-2,
) -> list[str]:
    """Check whether `candidate_df` (an existing GPA/GPA_noscale sheet) looks
    like genuine Procrustes-aligned output for this RAW sheet - never trust a
    sheet just because it is named GPA.
    """
    issues: list[str] = []
    # Tidied exactly as RAW was on import, or blank IDs / empty rows would
    # make an otherwise identical sheet look misaligned.
    if ID_COL in candidate_df.columns:
        candidate_df, _ = _tidy_rows(candidate_df)

    candidate_landmark_cols = natural_sort_landmark_columns(candidate_df.columns)
    if candidate_landmark_cols != landmark_cols:
        issues.append(
            f"Landmark columns don't match RAW: expected {len(landmark_cols)} "
            f"({landmark_cols[0]}...{landmark_cols[-1]}), found {len(candidate_landmark_cols)}."
        )
        return issues  # structural mismatch - nothing further is meaningful to check

    if len(candidate_df) != len(raw_df):
        issues.append(f"Row count ({len(candidate_df)}) does not match RAW ({len(raw_df)}).")

    if (
        ID_COL in candidate_df.columns
        and ID_COL in raw_df.columns
        and len(candidate_df) == len(raw_df)
    ):
        if list(candidate_df[ID_COL].astype(str)) != list(raw_df[ID_COL].astype(str)):
            issues.append("ID column does not match RAW's ID order/values row-for-row.")

    if issues:
        return issues  # don't bother checking geometry on a structurally-broken sheet

    shapes = extract_landmark_array(candidate_df, landmark_cols, n_dim)
    valid_mask = ~np.isnan(shapes).any(axis=(1, 2))
    if not valid_mask.any():
        issues.append("No complete (non-missing) specimens to check in this sheet.")
        return issues

    centroids = shapes[valid_mask].mean(axis=1)
    max_centroid_offset = float(np.abs(centroids).max())
    if max_centroid_offset > centroid_tolerance:
        issues.append(
            f"Specimens are not centered on the origin (max centroid offset "
            f"{max_centroid_offset:.4g}, expected < {centroid_tolerance:g}) - this sheet may not "
            "actually be Procrustes-aligned coordinates."
        )

    if expect_unit_scale:
        sizes = np.array(
            [centroid_size(shapes[i]) for i in range(shapes.shape[0]) if valid_mask[i]]
        )
        max_size_offset = float(np.abs(sizes - 1.0).max())
        if max_size_offset > scale_tolerance:
            issues.append(
                f"Centroid sizes are not close to 1.0 (max deviation {max_size_offset:.4g}, "
                f"expected < {scale_tolerance:g}) - this doesn't look like scaled/full GPA output."
            )

    return issues


def load_and_augment_gpa_sheet(
    path: str | Path, sheet_name: str, landmark_cols: list[str], n_dim: int
) -> pd.DataFrame:
    """Read an existing GPA/GPA_noscale sheet, canonicalize it the same way
    RAW is canonicalized, and fill in CentroidSize / ProcrustesDistance_to_
    consensus if the sheet doesn't already have them (computed directly from
    its own coordinates - no re-alignment, since reuse only makes sense for a
    sheet already validated as properly aligned).
    """
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl", dtype={ID_COL: str})
    df, _ = _tidy_rows(df)

    shapes = extract_landmark_array(df, landmark_cols, n_dim)
    valid_mask = ~np.isnan(shapes).any(axis=(1, 2))

    if "CentroidSize" not in df.columns:
        df["CentroidSize"] = [
            centroid_size(shapes[i]) if valid_mask[i] else float("nan")
            for i in range(shapes.shape[0])
        ]

    if "ProcrustesDistance_to_consensus" not in df.columns:
        normalized = np.full_like(shapes, np.nan)
        for i in range(shapes.shape[0]):
            if valid_mask[i]:
                size = centroid_size(shapes[i])
                normalized[i] = center_shape(shapes[i]) / size if size > 0 else np.nan
        if valid_mask.any():
            consensus = np.nanmean(normalized[valid_mask], axis=0)
            distances = [
                float(np.linalg.norm(normalized[i] - consensus)) if valid_mask[i] else float("nan")
                for i in range(shapes.shape[0])
            ]
        else:
            distances = [float("nan")] * shapes.shape[0]
        df["ProcrustesDistance_to_consensus"] = distances

    return df
