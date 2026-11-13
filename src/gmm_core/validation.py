"""Structural validation for imported datasets.

Produces one shared validation table that every importer and every later
quality-control step (:mod:`gmm_core.qc`) appends to. Nothing here ever drops
a row; it only flags rows, so every exclusion stays visible.

Status levels:
    Fatal          The row cannot be used in this form (no coordinates at
                   all, or a landmark count that differs from the rest of the
                   batch). It is left out of the combined table but is still
                   reported, never dropped silently.
    Warning        Usable, but worth a look before trusting results
                   (duplicate ID, partially missing coordinates, an extreme
                   QC metric).
    Informational  A non-blocking note (for example an unrecognised but
                   preserved sex code).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from gmm_core.data_model import ID_COL, SEX_COL, normalize_sex

STATUS_FATAL = "Fatal"
STATUS_WARNING = "Warning"
STATUS_INFO = "Informational"

VALIDATION_COLUMNS = ["Row", "ID", "Status", "Reason", "Missing_landmarks", "Suggested_action"]


def empty_validation_table() -> pd.DataFrame:
    return pd.DataFrame(columns=VALIDATION_COLUMNS)


def _row(
    row_index: int | None,
    id_value: object,
    status: str,
    reason: str,
    missing_landmarks: str = "",
    suggested_action: str = "",
) -> dict:
    return {
        "Row": row_index if row_index is not None else "",
        "ID": id_value,
        "Status": status,
        "Reason": reason,
        "Missing_landmarks": missing_landmarks,
        "Suggested_action": suggested_action,
    }


def combine_validation_tables(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [t for t in tables if t is not None and not t.empty]
    if not non_empty:
        return empty_validation_table()
    return pd.concat(non_empty, ignore_index=True)[VALIDATION_COLUMNS]


def check_duplicate_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Flag every row sharing an ID with at least one other row.

    Duplicate IDs are flagged, never merged or dropped: repeated rows of one
    individual must be neither treated as independent specimens nor silently
    collapsed. A human decides what a duplicate ID means.
    """
    rows: list[dict] = []
    if ID_COL not in df.columns:
        return empty_validation_table()
    dup_mask = df[ID_COL].astype(str).duplicated(keep=False)
    for idx in df.index[dup_mask]:
        rows.append(
            _row(
                int(idx),
                df.at[idx, ID_COL],
                STATUS_WARNING,
                "Duplicate ID: another row shares this exact ID.",
                suggested_action=(
                    "Confirm whether these are genuinely repeated observations of the same "
                    "individual (e.g. re-digitizing, bilateral sides) before treating them as "
                    "independent specimens in any analysis."
                ),
            )
        )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS) if rows else empty_validation_table()


# A row imported with a blank ID gets this placeholder plus its spreadsheet row
# number. The ID column must be text everywhere downstream (sorting, joining,
# exporting); a bare NaN there breaks those steps and leaves the row unflagged.
MISSING_ID_PREFIX = "(missing ID"


def fill_missing_ids(df: pd.DataFrame, first_data_row: int = 2) -> pd.DataFrame:
    """Replace blank or missing IDs with a visible placeholder naming the source row.

    `df.index` must still be the position each row had when read (pandas'
    default), and `first_data_row` the spreadsheet row of position 0 (2 for a
    sheet with one header row), so the placeholder points at the cell to fill in.
    The ID is never guessed; `check_missing_ids` flags every placeholder as Fatal.
    """
    if ID_COL not in df.columns:
        return df
    ids = df[ID_COL]
    blank = ids.isna() | (
        ids.astype(object).map(lambda v: str(v).strip() if v is not None else "") == ""
    )
    if not blank.any():
        return df
    df = df.copy()
    positions = np.flatnonzero(blank.to_numpy())
    labels = [
        f"{MISSING_ID_PREFIX}, spreadsheet row {first_data_row + int(df.index[p])})"
        for p in positions
    ]
    df[ID_COL] = df[ID_COL].astype(object)
    df.loc[df.index[positions], ID_COL] = labels
    df[ID_COL] = df[ID_COL].astype(str)
    return df


def check_missing_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows whose ID was blank in the source (or is still missing)."""
    if ID_COL not in df.columns:
        return empty_validation_table()
    rows: list[dict] = []
    for idx in df.index:
        value = df.at[idx, ID_COL]
        text = "" if pd.isna(value) else str(value).strip()
        if text == "" or text.startswith(MISSING_ID_PREFIX):
            rows.append(
                _row(
                    int(idx),
                    text or value,
                    STATUS_FATAL,
                    "This row has no specimen ID in the source file.",
                    suggested_action=(
                        "Fill in the ID in the source spreadsheet and re-import, or "
                        "exclude this row. It cannot be matched to anything by ID."
                    ),
                )
            )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS) if rows else empty_validation_table()


def check_landmark_completeness(df: pd.DataFrame, landmark_cols: Sequence[str]) -> pd.DataFrame:
    """Flag rows with any missing landmark coordinate; Fatal if all are missing."""
    rows: list[dict] = []
    if not landmark_cols:
        return empty_validation_table()
    coords = df.loc[:, list(landmark_cols)]
    for idx in df.index:
        vals = coords.loc[idx].to_numpy(dtype=float)
        n_missing = int(np.isnan(vals).sum())
        if n_missing == 0:
            continue
        id_value = df.at[idx, ID_COL] if ID_COL in df.columns else idx
        if n_missing == len(vals):
            rows.append(
                _row(
                    int(idx),
                    id_value,
                    STATUS_FATAL,
                    "No landmark coordinates present for this row.",
                    missing_landmarks="all",
                    suggested_action="Re-check the source file; this row cannot be used in any GPA or shape analysis.",
                )
            )
        else:
            missing_names = [landmark_cols[i] for i, v in enumerate(vals) if np.isnan(v)]
            rows.append(
                _row(
                    int(idx),
                    id_value,
                    STATUS_WARNING,
                    f"{n_missing} of {len(vals)} landmark coordinate values are missing.",
                    missing_landmarks=", ".join(missing_names),
                    suggested_action="Complete or exclude this specimen before running GPA; partial rows cannot be aligned.",
                )
            )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS) if rows else empty_validation_table()


def check_sex_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Flag Sex values that don't normalize to M/F/X, as an informational note.

    The raw value is preserved either way (see gmm_core.data_model.normalize_sex);
    this only surfaces it for review.
    """
    rows: list[dict] = []
    if SEX_COL not in df.columns:
        return empty_validation_table()
    for idx in df.index:
        raw_value = df.at[idx, SEX_COL]
        normalized = normalize_sex(raw_value)
        if normalized not in {"M", "F", "X"}:
            id_value = df.at[idx, ID_COL] if ID_COL in df.columns else idx
            rows.append(
                _row(
                    int(idx),
                    id_value,
                    STATUS_INFO,
                    f"Unrecognized Sex code {raw_value!r} (kept as-is, not coerced to M/F/X).",
                    suggested_action="Review and correct the source data if this was a data-entry error.",
                )
            )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS) if rows else empty_validation_table()


def validate_dataset(df: pd.DataFrame, landmark_cols: Sequence[str]) -> pd.DataFrame:
    """Run every structural check and return one combined validation table."""
    return combine_validation_tables(
        [
            check_missing_ids(df),
            check_duplicate_ids(df),
            check_landmark_completeness(df, landmark_cols),
            check_sex_codes(df),
        ]
    )


def flag_landmark_count_mismatch(counts_by_source: dict) -> tuple:
    """Given {source_name: landmark_count}, return (expected_count, mismatch_rows).

    Used by every importer that reads one specimen per file (Amira, Slicer,
    Morphologika) to apply the same rule everywhere: files in the same batch with a different landmark count are flagged
    and excluded from the combined table - never padded, never guessed,
    never silently dropped.

    `expected_count` is the majority count across the batch. Ties fall back
    to the smallest count seen, to avoid guessing a specific "correct" template.
    """
    if not counts_by_source:
        return None, empty_validation_table()
    counts = list(counts_by_source.values())
    values, freq = np.unique(counts, return_counts=True)
    max_freq = freq.max()
    candidates = sorted(values[freq == max_freq])
    expected = int(candidates[0])

    rows: list[dict] = []
    for source_name, count in counts_by_source.items():
        if count != expected:
            rows.append(
                _row(
                    None,
                    source_name,
                    STATUS_FATAL,
                    f"Landmark count ({count}) does not match the batch's expected count ({expected}).",
                    missing_landmarks="",
                    suggested_action=(
                        "Excluded from the combined table. Check whether this file uses a "
                        "different template or is missing/extra landmarks, then re-import separately."
                    ),
                )
            )
    return expected, (
        pd.DataFrame(rows, columns=VALIDATION_COLUMNS) if rows else empty_validation_table()
    )
