"""The canonical data model shared by every importer and analysis.

A dataset is one pandas table: an ``ID`` column, optional metadata columns,
and landmark coordinates in ``LM<n>_X``, ``LM<n>_Y`` (and ``LM<n>_Z``)
columns. This module defines those names and the helpers that move between
the table and ``(n_specimens, n_landmarks, n_dim)`` arrays. It has no UI
dependencies and is safe to use from plain Python.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

ID_COL = "ID"
SOURCE_FORMAT_COL = "SourceFormat"
SOURCE_FILE_COL = "SourceFile"
SEX_COL = "Sex"

# Columns that are never treated as metadata and never treated as landmarks.
RESERVED_COLS = {ID_COL, SOURCE_FORMAT_COL, SOURCE_FILE_COL}

VALID_SEX_CODES = {"M", "F", "X"}

# Case-insensitive on purpose: real datasets write `LM0_x` as often as
# `LM0_X`, and rejecting one of them stops a whole file from loading for a
# reason nothing in the error message makes obvious. The canonical form
# written by this package is always upper-case - see `canonical_landmark_name`.
_LM_COL_PATTERN = re.compile(r"^LM(\d+)_([XYZ])$", re.IGNORECASE)
_AXIS_ORDER = {"X": 0, "Y": 1, "Z": 2}
_AXES_BY_DIM = {2: ["X", "Y"], 3: ["X", "Y", "Z"]}


def is_landmark_column(name: object) -> bool:
    return bool(_LM_COL_PATTERN.match(str(name)))


def canonical_landmark_name(name: object) -> str:
    """`LM0_x` -> `LM0_X`. Any non-landmark name is returned unchanged."""
    match = _LM_COL_PATTERN.match(str(name))
    if not match:
        return str(name)
    return f"LM{int(match.group(1))}_{match.group(2).upper()}"


def canonicalise_landmark_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename landmark columns to the canonical upper-case spelling.

    Applied by every importer so every later step sees one spelling only.
    Returns the same frame when nothing needs renaming, so the common case
    costs nothing.
    """
    renames = {
        c: canonical_landmark_name(c)
        for c in df.columns
        if is_landmark_column(c) and str(c) != canonical_landmark_name(c)
    }
    return df.rename(columns=renames) if renames else df


def landmark_columns(n_landmarks: int, n_dim: int = 3, first_index: int = 0) -> list[str]:
    """Generate canonical landmark column names for i in [first_index, first_index+n_landmarks)."""
    axes = _AXES_BY_DIM[n_dim]
    cols: list[str] = []
    for i in range(first_index, first_index + n_landmarks):
        for axis in axes:
            cols.append(f"LM{i}_{axis}")
    return cols


def natural_sort_landmark_columns(columns: Sequence[object]) -> list[str]:
    """Return the landmark columns present in `columns`, sorted numerically by
    landmark index then axis order (X, Y, Z) - never alphabetically.

    Alphabetical sort would place LM10_X before LM2_X, silently reordering
    landmarks.
    """
    found: list[tuple] = []
    for col in columns:
        match = _LM_COL_PATTERN.match(str(col))
        if match:
            index = int(match.group(1))
            axis = _AXIS_ORDER[match.group(2).upper()]
            found.append((index, axis, str(col)))
    found.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in found]


def detect_dimensionality(columns: Sequence[object]) -> int:
    """Return 3 if any LM#_Z column is present among the landmark columns, else 2."""
    for col in columns:
        match = _LM_COL_PATTERN.match(str(col))
        if match and match.group(2).upper() == "Z":
            return 3
    return 2


def get_landmark_indices(columns: Sequence[object]) -> list[int]:
    """Sorted unique landmark indices implied by the columns present."""
    indices = set()
    for col in columns:
        match = _LM_COL_PATTERN.match(str(col))
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def landmark_count(columns: Sequence[object]) -> int:
    return len(get_landmark_indices(columns))


def extract_landmark_array(
    df: pd.DataFrame, landmark_cols: Sequence[str], n_dim: int
) -> np.ndarray:
    """(n_specimens, n_landmarks, n_dim) float array, in the column order given."""
    n_landmarks = len(landmark_cols) // n_dim
    values = df.loc[:, list(landmark_cols)].to_numpy(dtype=float)
    return values.reshape(len(df), n_landmarks, n_dim)


def flatten_landmark_array(shapes: np.ndarray) -> np.ndarray:
    return shapes.reshape(shapes.shape[0], -1)


def landmarks_to_row_dict(shape: np.ndarray, n_dim: int = 3) -> dict[str, float]:
    """One specimen's (n_landmarks, n_dim) array -> {'LM0_X': ..., ...} dict."""
    axes = _AXES_BY_DIM[n_dim]
    out: dict[str, float] = {}
    for i in range(shape.shape[0]):
        for a, axis in enumerate(axes):
            out[f"LM{i}_{axis}"] = float(shape[i, a])
    return out


def normalize_sex(value: Any) -> str:
    """Normalize a sex code to M/F/X.

    Unknown non-blank text is preserved verbatim (not collapsed to X) so it
    surfaces in validation rather than silently becoming "unknown". Unknown-sex
    specimens are never excluded automatically.
    """
    if value is None:
        return "X"
    if isinstance(value, float) and np.isnan(value):
        return "X"
    text = str(value).strip()
    if text == "":
        return "X"
    upper = text.upper()
    if upper in {"NA", "N/A", "NAN", "UNKNOWN", "UNK", "?", "X"}:
        return "X"
    if upper in {"M", "MALE"}:
        return "M"
    if upper in {"F", "FEMALE"}:
        return "F"
    return upper  # preserved verbatim; validation flags unrecognized codes


def find_landmark_structure_issues(columns: Sequence[object]) -> list[str]:
    """Sanity-check the landmark columns present, without changing anything.

    Returns a list of human-readable issue strings (empty if the structure is
    clean): indices must start at 0, run contiguously, and every landmark
    index must have all of its axis columns present. Callers should surface
    these as hard errors rather than guessing a fix - silently renumbering or
    padding would silently change landmark order.
    """
    indices = get_landmark_indices(columns)
    issues: list[str] = []
    if not indices:
        return ["No LM#_X/Y/Z landmark columns found."]

    if indices[0] != 0:
        issues.append(f"Landmark indices do not start at 0 (first index found: LM{indices[0]}).")

    expected_indices = list(range(indices[0], indices[-1] + 1))
    missing_indices = sorted(set(expected_indices) - set(indices))
    if missing_indices:
        issues.append(f"Landmark indices have gaps (missing index numbers: {missing_indices}).")

    n_dim = detect_dimensionality(columns)
    axes = _AXES_BY_DIM[n_dim]
    col_set = {str(c) for c in columns}
    for i in indices:
        for axis in axes:
            name = f"LM{i}_{axis}"
            if name not in col_set:
                issues.append(f"Landmark {i} is missing column {name} (has some axes but not all).")

    return issues


def metadata_columns(columns: Sequence[object]) -> list[str]:
    """Every column that is neither a landmark column nor a reserved column."""
    return [str(c) for c in columns if not is_landmark_column(c) and c not in RESERVED_COLS]


AGE_COL = "Age"
AGE_CLASS_COL = "AgeClass"
DEFAULT_SUBADULT_AGE_THRESHOLD = 16.0

_SUBADULT_TEXT_VALUES = {"subadult", "sub-adult", "sub adult", "juvenile", "juv", "immature"}
_ADULT_TEXT_VALUES = {"adult", "mature"}


def classify_age(value: Any, threshold: float = DEFAULT_SUBADULT_AGE_THRESHOLD) -> str:
    """Classify one Age cell into "Adult" / "Subadult" / "Unknown".

    Age is allowed to mix numeric years and free text ("Adult"/"subadult") in
    the same column. Convention: a specimen is Subadult if its text says
    so, or if its numeric age is below `threshold` years; anything else
    parseable is Adult. A blank/missing cell is classified "Unknown" rather
    than guessed as Adult.
    """
    if value is None:
        return "Unknown"
    if isinstance(value, float) and np.isnan(value):
        return "Unknown"
    text = str(value).strip()
    if text == "":
        return "Unknown"

    lower = text.lower()
    if lower in _SUBADULT_TEXT_VALUES:
        return "Subadult"
    if lower in _ADULT_TEXT_VALUES:
        return "Adult"

    try:
        numeric_age = float(text)
    except ValueError:
        return "Unknown"  # unrecognized text; don't guess
    return "Subadult" if numeric_age < threshold else "Adult"


def compute_age_class_column(
    df: pd.DataFrame, threshold: float = DEFAULT_SUBADULT_AGE_THRESHOLD
) -> pd.Series:
    return df[AGE_COL].map(lambda v: classify_age(v, threshold))


def virtual_metadata_column_names(columns: Sequence[object]) -> list[str]:
    """Derived metadata columns available for grouping/coloring/filtering,
    beyond what's literally in the dataframe. Currently just AgeClass.
    """
    names: list[str] = []
    if AGE_COL in columns and AGE_CLASS_COL not in columns:
        names.append(AGE_CLASS_COL)
    return names


def add_virtual_metadata_columns(
    df: pd.DataFrame, threshold: float = DEFAULT_SUBADULT_AGE_THRESHOLD
) -> pd.DataFrame:
    """Return a copy of df with derived metadata columns actually populated
    (currently AgeClass from Age, if present and not already materialized).

    Any caller that lists `virtual_metadata_column_names` as available
    options must call this on the dataframe it actually plots/groups, or the
    column named in the dropdown won't exist in the data.
    """
    if AGE_COL in df.columns and AGE_CLASS_COL not in df.columns:
        df = df.copy()
        df[AGE_CLASS_COL] = compute_age_class_column(df, threshold)
    return df


def numeric_coercible_columns(df: pd.DataFrame, candidate_columns: Sequence[str]) -> list[str]:
    """Which of candidate_columns have at least one numeric-coercible value.

    Used to offer sensible options for continuous/gradient coloring (e.g. a
    mixed Age column where most rows are numeric years but a few are text)
    without requiring the whole column to already be numeric dtype.
    """
    result: list[str] = []
    for col in candidate_columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            result.append(col)
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().any():
            result.append(col)
    return result


LABEL_COL = "Label"


def resolve_display_label(id_value: object, id_labels: dict[str, str]) -> str:
    """One ID's display label: the user-assigned alias if one exists for
    this exact ID, else the ID itself, unchanged.
    """
    return id_labels.get(str(id_value), str(id_value))


def compute_label_column(df: pd.DataFrame, id_labels: dict[str, str]) -> pd.Series:
    return df[ID_COL].map(lambda i: resolve_display_label(i, id_labels))


def add_display_label_column(df: pd.DataFrame, id_labels: dict[str, str]) -> pd.DataFrame:
    """Return a copy of df with a `Label` column added - the real `ID` value
    unless that specimen has been given a shorter display alias. `ID` itself is
    never modified; `Label` is only what plots and tables show instead.
    """
    df = df.copy()
    df[LABEL_COL] = compute_label_column(df, id_labels)
    return df


@dataclass
class GMMDataset:
    """All working data for one analysis, in a single serialisable object."""

    raw: pd.DataFrame
    landmark_cols: list[str]
    n_landmarks: int
    n_dim: int
    source_files: list[str] = field(default_factory=list)

    gpa: pd.DataFrame | None = None
    gpa_noscale: pd.DataFrame | None = None
    gpa_settings: dict[str, Any] = field(default_factory=dict)

    validation: pd.DataFrame | None = None
    excluded_ids: list[str] = field(default_factory=list)

    id_labels: dict[str, str] = field(default_factory=dict)

    run_settings: dict[str, Any] = field(default_factory=dict)

    derived_variable_defs: list[dict[str, Any]] = field(default_factory=list)
    wireframe_chains: list[list[int]] = field(default_factory=list)

    # Named landmark subsets ("inlet", "acetabulum", ...) -> 0-based landmark
    # indices. Entered by hand like wireframe chains: an anatomical region is
    # the user's knowledge, never inferred from the coordinates. Used as the
    # anchor and/or scale reference for subset-anchored alignment.
    landmark_subsets: dict[str, list[int]] = field(default_factory=dict)

    # Semilandmark curves: [{"name": str, "landmarks": [int], "closed": bool}].
    # Which points are semilandmarks is the user's digitizing protocol and is
    # entered by hand, never inferred - same stance as wireframe connectivity
    # and landmark subsets.
    semilandmark_curves: list[dict[str, Any]] = field(default_factory=list)

    # Landmarks in `raw` that are estimates rather than observations: ID -> 0-based indices. Once
    # aligned, an estimate looks like any other point, so this is the only record. See
    # gmm_core.missing_data.apply_estimates.
    estimated_landmarks: dict[str, list[int]] = field(default_factory=dict)

    module_outputs: dict[str, Any] = field(default_factory=dict)

    def meta_cols(self) -> list[str]:
        real_cols = metadata_columns(self.raw.columns)
        return real_cols + virtual_metadata_column_names(self.raw.columns)

    def included_raw(self) -> pd.DataFrame:
        """RAW rows minus specimens the user has explicitly excluded.

        Rows are never dropped from `self.raw` itself - only filtered out at
        analysis time - so nothing is silently lost. `self.excluded_ids` is only
        ever populated explicitly.
        """
        if not self.excluded_ids:
            return self.raw
        return self.raw.loc[~self.raw[ID_COL].isin(self.excluded_ids)].reset_index(drop=True)

    def included_gpa(self) -> pd.DataFrame | None:
        if self.gpa is None:
            return None
        if not self.excluded_ids:
            return self.gpa
        return self.gpa.loc[~self.gpa[ID_COL].isin(self.excluded_ids)].reset_index(drop=True)

    def included_gpa_noscale(self) -> pd.DataFrame | None:
        if self.gpa_noscale is None:
            return None
        if not self.excluded_ids:
            return self.gpa_noscale
        return self.gpa_noscale.loc[~self.gpa_noscale[ID_COL].isin(self.excluded_ids)].reset_index(
            drop=True
        )
