"""Shared assembly for importers that read one coordinate array per specimen
(Amira, 3D Slicer, Morphologika).

These formats deliver a batch of independently parsed specimens with no
metadata, and none is ever inferred from file names. Each importer turns
files into ``{specimen_id: (n_landmarks, n_dim) array}``, flags what it could
not parse, and hands off to :func:`build_canonical_from_specimens`, which
applies the checks every importer shares:

* landmark-count mismatches within the batch are flagged and excluded,
  never padded or guessed;
* the usual structural validation (duplicate IDs and so on).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gmm_core.data_model import (
    ID_COL,
    SOURCE_FILE_COL,
    SOURCE_FORMAT_COL,
    landmark_columns,
    landmarks_to_row_dict,
)
from gmm_core.validation import (
    STATUS_WARNING,
    VALIDATION_COLUMNS,
    combine_validation_tables,
    empty_validation_table,
    flag_landmark_count_mismatch,
    validate_dataset,
)


@dataclass
class ImportBatchResult:
    raw: pd.DataFrame
    landmark_cols: list[str]
    n_landmarks: int
    n_dim: int
    validation: pd.DataFrame
    source_files: list[str]
    landmark_names: list[str] | None = None
    wireframe_chains: list[list[int]] = field(default_factory=list)


def _batch_row(status: str, reason: str, action: str) -> dict:
    return {
        "Row": "",
        "ID": "(batch)",
        "Status": status,
        "Reason": reason,
        "Missing_landmarks": "",
        "Suggested_action": action,
    }


def reconcile_labels(
    specimens: dict[str, np.ndarray],
    labels: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], list[str] | None, list[dict]]:
    """Files that name their points are compared by name, not trusted by position.

    Same names in a different order: reordered to the commonest order, and reported. Slicer
    keeps points in placing order, so a point placed late sits in another position, and a
    positional read would compare different anatomy without any error.
    Different names: left as they are and reported, because which point is which then needs
    a person. Only a batch where every file agrees yields landmark names.
    """
    if not labels:
        return specimens, None, []
    orders = {sid: tuple(labels[sid]) for sid in specimens if sid in labels}
    if not orders:
        return specimens, None, []
    counts: dict[tuple[str, ...], int] = {}
    for order in orders.values():
        counts[order] = counts.get(order, 0) + 1
    reference = max(counts, key=counts.get)
    if len(set(reference)) != len(reference):
        return (
            specimens,
            None,
            [
                _batch_row(
                    STATUS_WARNING,
                    "Some landmark labels repeat within a file, so points were kept in file order.",
                    "Give each landmark its own label to have them matched by name.",
                )
            ],
        )
    out = dict(specimens)
    reordered, mismatched = [], []
    for sid, order in orders.items():
        if order == reference:
            continue
        if sorted(order) == sorted(reference) and len(set(order)) == len(order):
            out[sid] = specimens[sid][[order.index(name) for name in reference]]
            reordered.append(sid)
        elif len(order) == len(reference):
            mismatched.append(sid)
    rows = []
    if reordered:
        listed = ", ".join(reordered[:6]) + (
            f" and {len(reordered) - 6} more" if len(reordered) > 6 else ""
        )
        rows.append(
            _batch_row(
                STATUS_WARNING,
                f"{len(reordered)} file(s) list the same landmark names in a different order and were "
                f"reordered by name to match the rest: {listed}.",
                "Nothing to do unless the names themselves are wrong.",
            )
        )
    if mismatched:
        listed = ", ".join(mismatched[:6]) + (
            f" and {len(mismatched) - 6} more" if len(mismatched) > 6 else ""
        )
        rows.append(
            _batch_row(
                STATUS_WARNING,
                f"{len(mismatched)} file(s) use different landmark names from the rest and were kept in "
                f"file order, so their points may not correspond: {listed}.",
                "Check those files. Rename the points in Slicer if they are the same anatomy.",
            )
        )
        return out, None, rows
    return out, list(reference), rows


def reconcile_coordinate_systems(
    specimens: dict[str, np.ndarray],
    systems: dict[str, str],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """LPS and RAS differ by negating x and y, a half turn about z. Shape is unaffected, but
    coordinates, sizes along an axis and any overlay on a scan are not. Files declaring the
    minority system are converted to the majority one, and that is reported."""
    declared = {sid: s for sid, s in systems.items() if s in ("LPS", "RAS") and sid in specimens}
    if len(set(declared.values())) < 2:
        return specimens, []
    values = list(declared.values())
    majority = max(set(values), key=values.count)
    out = dict(specimens)
    converted = []
    for sid, system in declared.items():
        if system != majority:
            flipped = np.array(specimens[sid], dtype=float, copy=True)
            flipped[:, :2] *= -1.0
            out[sid] = flipped
            converted.append(sid)
    return out, [
        _batch_row(
            STATUS_WARNING,
            f"This batch mixes coordinate systems. {len(converted)} file(s) declared the minority "
            f"system and were converted to {majority}: {', '.join(converted[:6])}"
            f"{' and more' if len(converted) > 6 else ''}.",
            "Check that those files really are in the system they declare.",
        )
    ]


def build_canonical_from_specimens(
    specimens: dict[str, np.ndarray],
    source_format: str,
    source_file_by_id: dict[str, str],
    n_dim: int,
    extra_validation_rows: list[dict],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Combine per-specimen (n_landmarks, n_dim) arrays into the canonical
    schema. Returns (raw_df, landmark_cols, validation_table).
    """
    counts = {sid: arr.shape[0] for sid, arr in specimens.items()}
    expected_n, mismatch_table = flag_landmark_count_mismatch(counts)

    mismatched_ids = set(mismatch_table["ID"]) if not mismatch_table.empty else set()
    kept_ids = [sid for sid in specimens if sid not in mismatched_ids]

    landmark_cols = landmark_columns(expected_n, n_dim) if expected_n else []
    all_cols = [ID_COL] + landmark_cols + [SOURCE_FORMAT_COL, SOURCE_FILE_COL]

    rows = []
    for sid in kept_ids:
        row = {ID_COL: sid}
        row.update(landmarks_to_row_dict(specimens[sid], n_dim))
        row[SOURCE_FORMAT_COL] = source_format
        row[SOURCE_FILE_COL] = source_file_by_id.get(sid, "")
        rows.append(row)

    df = pd.DataFrame(rows, columns=all_cols) if rows else pd.DataFrame(columns=all_cols)

    extra_table = (
        pd.DataFrame(extra_validation_rows, columns=VALIDATION_COLUMNS)
        if extra_validation_rows
        else empty_validation_table()
    )
    structural_table = (
        validate_dataset(df, landmark_cols) if not df.empty else empty_validation_table()
    )
    validation = combine_validation_tables([extra_table, mismatch_table, structural_table])

    return df, landmark_cols, validation
