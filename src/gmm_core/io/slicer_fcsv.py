"""3D Slicer .fcsv importer (Markups Fiducial CSV).

The older sibling of .mrk.json, and still what most existing landmark sets are
saved as, including ALPACA templates. One specimen per file.

Format: comment lines starting with '#', then one row per point:

    id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID

Coordinate system matters here and is not guessable from the numbers. Slicer
writes `# CoordinateSystem = LPS` or `RAS` in the header. LPS and RAS differ by
negating x and y, a half turn about z. GPA absorbs that, so shape is unaffected,
but coordinates and any overlay on a scan are not. The declared system is therefore
read and recorded; a batch that mixes them is converted to its majority, and says so.

Like every other importer here: no metadata is parsed from filenames, and a
file that cannot be fully parsed is excluded whole and reported rather than
partially trusted.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from gmm_core.io.common import (
    ImportBatchResult,
    build_canonical_from_specimens,
    reconcile_coordinate_systems,
    reconcile_labels,
)
from gmm_core.validation import STATUS_FATAL

SOURCE_FORMAT_NAME = "Slicer_fcsv"
N_FIXED_COLUMNS = 4  # id,x,y,z - everything after is optional in practice
LABEL_COLUMN = 11


def list_fcsv_files_recursive(folder: str | Path) -> list[Path]:
    return sorted(Path(folder).rglob("*.fcsv"))


def read_fcsv(path: str | Path) -> tuple[np.ndarray, list[str], str]:
    """(coordinates, labels, coordinate_system) for one .fcsv file.

    Raises ValueError with a specific message on anything malformed - callers
    that want the file skipped rather than raised should catch it.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"{path.name}: could not read file ({exc}).") from None

    coordinate_system = "unknown"
    coords: list[list[float]] = []
    labels: list[str] = []

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            lowered = stripped.lower().replace(" ", "")
            if "coordinatesystem=" in lowered:
                coordinate_system = lowered.split("coordinatesystem=", 1)[1].strip().upper()
            continue

        fields = stripped.split(",")
        if len(fields) < N_FIXED_COLUMNS:
            raise ValueError(
                f"{path.name}: line {number} has {len(fields)} field(s), expected at least "
                f"{N_FIXED_COLUMNS} (id,x,y,z)."
            )
        try:
            coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
        except ValueError:
            raise ValueError(
                f"{path.name}: line {number} has non-numeric coordinates ({fields[1:4]!r})."
            ) from None
        labels.append(
            fields[LABEL_COLUMN].strip() if len(fields) > LABEL_COLUMN else f"LM_{len(coords)}"
        )

    if not coords:
        raise ValueError(f"{path.name}: no fiducial points found.")
    return np.asarray(coords, dtype=float), labels, coordinate_system


def lps_to_ras(coords: np.ndarray) -> np.ndarray:
    """Convert between LPS and RAS by negating x and y. The conversion is its
    own inverse, so one function covers both directions."""
    out = np.array(coords, dtype=float, copy=True)
    out[:, 0] *= -1.0
    out[:, 1] *= -1.0
    return out


def import_fcsv_files(
    paths: Sequence[str | Path],
    to_ras: bool = False,
) -> ImportBatchResult:
    specimens: dict[str, np.ndarray] = {}
    labels: dict[str, list[str]] = {}
    systems: dict[str, str] = {}
    source_file_by_id: dict[str, str] = {}
    extra_rows: list[dict] = []

    for raw_path in paths:
        path = Path(raw_path)
        specimen_id = path.stem
        try:
            coords, point_labels, system = read_fcsv(path)
        except ValueError as exc:
            extra_rows.append(
                {
                    "Row": "",
                    "ID": specimen_id,
                    "Status": STATUS_FATAL,
                    "Reason": str(exc),
                    "Missing_landmarks": "",
                    "Suggested_action": (
                        "Fix the source .fcsv file and re-import; this file was excluded from "
                        "the combined table."
                    ),
                }
            )
            continue
        specimens[specimen_id] = coords
        labels[specimen_id] = point_labels
        systems[specimen_id] = system
        source_file_by_id[specimen_id] = path.name

    specimens, system_rows = reconcile_coordinate_systems(specimens, systems)
    if to_ras:
        for specimen_id, system in systems.items():
            if specimen_id in specimens and system == "LPS" and not system_rows:
                specimens[specimen_id] = lps_to_ras(specimens[specimen_id])
    specimens, names, label_rows = reconcile_labels(specimens, labels)
    df, landmark_cols, validation = build_canonical_from_specimens(
        specimens,
        SOURCE_FORMAT_NAME,
        source_file_by_id,
        n_dim=3,
        extra_validation_rows=extra_rows + system_rows + label_rows,
    )
    n_landmarks = len(landmark_cols) // 3 if landmark_cols else 0
    return ImportBatchResult(
        raw=df,
        landmark_cols=landmark_cols,
        n_landmarks=n_landmarks,
        n_dim=3,
        validation=validation,
        source_files=sorted(set(source_file_by_id.values())),
        landmark_names=names if names and len(names) == n_landmarks else None,
    )
