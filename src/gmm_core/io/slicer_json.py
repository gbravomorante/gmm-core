"""3D Slicer ``.mrk.json`` importer.

One specimen per file, read from ``markups[0].controlPoints``. Besides the
position, it reads each point's label and ``positionStatus``: SlicerMorph
records a point it could not place as ``"missing"`` or ``"undefined"``,
usually with a leftover position, and those become real gaps rather than
wrong coordinates.

:func:`import_slicer_any` reads ``.mrk.json`` and ``.fcsv`` files together,
because real collections often contain both.

There is no filter on an expected landmark count: a specimen with a
different count is flagged and excluded (see
:func:`gmm_core.io.common.build_canonical_from_specimens`), never silently
dropped. No metadata is parsed from file names.
"""

from __future__ import annotations

import json
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

SOURCE_FORMAT_NAME = "Slicer_mrk_json"
_MISSING_STATUS = {"missing", "undefined"}


def list_slicer_json_files_recursive(folder: str | Path) -> list[Path]:
    return sorted(Path(folder).rglob("*.mrk.json"))


def _specimen_id_for(path: Path) -> str:
    name = path.name
    return name[: -len(".mrk.json")] if name.endswith(".mrk.json") else path.stem


def _parse_one_file(path: Path) -> tuple[np.ndarray, list[str], list[str], str]:
    """(coordinates, labels, issues, coordinate system)."""
    empty = np.empty((0, 3))
    try:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return empty, [], [f"{path.name}: could not read file ({exc})."], "unknown"

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return empty, [], [f"{path.name}: invalid JSON ({exc})."], "unknown"

    markups = data.get("markups")
    if not markups:
        return empty, [], [f"{path.name}: no 'markups' entry found."], "unknown"

    control_points = markups[0].get("controlPoints") or []
    if not control_points:
        return empty, [], [f"{path.name}: no controlPoints found in the first markup."], "unknown"

    coords: list[list[float]] = []
    labels: list[str] = []
    issues: list[str] = []
    for i, point in enumerate(control_points):
        labels.append(str(point.get("label") or f"F-{i + 1}"))
        if str(point.get("positionStatus", "defined")).lower() in _MISSING_STATUS:
            coords.append([np.nan, np.nan, np.nan])
            continue
        position = point.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            issues.append(
                f"{path.name}: controlPoint {i} has an invalid 'position' value ({position!r})."
            )
            continue
        try:
            coords.append([float(v) for v in position])
        except (TypeError, ValueError):
            issues.append(f"{path.name}: controlPoint {i} position is not numeric ({position!r}).")

    if issues:
        return empty, [], issues, "unknown"
    system = str(markups[0].get("coordinateSystem") or "unknown").upper()
    return np.array(coords, dtype=float), labels, [], system


def _assemble(
    specimens: dict[str, np.ndarray],
    labels: dict[str, list[str]],
    systems: dict[str, str],
    source_file_by_id: dict[str, str],
    extra_rows: list[dict],
    source_format: str,
) -> ImportBatchResult:
    specimens, system_rows = reconcile_coordinate_systems(specimens, systems)
    specimens, names, label_rows = reconcile_labels(specimens, labels)
    df, landmark_cols, validation = build_canonical_from_specimens(
        specimens,
        source_format,
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


def import_slicer_any(paths: Sequence[str | Path]) -> ImportBatchResult:
    """A batch of .mrk.json and .fcsv files, in any mixture."""
    from gmm_core.io.slicer_fcsv import read_fcsv

    specimens: dict[str, np.ndarray] = {}
    labels: dict[str, list[str]] = {}
    systems: dict[str, str] = {}
    source_file_by_id: dict[str, str] = {}
    extra_rows: list[dict] = []
    kinds = set()

    for raw_path in paths:
        path = Path(raw_path)
        is_fcsv = path.suffix.lower() == ".fcsv"
        specimen_id = path.stem if is_fcsv else _specimen_id_for(path)
        if is_fcsv:
            try:
                coords, point_labels, system = read_fcsv(path)
                issues: list[str] = []
            except ValueError as exc:
                coords, point_labels, system, issues = np.empty((0, 3)), [], "unknown", [str(exc)]
        else:
            coords, point_labels, issues, system = _parse_one_file(path)
        if issues:
            for issue in issues:
                extra_rows.append(
                    {
                        "Row": "",
                        "ID": specimen_id,
                        "Status": STATUS_FATAL,
                        "Reason": issue,
                        "Missing_landmarks": "",
                        "Suggested_action": (
                            "Fix the source file and re-import; this file was excluded "
                            "from the combined table."
                        ),
                    }
                )
            continue
        kinds.add("Slicer_fcsv" if is_fcsv else SOURCE_FORMAT_NAME)
        specimens[specimen_id] = coords
        labels[specimen_id] = point_labels
        systems[specimen_id] = system
        source_file_by_id[specimen_id] = path.name

    source_format = kinds.pop() if len(kinds) == 1 else "Slicer"
    return _assemble(specimens, labels, systems, source_file_by_id, extra_rows, source_format)


def import_slicer_files(paths: Sequence[str | Path]) -> ImportBatchResult:
    return import_slicer_any(paths)
