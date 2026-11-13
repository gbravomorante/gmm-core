"""Amira ``.landmarkAscii`` importer.

One specimen per file; coordinates sit between the ``@1`` and ``@2`` markers.

A file with any malformed line is flagged and excluded whole, rather than
padded or partially trusted. A file whose landmark count differs from the
rest of the batch is flagged and excluded too (see
:func:`gmm_core.io.common.build_canonical_from_specimens`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from gmm_core.io.common import ImportBatchResult, build_canonical_from_specimens
from gmm_core.validation import STATUS_FATAL, STATUS_INFO, STATUS_WARNING

SOURCE_FORMAT_NAME = "Amira_landmarkAscii"

# Amira/Avizo records the coordinate unit in the file's own Parameters block:
#
#     Parameters {
#         Units {
#             Coordinates "nm"
#         }
#
# This is not decoration. Real files exported from Avizo in this project's own
# data declare **nanometres**, so a coordinate of -1.33e+07 means -13.3 mm.
# Reading them as millimetres leaves shape analysis unharmed - GPA removes
# scale - but silently corrupts everything size-related: centroid size, any
# derived distance, and above all allometry, which regresses shape on log(size).
# Worse, mixing an nm file and a mm file in one batch puts specimens a million-
# fold apart in size while GPA still happily aligns them.
#
# Everything is therefore converted to millimetres on import, and the declared
# unit is reported so the conversion is visible rather than assumed.
MM_PER_UNIT = {
    "nm": 1e-6,
    "nanometer": 1e-6,
    "nanometre": 1e-6,
    "um": 1e-3,
    "µm": 1e-3,
    "micron": 1e-3,
    "micrometer": 1e-3,
    "micrometre": 1e-3,
    "mm": 1.0,
    "millimeter": 1.0,
    "millimetre": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimetre": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "metre": 1000.0,
}
DEFAULT_UNIT = "mm"


def list_landmarkascii_files_recursive(folder: str | Path) -> list[Path]:
    return sorted(Path(folder).rglob("*.landmarkAscii"))


def _declared_unit(lines: Sequence[str]) -> str:
    """The unit named in the Parameters/Units block, lowercased.

    Returns "" when the file does not declare one, which is different from
    declaring millimetres and is reported differently.
    """
    in_units = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("Units"):
            in_units = True
            continue
        if in_units:
            if stripped.startswith("}"):
                break
            if stripped.lower().startswith("coordinates"):
                remainder = stripped[len("coordinates") :].strip()
                return remainder.strip('",').strip().lower()
        if stripped.startswith("@1"):
            break
    return ""


def _declared_count(header_lines: Sequence[str]) -> int | None:
    """The N of 'define Markers N' in the header, or None when the file does not say."""
    for line in header_lines:
        match = re.match(r"\s*define\s+Markers\s+(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def _parse_one_file(path: Path, to_mm: bool = True) -> tuple[np.ndarray, list[str]]:
    """Returns (coords (n, 3), issues). coords is empty when issues is non-empty.

    Coordinates are converted to millimetres using the unit the file declares.
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return np.empty((0, 3)), [f"{path.name}: could not read file ({exc})."]

    unit = _declared_unit(lines)
    if unit and unit not in MM_PER_UNIT:
        return np.empty((0, 3)), [
            f"{path.name}: declares an unrecognised coordinate unit {unit!r}. Supported: "
            f"{', '.join(sorted(set(MM_PER_UNIT)))}."
        ]

    start_idx = None
    end_idx = len(lines)
    for i, line in enumerate(lines):
        s = line.strip()
        if s == "@1" and start_idx is None:
            start_idx = i + 1
        elif s == "@2" and start_idx is not None:
            end_idx = i
            break

    if start_idx is None:
        return np.empty((0, 3)), [f"{path.name}: no '@1' marker found."]

    coords: list[list[float]] = []
    issues: list[str] = []
    for line_no, line in enumerate(lines[start_idx:end_idx], start=start_idx + 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) != 3:
            issues.append(
                f"{path.name} line {line_no}: expected 3 coordinate values, found {len(parts)} ({s!r})."
            )
            continue
        try:
            coords.append([float(p) for p in parts])
        except ValueError:
            issues.append(f"{path.name} line {line_no}: could not parse coordinates ({s!r}).")

    if issues:
        return np.empty((0, 3)), issues
    if not coords:
        return np.empty((0, 3)), [f"{path.name}: no coordinate lines found between @1 and @2."]
    # The file states its own landmark count. Without this check a truncated file imported
    # alone became a valid-looking specimen with fewer landmarks (all 9807 real files checked
    # on 2026-10-28 agree with their declaration).
    declared = _declared_count(lines[:start_idx])
    if declared is not None and declared != len(coords):
        return np.empty((0, 3)), [
            f"{path.name}: declares {declared} landmarks ('define Markers {declared}') but "
            f"{len(coords)} coordinate lines were found - the file may be truncated."
        ]

    array = np.array(coords, dtype=float)
    if to_mm and unit:
        array = array * MM_PER_UNIT[unit]
    return array, []


def file_unit(path: str | Path) -> str:
    """The coordinate unit a file declares, or "" if it declares none."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return _declared_unit(lines)


def import_amira_files(paths: Sequence[str | Path]) -> ImportBatchResult:
    specimens: dict[str, np.ndarray] = {}
    source_file_by_id: dict[str, str] = {}
    extra_rows: list[dict] = []

    units_seen: dict[str, list[str]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        unit = file_unit(path)
        units_seen.setdefault(unit or "(not declared)", []).append(path.name)
        coords, issues = _parse_one_file(path)
        if issues:
            for issue in issues:
                extra_rows.append(
                    {
                        "Row": "",
                        "ID": path.stem,
                        "Status": STATUS_FATAL,
                        "Reason": issue,
                        "Missing_landmarks": "",
                        "Suggested_action": (
                            "Fix the source .landmarkAscii file and re-import; this file was "
                            "excluded from the combined table."
                        ),
                    }
                )
            continue
        specimens[path.stem] = coords
        source_file_by_id[path.stem] = path.name

    # Report the unit conversion rather than performing it invisibly. A file
    # in nanometres becoming millimetres changes every size-based result, and a
    # batch that mixes units silently corrupts any allometry analysis, so both
    # are surfaced here, at import time.
    converted = {
        u: files for u, files in units_seen.items() if u not in ("(not declared)", DEFAULT_UNIT)
    }
    for unit, files in converted.items():
        extra_rows.append(
            {
                "Row": "",
                "ID": "(batch)",
                "Status": STATUS_INFO,
                "Reason": (
                    f"{len(files)} file(s) declared coordinates in '{unit}' and were "
                    f"converted to millimetres (x{MM_PER_UNIT.get(unit, 1.0):g})."
                ),
                "Missing_landmarks": "",
                "Suggested_action": "No action needed - recorded so the change to size is visible.",
            }
        )
    if len(units_seen) > 1:
        extra_rows.append(
            {
                "Row": "",
                "ID": "(batch)",
                "Status": STATUS_WARNING,
                "Reason": (
                    "This batch mixes coordinate units ("
                    + ", ".join(f"{u}: {len(f)} file(s)" for u, f in sorted(units_seen.items()))
                    + "). Files declaring a unit were converted to millimetres; files declaring "
                    "none were assumed to be in millimetres already."
                ),
                "Missing_landmarks": "",
                "Suggested_action": (
                    "Check the files that declare no unit. If they are not in "
                    "millimetres, every size-based result will be wrong."
                ),
            }
        )

    df, landmark_cols, validation = build_canonical_from_specimens(
        specimens,
        SOURCE_FORMAT_NAME,
        source_file_by_id,
        n_dim=3,
        extra_validation_rows=extra_rows,
    )

    return ImportBatchResult(
        raw=df,
        landmark_cols=landmark_cols,
        n_landmarks=len(landmark_cols) // 3 if landmark_cols else 0,
        n_dim=3,
        validation=validation,
        source_files=sorted(set(source_file_by_id.values())),
    )
