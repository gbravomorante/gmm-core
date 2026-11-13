"""General tabular landmark importer: long CSV/TXT, wide CSV/TXT, and TPS files.

These are the layouts published GMM datasets actually arrive in, none of which
is the canonical workbook:

  * **long** - one row per landmark: specimen, landmark name, x, y[, z], extras.
  * **wide** - one row per specimen: coordinate columns plus metadata columns.
    Coordinates are recognised by name (LM1_X, X1, 1.X, Landmark3_y) or as a
    sequential run (V1, V2, ... read x, y, z, x, y, z), never by being numeric:
    an Age or a Length column must not become a landmark.
  * **TPS** - LM=/LM3= blocks with ID=, IMAGE=, SCALE=.

Everything is detected from the file and can be overridden explicitly
(`TabularOptions`). What was detected is recorded on the result and in the
validation table, so an import can be audited rather than trusted.

The rules every other importer follows apply here too:
a specimen with a different landmark set is excluded and reported, never padded;
landmarks are matched by NAME, never by row position; nothing is silently
dropped. Unlike Amira/Slicer/Morphologika these files do carry metadata, so
those columns are kept. Filenames are never parsed for any.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gmm_core.data_model import ID_COL, RESERVED_COLS, SOURCE_FORMAT_COL
from gmm_core.io.common import ImportBatchResult, build_canonical_from_specimens
from gmm_core.validation import (
    STATUS_FATAL,
    STATUS_INFO,
    STATUS_WARNING,
    VALIDATION_COLUMNS,
    check_sex_codes,
    combine_validation_tables,
)

FORMAT_LONG = "Tabular (long)"
FORMAT_WIDE = "Tabular (wide)"
FORMAT_TPS = "TPS"

LAYOUT_AUTO, LAYOUT_LONG, LAYOUT_WIDE, LAYOUT_TPS = "auto", "long", "wide", "tps"
LAYOUTS = [LAYOUT_AUTO, LAYOUT_LONG, LAYOUT_WIDE, LAYOUT_TPS]

# Text codes compare case-insensitively; numeric codes compare by value, so -999.0 and -999 agree.
DEFAULT_MISSING_CODES: tuple[str, ...] = ("", "NA", "N/A", "NaN", "-999", "9999")

_ID_NAMES = (
    "id",
    "specimen",
    "specimen_id",
    "specimenid",
    "specimen id",
    "individual",
    "indiv",
    "individual_id",
    "sample",
    "sample_id",
    "name",
    "specimen_name",
)
_LANDMARK_NAMES = (
    "landmark",
    "landmarks",
    "lm",
    "landmark_name",
    "landmark_id",
    "lm_name",
    "lmk",
    "point",
    "point_name",
)
_AXES = "xyz"

# LM1_X, lm1x, Landmark3.y, X1, x_1, 1.X, 1_x. Group order: (prefix-number, axis) or (axis, number).
_NUMBER_THEN_AXIS = re.compile(r"^(?:lm|lmk|landmark|point|pt)?[ _.\-]*(\d+)[ _.\-]*([xyz])$", re.I)
_AXIS_THEN_NUMBER = re.compile(r"^([xyz])[ _.\-]*(\d+)$", re.I)
_SEQUENTIAL = re.compile(r"^(?:v|var|coord)[ _.\-]*(\d+)$", re.I)
_DECIMAL_COMMA = re.compile(r"^[+\-]?\d+,\d+$")


@dataclass
class TabularOptions:
    """Explicit overrides; every None means "detect it"."""

    layout: str = LAYOUT_AUTO
    delimiter: str | None = None
    decimal: str | None = None
    n_dim: int | None = None
    id_col: str | None = None
    landmark_col: str | None = None
    coord_cols: Sequence[str] | None = None  # wide: in x, y, z, x, y, z order. long: x, y[, z].
    missing_codes: Sequence[str] = DEFAULT_MISSING_CODES
    apply_tps_scale: bool = True


@dataclass
class TabularImportResult(ImportBatchResult):
    # Original landmark name of LM0, LM1, ... - the canonical columns are positional, so this
    # is the only record of which point is which.
    landmark_names: list[str] = field(default_factory=list)
    detected: dict[str, dict[str, str]] = field(default_factory=dict)  # per file: what was detected
    source_format: str = ""


@dataclass
class _Parsed:
    """One file's specimens before the batch is combined."""

    specimens: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    landmark_names: list[str] = field(default_factory=list)
    n_dim: int | None = None
    source_format: str = ""
    detected: dict[str, str] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)

    def report(self, status: str, reason: str, action: str, spec_id: object = "") -> None:
        self.rows.append(
            {
                "Row": "",
                "ID": spec_id,
                "Status": status,
                "Reason": reason,
                "Missing_landmarks": "",
                "Suggested_action": action,
            }
        )


class _Missing:
    """Decides whether a cell is a missing-value code, and parses the rest."""

    def __init__(self, codes: Sequence[str], decimal: str = "."):
        self.decimal = decimal
        self.text = {str(c).strip().lower() for c in codes}
        self.values = set()
        for code in codes:
            try:
                self.values.add(float(code))
            except ValueError:
                pass

    def parse(self, cell: object) -> float:
        """A float, NaN for a missing code. Raises ValueError for anything else."""
        text = str(cell).strip()
        if text.lower() in self.text:
            return float("nan")
        if self.decimal != ".":
            text = text.replace(self.decimal, ".")
        value = float(text)
        if value in self.values or np.isinf(value):
            return float("nan")
        return value


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def sniff_delimiter(lines: Sequence[str]) -> str:
    """The delimiter giving the most fields, the same number on every sampled line."""
    sample = [line for line in lines[:50] if line.strip()]
    best, best_fields = None, 1
    for candidate in (",", "\t", ";", "|"):
        counts = {len(row) for row in csv.reader(sample, delimiter=candidate)}
        if len(counts) == 1 and next(iter(counts)) > best_fields:
            best, best_fields = candidate, next(iter(counts))
    if best is not None:
        return best
    counts = {len(line.split()) for line in sample}
    if len(counts) == 1 and next(iter(counts)) > 1:
        return " "
    raise ValueError(
        "Could not detect the delimiter: no separator gives the same number of "
        "columns on every line. Choose it explicitly."
    )


def _read_table(text: str, options: TabularOptions) -> tuple[pd.DataFrame, str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("The file holds no data rows.")
    delimiter = options.delimiter or sniff_delimiter(lines)
    if delimiter == " ":
        rows = [line.split() for line in lines]
    else:
        rows = list(csv.reader(lines, delimiter=delimiter))
    width = len(rows[0])
    header = [c.strip() for c in rows[0]]
    bad = [i + 2 for i, r in enumerate(rows[1:]) if len(r) != width]
    if bad:
        raise ValueError(f"{len(bad)} line(s) do not have {width} columns (first: line {bad[0]}).")
    if len(set(header)) != len(header):
        repeated = sorted({c for c in header if header.count(c) > 1})
        raise ValueError(f"Column name(s) appear more than once: {repeated[:5]}.")
    frame = pd.DataFrame(rows[1:], columns=header, dtype=object)

    decimal = options.decimal
    if decimal is None:
        # A comma decimal is only possible when the comma is not the delimiter - the normal
        # CSV of a Spanish or Swiss-German Excel. Judged on the cells, not the locale.
        decimal = "."
        if delimiter != ",":
            cells = frame.to_numpy().ravel()[:2000]
            if any(_DECIMAL_COMMA.match(str(c).strip()) for c in cells):
                decimal = ","
    return frame, delimiter, decimal


def _named_column(columns: Sequence[str], names: Sequence[str]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def _require_column(frame: pd.DataFrame, name: str, role: str) -> str:
    if name not in frame.columns:
        raise ValueError(
            f"No column named '{name}' for the {role}. Found: {list(frame.columns)[:12]}."
        )
    return name


def wide_coordinate_columns(
    columns: Sequence[str], n_dim: int | None
) -> tuple[list[str], list[str], int]:
    """Find a wide table's coordinate columns from their names.

    Returns (columns in x,y,z,x,y,z order, landmark names, n_dim).
    """
    named: dict[int, dict[str, str]] = {}
    sequential: list[tuple[int, str]] = []
    for column in columns:
        text = str(column).strip()
        match = _NUMBER_THEN_AXIS.match(text)
        if match:
            named.setdefault(int(match.group(1)), {})[match.group(2).lower()] = column
            continue
        match = _AXIS_THEN_NUMBER.match(text)
        if match:
            named.setdefault(int(match.group(2)), {})[match.group(1).lower()] = column
            continue
        match = _SEQUENTIAL.match(text)
        if match:
            sequential.append((int(match.group(1)), column))

    if named:
        axes_found = {axis for axes in named.values() for axis in axes}
        dim = n_dim or (3 if "z" in axes_found else 2)
        needed = _AXES[:dim]
        incomplete = sorted(i for i, axes in named.items() if not set(needed) <= set(axes))
        if incomplete:
            raise ValueError(f"Landmark(s) {incomplete[:8]} do not have all of {list(needed)}.")
        ordered = sorted(named)
        return (
            [named[i][axis] for i in ordered for axis in needed],
            [str(i) for i in ordered],
            dim,
        )

    if sequential:
        sequential.sort()
        numbers = [n for n, _ in sequential]
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise ValueError("The sequential coordinate columns have gaps in their numbering.")
        count = len(sequential)
        if n_dim is None:
            fits = [d for d in (2, 3) if count % d == 0]
            if len(fits) != 1:
                raise ValueError(
                    f"{count} sequential coordinate columns could be 2D or 3D "
                    "(or neither). Choose the dimensions explicitly."
                )
            n_dim = fits[0]
        if count % n_dim:
            raise ValueError(
                f"{count} sequential coordinate columns do not divide into {n_dim}D landmarks."
            )
        return ([c for _, c in sequential], [str(i + 1) for i in range(count // n_dim)], n_dim)

    raise ValueError(
        "Could not recognise coordinate columns. Expected names such as LM1_X, X1, 1.X "
        "or V1, V2, ... - or list them explicitly."
    )


def detect_layout(frame: pd.DataFrame) -> str:
    axes = _named_column(frame.columns, ("x",)), _named_column(frame.columns, ("y",))
    return LAYOUT_LONG if all(axes) else LAYOUT_WIDE


def _keep_metadata(parsed: _Parsed, spec_id: str, values: dict[str, object]) -> None:
    cleaned = {}
    for key, value in values.items():
        text = "" if value is None else str(value).strip()
        cleaned[key] = text if text else None
    parsed.metadata[spec_id] = cleaned


def _parse_long(
    frame: pd.DataFrame, options: TabularOptions, decimal: str, parsed: _Parsed
) -> None:
    id_col = (
        _require_column(frame, options.id_col, "specimen ID")
        if options.id_col
        else _named_column(frame.columns, _ID_NAMES)
    )
    lm_col = (
        _require_column(frame, options.landmark_col, "landmark name")
        if options.landmark_col
        else _named_column(frame.columns, _LANDMARK_NAMES)
    )
    if options.coord_cols:
        axes = [_require_column(frame, c, "coordinates") for c in options.coord_cols]
    else:
        axes = [c for c in (_named_column(frame.columns, (a,)) for a in _AXES) if c]
    if id_col is None or lm_col is None or len(axes) < 2:
        raise ValueError(
            f"A long table needs specimen, landmark and x/y[/z] columns. Found: "
            f"{list(frame.columns)[:12]}. Name them explicitly."
        )
    dim = options.n_dim or len(axes)
    axes = axes[:dim]
    extras = [c for c in frame.columns if c not in {id_col, lm_col, *axes}]
    missing = _Missing(options.missing_codes, decimal)

    frame = frame.assign(
        **{
            id_col: frame[id_col].astype(str).str.strip(),
            lm_col: frame[lm_col].astype(str).str.strip(),
        }
    )
    names = list(dict.fromkeys(frame[lm_col]))  # first-appearance order is the reference order
    position = {name: i for i, name in enumerate(names)}
    reordered, dropped_extras = 0, set()

    for spec_id, rows in frame.groupby(id_col, sort=False):
        if not spec_id:
            parsed.report(
                STATUS_FATAL,
                f"{len(rows)} row(s) have a blank specimen ID.",
                "Fill in the specimen ID.",
            )
            continue
        labels = rows[lm_col].tolist()
        if len(set(labels)) != len(labels):
            repeats = sorted({n for n in labels if labels.count(n) > 1})[:3]
            parsed.report(
                STATUS_FATAL,
                f"Specimen '{spec_id}' has landmark(s) listed twice: {repeats}.",
                "Remove the duplicated rows and re-import.",
                spec_id,
            )
            continue
        absent = [n for n in names if n not in set(labels)]
        if absent:
            parsed.report(
                STATUS_FATAL,
                f"Specimen '{spec_id}' has no row for {len(absent)} landmark(s) "
                f"(e.g. {absent[:3]}); excluded rather than padded. A landmark that was "
                "not observed should have a row with blank coordinates.",
                "Add the rows with blank coordinates, or fix the landmark names.",
                spec_id,
            )
            continue
        coords = np.full((len(names), dim), np.nan)
        try:
            for label, values in zip(labels, rows[axes].to_numpy()):
                coords[position[label]] = [missing.parse(v) for v in values]
        except ValueError as exc:
            parsed.report(
                STATUS_FATAL,
                f"Specimen '{spec_id}' has a non-numeric coordinate ({exc}).",
                "Fix the value, or add its code to the missing-value codes.",
                spec_id,
            )
            continue
        if labels != names:
            reordered += 1
        meta = {}
        for extra in extras:
            values = rows[extra].astype(str).str.strip().unique()
            if len(values) == 1:
                meta[extra] = values[0]
            else:
                dropped_extras.add(extra)
        _keep_metadata(parsed, spec_id, meta)
        parsed.specimens[spec_id] = coords

    if reordered:
        parsed.report(
            STATUS_INFO,
            f"{reordered} specimen(s) list their landmarks in a different "
            "order; they were matched by landmark name.",
            "None needed.",
        )
    if dropped_extras:
        parsed.report(
            STATUS_WARNING,
            f"Column(s) {sorted(dropped_extras)} vary within a specimen, so they describe "
            "landmarks rather than specimens and were not kept as metadata.",
            "Keep the source file if you need them.",
        )
    for spec_id in parsed.metadata:
        for extra in dropped_extras:
            parsed.metadata[spec_id].pop(extra, None)
    parsed.landmark_names = names
    parsed.n_dim = dim
    parsed.source_format = FORMAT_LONG
    parsed.detected.update(id_col=id_col, landmark_col=lm_col, coord_cols=", ".join(axes))


def _parse_wide(
    frame: pd.DataFrame, options: TabularOptions, decimal: str, parsed: _Parsed
) -> None:
    if options.coord_cols:
        coord_cols = [_require_column(frame, c, "coordinates") for c in options.coord_cols]
        if options.n_dim is None:
            raise ValueError(
                "Listing coordinate columns explicitly also needs the number of dimensions."
            )
        dim = options.n_dim
        if len(coord_cols) % dim:
            raise ValueError(
                f"{len(coord_cols)} coordinate columns do not divide into {dim}D landmarks."
            )
        names = [str(i + 1) for i in range(len(coord_cols) // dim)]
    else:
        coord_cols, names, dim = wide_coordinate_columns(list(frame.columns), options.n_dim)

    id_col = (
        _require_column(frame, options.id_col, "specimen ID")
        if options.id_col
        else _named_column(frame.columns, _ID_NAMES)
    )
    if id_col in coord_cols:
        id_col = None
    if id_col is None:
        others = [c for c in frame.columns if c not in set(coord_cols)]
        if not others:
            raise ValueError("No column is left to identify the specimens. Add an ID column.")
        id_col = others[0]
        parsed.report(
            STATUS_INFO,
            f"No column is named like a specimen ID; used the first "
            f"non-coordinate column, '{id_col}'.",
            "Check this is the specimen ID.",
        )

    extras = [c for c in frame.columns if c != id_col and c not in set(coord_cols)]
    missing = _Missing(options.missing_codes, decimal)
    seen: dict[str, int] = {}
    for row_number, row in enumerate(frame.to_dict("records"), start=2):
        spec_id = str(row[id_col]).strip()
        if not spec_id:
            parsed.report(
                STATUS_FATAL,
                f"Line {row_number} has a blank specimen ID.",
                "Fill in the specimen ID.",
            )
            continue
        seen[spec_id] = seen.get(spec_id, 0) + 1
        if seen[spec_id] > 1:
            parsed.specimens.pop(spec_id, None)
            parsed.metadata.pop(spec_id, None)
            if seen[spec_id] == 2:
                parsed.report(
                    STATUS_FATAL,
                    f"Specimen ID '{spec_id}' appears more than once in this file - all "
                    "occurrences excluded. They may be repeats or different specimens "
                    "sharing a name, and the file cannot say which.",
                    "Give each specimen its own ID and re-import.",
                    spec_id,
                )
            continue
        try:
            values = [missing.parse(row[c]) for c in coord_cols]
        except ValueError as exc:
            parsed.report(
                STATUS_FATAL,
                f"Specimen '{spec_id}' has a non-numeric coordinate ({exc}).",
                "Fix the value, or add its code to the missing-value codes.",
                spec_id,
            )
            continue
        parsed.specimens[spec_id] = np.array(values, dtype=float).reshape(-1, dim)
        _keep_metadata(parsed, spec_id, {c: row[c] for c in extras})

    parsed.landmark_names = names
    parsed.n_dim = dim
    parsed.source_format = FORMAT_WIDE
    parsed.detected.update(
        id_col=str(id_col), coord_cols=f"{coord_cols[0]} ... {coord_cols[-1]} ({len(coord_cols)})"
    )


_TPS_LM = re.compile(r"^LM(\d?)\s*=\s*(\d+)\s*$", re.I)
_TPS_KEY = re.compile(r"^([A-Za-z_]+)\s*=\s*(.*)$")


def _parse_tps(text: str, options: TabularOptions, stem: str, parsed: _Parsed) -> None:
    """A TPS file is a run of blocks: LM=n, n coordinate rows, then KEY=value lines that belong to it."""
    missing = _Missing(options.missing_codes, options.decimal or ".")
    blocks: list[dict] = []
    lines = [line.strip() for line in text.splitlines()]
    skipped_keys = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line:
            continue
        lm = _TPS_LM.match(line)
        if lm:
            block = {
                "number": len(blocks) + 1,
                "dim": int(lm.group(1)) if lm.group(1) else None,
                "coords": [],
                "error": None,
                "keys": {},
            }
            for _ in range(int(lm.group(2))):
                if i >= len(lines) or _TPS_KEY.match(lines[i]) or not lines[i]:
                    block["error"] = (
                        f"has only {len(block['coords'])} of {lm.group(2)} landmark rows"
                    )
                    break
                block["coords"].append(lines[i].split())
                i += 1
            blocks.append(block)
            continue
        key = _TPS_KEY.match(line)
        if key and blocks:
            name = key.group(1).upper()
            blocks[-1]["keys"][name] = key.group(2).strip()
            if name not in ("ID", "IMAGE", "SCALE", "COMMENT"):
                skipped_keys.add(name)
        # Rows outside an LM block (CURVES= / POINTS= outlines) are not landmarks.

    if not blocks:
        raise ValueError("No LM= or LM3= block found.")

    dims = {b["dim"] or len(b["coords"][0]) for b in blocks if b["coords"]}
    dim = options.n_dim or (dims.pop() if len(dims) == 1 else None)
    if dim is None or dim not in (2, 3):
        raise ValueError(
            f"The blocks in this TPS file have inconsistent dimensions: {sorted(dims)}."
        )

    used_scale, seen = 0, {}
    for block in blocks:
        keys = block["keys"]
        spec_id = keys.get("ID") or keys.get("IMAGE")
        if not spec_id:
            spec_id = f"{stem}_{block['number']}"
            parsed.detected["unnamed_blocks"] = "named <file>_<block number>"
        seen[spec_id] = seen.get(spec_id, 0) + 1
        if seen[spec_id] > 1:
            parsed.specimens.pop(spec_id, None)
            parsed.metadata.pop(spec_id, None)
            if seen[spec_id] == 2:
                parsed.report(
                    STATUS_FATAL,
                    f"Specimen ID '{spec_id}' appears more than once in this TPS file - "
                    "all occurrences excluded.",
                    "Give each block its own ID.",
                    spec_id,
                )
            continue
        if block["error"]:
            parsed.report(
                STATUS_FATAL,
                f"Block {block['number']} ('{spec_id}') {block['error']}.",
                "Fix the block and re-import.",
                spec_id,
            )
            continue
        try:
            rows = []
            for row in block["coords"]:
                if len(row) != dim:
                    raise ValueError(f"a row has {len(row)} values, expected {dim}")
                rows.append([missing.parse(v) for v in row])
            coords = np.array(rows, dtype=float).reshape(-1, dim)
            meta: dict[str, object] = {}
            if keys.get("IMAGE") and keys.get("IMAGE") != spec_id:
                meta["Image"] = keys["IMAGE"]
            if keys.get("COMMENT"):
                meta["Comment"] = keys["COMMENT"]
            if keys.get("SCALE"):
                scale = float(keys["SCALE"])
                if scale <= 0:
                    raise ValueError(f"SCALE={keys['SCALE']} is not positive")
                meta["Scale"] = keys["SCALE"]
                if options.apply_tps_scale:
                    coords = coords * scale
                    used_scale += 1
        except ValueError as exc:
            parsed.report(
                STATUS_FATAL,
                f"Block {block['number']} ('{spec_id}'): {exc}.",
                "Fix the block and re-import.",
                spec_id,
            )
            continue
        parsed.specimens[spec_id] = coords
        _keep_metadata(parsed, spec_id, meta)

    n = len(parsed.specimens)
    if used_scale:
        parsed.report(
            STATUS_INFO,
            f"Coordinates of {used_scale} specimen(s) were multiplied by their "
            "SCALE= value (the TPS convention for pixels to real units).",
            "None needed.",
        )
        if used_scale != n:
            parsed.report(
                STATUS_WARNING,
                f"Only {used_scale} of {n} specimens have a SCALE=, so the "
                "rest stay in their original units: sizes are not comparable.",
                "Check the source file before using centroid size.",
            )
    if skipped_keys:
        parsed.report(
            STATUS_INFO, f"TPS field(s) {sorted(skipped_keys)} were not imported.", "None needed."
        )
    counts = {b["number"]: len(b["coords"]) for b in blocks}
    n_landmarks = max(counts.values(), key=list(counts.values()).count)
    parsed.landmark_names = [str(k + 1) for k in range(n_landmarks)]
    parsed.n_dim = dim
    parsed.source_format = FORMAT_TPS


def looks_like_tps(path: Path, text: str) -> bool:
    if path.suffix.lower() == ".tps":
        return True
    return any(_TPS_LM.match(line.strip()) for line in text.splitlines()[:5])


def _parse_file(path: Path, options: TabularOptions) -> _Parsed:
    parsed = _Parsed()
    text = _read_text(path)
    layout = options.layout
    if layout == LAYOUT_AUTO and looks_like_tps(path, text):
        layout = LAYOUT_TPS
    if layout == LAYOUT_TPS:
        _parse_tps(text, options, path.stem, parsed)
    else:
        frame, delimiter, decimal = _read_table(text, options)
        if layout == LAYOUT_AUTO:
            layout = detect_layout(frame)
        parsed.detected.update(delimiter=repr(delimiter), decimal=decimal)
        if layout == LAYOUT_LONG:
            _parse_long(frame, options, decimal, parsed)
        else:
            _parse_wide(frame, options, decimal, parsed)
    parsed.detected["layout"] = layout
    return parsed


def import_tabular_files(
    paths: Sequence[str | Path], options: TabularOptions | None = None
) -> TabularImportResult:
    options = options or TabularOptions()
    specimens: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, object]] = {}
    source_file_by_id: dict[str, str] = {}
    source_format_by_id: dict[str, str] = {}
    rows: list[dict] = []
    detected: dict[str, dict[str, str]] = {}
    batch_dim: int | None = None
    names: list[str] = []

    for raw_path in paths:
        path = Path(raw_path)
        try:
            parsed = _parse_file(path, options)
        except (OSError, ValueError) as exc:
            rows.append(
                {
                    "Row": "",
                    "ID": path.stem,
                    "Status": STATUS_FATAL,
                    "Reason": f"{path.name}: {exc}",
                    "Missing_landmarks": "",
                    "Suggested_action": "Fix the file, or set the layout and columns explicitly.",
                }
            )
            continue
        rows.extend(parsed.rows)
        detected[path.name] = parsed.detected
        if not parsed.specimens:
            continue
        if batch_dim is None:
            batch_dim = parsed.n_dim
        elif parsed.n_dim != batch_dim:
            rows.append(
                {
                    "Row": "",
                    "ID": path.stem,
                    "Status": STATUS_FATAL,
                    "Reason": f"{path.name} holds {parsed.n_dim}D landmarks; the rest of this "
                    f"batch is {batch_dim}D.",
                    "Missing_landmarks": "",
                    "Suggested_action": "Import different dimensionalities separately.",
                }
            )
            continue
        if not names:
            names = parsed.landmark_names
        elif parsed.landmark_names != names and len(parsed.landmark_names) == len(names):
            rows.append(
                {
                    "Row": "",
                    "ID": path.stem,
                    "Status": STATUS_FATAL,
                    "Reason": f"{path.name} uses different landmark names from the rest of "
                    "this batch, so its landmarks cannot be assumed to correspond.",
                    "Missing_landmarks": "",
                    "Suggested_action": "Import it separately, or make the names agree.",
                }
            )
            continue
        for spec_id, coords in parsed.specimens.items():
            if spec_id in specimens:
                rows.append(
                    {
                        "Row": "",
                        "ID": spec_id,
                        "Status": STATUS_FATAL,
                        "Reason": f"Specimen ID '{spec_id}' appears in more than one file; "
                        f"the copy in {path.name} was excluded.",
                        "Missing_landmarks": "",
                        "Suggested_action": "Rename one of them so IDs are unique.",
                    }
                )
                continue
            specimens[spec_id] = coords
            metadata[spec_id] = parsed.metadata.get(spec_id, {})
            source_file_by_id[spec_id] = path.name
            source_format_by_id[spec_id] = parsed.source_format

    n_dim = batch_dim or (options.n_dim or 3)
    formats = sorted(set(source_format_by_id.values()))
    source_format = formats[0] if len(formats) == 1 else " + ".join(formats)
    frame, landmark_cols, validation = build_canonical_from_specimens(
        specimens, source_format, source_file_by_id, n_dim=n_dim, extra_validation_rows=rows
    )

    if not frame.empty:
        frame[SOURCE_FORMAT_COL] = frame[ID_COL].map(source_format_by_id)
        meta_names = list(dict.fromkeys(k for meta in metadata.values() for k in meta))
        clashes = [c for c in meta_names if c in RESERVED_COLS or c in set(landmark_cols)]
        if clashes:
            validation = _with_note(
                validation,
                STATUS_WARNING,
                f"Metadata column(s) {clashes} clash with reserved names and were not kept.",
                "Rename them in the source file.",
            )
            meta_names = [c for c in meta_names if c not in clashes]
        added = pd.DataFrame(
            {c: _coerce_column([metadata[i].get(c) for i in frame[ID_COL]]) for c in meta_names},
            index=frame.index,
        )
        frame = pd.concat([frame[[ID_COL]], added, frame.drop(columns=[ID_COL])], axis=1)
        if "Sex" in meta_names:  # the shared builder validated before metadata existed
            validation = combine_validation_tables([validation, check_sex_codes(frame)])

    n_landmarks = len(landmark_cols) // n_dim if landmark_cols else 0
    return TabularImportResult(
        raw=frame,
        landmark_cols=landmark_cols,
        n_landmarks=n_landmarks,
        n_dim=n_dim,
        validation=validation,
        source_files=sorted(set(source_file_by_id.values())),
        landmark_names=list(names) if len(names) == n_landmarks else [],
        detected=detected,
        source_format=source_format,
    )


def _with_note(validation: pd.DataFrame, status: str, reason: str, action: str) -> pd.DataFrame:
    note = pd.DataFrame(
        [
            {
                "Row": "",
                "ID": "",
                "Status": status,
                "Reason": reason,
                "Missing_landmarks": "",
                "Suggested_action": action,
            }
        ],
        columns=VALIDATION_COLUMNS,
    )
    return combine_validation_tables([validation, note])


def _coerce_column(values: list[object]) -> list[object]:
    """A column of numbers stays numeric, so a Length can be used as one. Decided per column and
    never for leading zeros: a museum number that is only sometimes numeric, or 0071, stays text."""
    present = [str(v) for v in values if v is not None]
    try:
        if any(re.match(r"^[+\-]?0\d", v) for v in present):
            return values
        [float(v) for v in present]
    except ValueError:
        return values
    return [float(v) if v is not None else None for v in values]


def landmark_names_frame(names: Sequence[str]) -> pd.DataFrame:
    """Canonical LM index -> the name the source file used, for a sheet or a table."""
    return pd.DataFrame(
        {"Landmark": [f"LM{i}" for i in range(len(names))], "Source_name": list(names)}
    )


__all__ = [
    "DEFAULT_MISSING_CODES",
    "FORMAT_LONG",
    "FORMAT_TPS",
    "FORMAT_WIDE",
    "LAYOUTS",
    "LAYOUT_AUTO",
    "LAYOUT_LONG",
    "LAYOUT_TPS",
    "LAYOUT_WIDE",
    "TabularImportResult",
    "TabularOptions",
    "import_tabular_files",
    "landmark_names_frame",
    "sniff_delimiter",
    "wide_coordinate_columns",
]
