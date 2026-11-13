"""Morphologika ``.txt`` importer.

One file may hold many specimens. Reads the header sections and both layouts
found in real files:

* **Named blocks**: each specimen in ``[rawpoints]`` starts with an
  apostrophe-prefixed line. A malformed block is flagged and excluded on its
  own.
* **Standard**: a ``[names]`` section and bare numbers in ``[rawpoints]``, as
  read by geomorph and EVAN. Numbers are paired with names by position, so a
  file whose block count differs from its name count is refused whole;
  pairing it anyway would silently mislabel every specimen after the
  discrepancy.

``[labels]`` with ``[labelvalues]`` become metadata columns, and
``[wireframe]`` becomes wireframe chains. Landmarks are renumbered to the
canonical 0-indexed ``LM0_X, LM1_X, ...`` convention, because the format
identifies landmarks by position only.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from gmm_core.io.common import ImportBatchResult, build_canonical_from_specimens
from gmm_core.validation import STATUS_FATAL, STATUS_WARNING

SOURCE_FORMAT_NAME = "Morphologika"

_NUM_RE = re.compile(r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?")
_SECTION_RE = re.compile(r"^\s*\[([A-Za-z]+)\]\s*(.*)$")
_PLACEHOLDER_RE = re.compile(r"^#?\s*\d+$")


def list_morphologika_files_recursive(folder: str | Path) -> list[Path]:
    return sorted(Path(folder).rglob("*.txt"))


def _sections(text: str) -> dict[str, list[str]]:
    """Section name (lower case) -> its lines. A value on the header line itself counts."""
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = _SECTION_RE.match(line)
        if match:
            current = match.group(1).lower()
            out.setdefault(current, [])
            if match.group(2).strip():
                out[current].append(match.group(2).strip())
        elif current is not None:
            out[current].append(line)
    return out


def _first_int(lines: list[str] | None) -> int | None:
    for line in lines or []:
        if line.strip():
            try:
                return int(line.split()[0])
            except ValueError:
                return None
    return None


def _clean(lines: list[str] | None) -> list[str]:
    return [line.strip() for line in (lines or []) if line.strip()]


class _FileResult:
    def __init__(self) -> None:
        self.specimens: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.n_dim: int | None = None
        self.issues: list[str] = []
        self.warnings: list[str] = []
        self.wireframe: list[list[int]] = []


def _parse_named_blocks(
    lines: list[str], n_landmarks: int, n_dim: int, names: list[str], name: str, out: _FileResult
) -> list[str]:
    """Blocks that begin with an apostrophe line. Returns the block names in file order."""
    order: list[str] = []
    n_blocks = 0
    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip().startswith("'"):
            i += 1
        if i >= len(lines):
            break
        header = lines[i].strip()[1:].strip()
        # '#1 or '1 is a counter, not a name: the real one is in [names].
        specimen = (
            names[n_blocks] if (_PLACEHOLDER_RE.match(header) and n_blocks < len(names)) else header
        )
        n_blocks += 1
        i += 1
        coords: list[list[float]] = []
        ok = True
        j = 0
        while j < n_landmarks:
            if i >= len(lines):
                out.issues.append(
                    f"{name}: specimen '{specimen}' ended early ({j}/{n_landmarks} landmarks read)."
                )
                ok = False
                break
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.startswith("'"):
                out.issues.append(
                    f"{name}: specimen '{specimen}' has another specimen starting before "
                    f"finishing ({j}/{n_landmarks} landmarks read)."
                )
                ok = False
                break
            numbers = _NUM_RE.findall(line)
            if len(numbers) < n_dim:
                out.issues.append(
                    f"{name}: specimen '{specimen}' has a malformed coordinate line: {line!r}."
                )
                ok = False
                i += 1
                break
            coords.append([float(v) for v in numbers[:n_dim]])
            j += 1
            i += 1
        order.append(specimen)
        if not ok:
            continue
        if specimen in out.specimens:
            out.issues.append(
                f"{name}: duplicate specimen name '{specimen}' within this file - both occurrences excluded."
            )
            del out.specimens[specimen]
            continue
        out.specimens[specimen] = np.array(coords, dtype=float)
    return order


def _parse_bare(
    lines: list[str],
    n_landmarks: int,
    n_dim: int,
    names: list[str],
    stem: str,
    name: str,
    out: _FileResult,
) -> list[str]:
    """Bare numbers. Paired with names strictly by position, so counts must agree."""
    values: list[float] = []
    for number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        tokens = text.split()
        try:
            values.extend(float(t) for t in tokens)
        except ValueError:
            out.issues.append(
                f"{name}: [rawpoints] line {number} is not numeric: {text!r}. The file was excluded."
            )
            return []
    per_block = n_landmarks * n_dim
    if not values or len(values) % per_block:
        out.issues.append(
            f"{name}: [rawpoints] holds {len(values)} numbers, which is not a whole number of "
            f"{n_landmarks} x {n_dim} configurations. The file was excluded."
        )
        return []
    n_blocks = len(values) // per_block
    if names and len(names) != n_blocks:
        out.issues.append(
            f"{name}: {len(names)} names but {n_blocks} coordinate blocks. Names are paired with "
            "blocks by position, so every specimen after the discrepancy would get the wrong "
            "name. The file was excluded."
        )
        return []
    if not names:
        names = [f"{stem}_{i + 1}" for i in range(n_blocks)]
        out.warnings.append(
            f"{name}: no [names] section, so specimens were named {names[0]} and so on."
        )
    blocks = np.array(values).reshape(n_blocks, n_landmarks, n_dim)
    seen = {}
    for i, specimen in enumerate(names):
        seen.setdefault(specimen, []).append(i)
    for specimen, where in seen.items():
        if len(where) > 1:
            out.issues.append(
                f"{name}: duplicate specimen name '{specimen}' within this file - all "
                f"{len(where)} occurrences excluded."
            )
            continue
        out.specimens[specimen] = blocks[where[0]]
    return list(names)


def _read_labels(
    sections: dict[str, list[str]], order: list[str], name: str, out: _FileResult
) -> None:
    labels = " ".join(_clean(sections.get("labels"))).split()
    rows = _clean(sections.get("labelvalues"))
    if not labels or not rows:
        return
    if len(rows) != len(order):
        out.warnings.append(
            f"{name}: {len(rows)} [labelvalues] rows for {len(order)} specimens, so they were "
            "not used. Pairing them would misassign metadata."
        )
        return
    bad = 0
    for specimen, row in zip(order, rows):
        values = row.split()
        if len(values) != len(labels):
            bad += 1
            continue
        if specimen in out.specimens:
            out.metadata[specimen] = dict(zip(labels, values))
    if bad:
        out.warnings.append(
            f"{name}: {bad} [labelvalues] rows do not have one value per label "
            f"({', '.join(labels)}), and those specimens have blank metadata."
        )


def _read_wireframe(sections: dict[str, list[str]], n_landmarks: int, out: _FileResult) -> None:
    for row in _clean(sections.get("wireframe")):
        parts = row.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            a, b = int(parts[0]) - 1, int(parts[1]) - 1  # 1-based in the file
            if 0 <= a < n_landmarks and 0 <= b < n_landmarks and a != b:
                out.wireframe.append([a, b])


def _parse_one_file(path: Path) -> _FileResult:
    out = _FileResult()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        out.issues.append(f"{path.name}: could not read file ({exc}).")
        return out

    sections = _sections(text)
    n_landmarks = _first_int(sections.get("landmarks"))
    if n_landmarks is None or n_landmarks < 1:
        out.issues.append(f"{path.name}: no [landmarks] section found.")
        return out
    n_dim = _first_int(sections.get("dimensions")) or 3
    if n_dim not in (2, 3):
        out.issues.append(f"{path.name}: unsupported [dimensions] value {n_dim} (expected 2 or 3).")
        return out
    out.n_dim = n_dim
    if "rawpoints" not in sections:
        out.issues.append(f"{path.name}: no [rawpoints] section found.")
        return out

    names = [line.lstrip("'").strip() for line in _clean(sections.get("names"))]
    lines = sections["rawpoints"]
    if any(line.strip().startswith("'") for line in lines):
        order = _parse_named_blocks(lines, n_landmarks, n_dim, names, path.name, out)
    else:
        order = _parse_bare(lines, n_landmarks, n_dim, names, path.stem, path.name, out)

    # A header count that disagrees with the blocks present is how a truncated file looks:
    # an interrupted copy loses whole specimens off the end and nothing else says so.
    declared = _first_int(sections.get("individuals"))
    if declared is not None and order and declared != len(order):
        out.warnings.append(
            f"{path.name}: the header declares {declared} individuals but "
            f"{len(order)} specimen block(s) were found - the file may be truncated."
        )
    if order:
        _read_labels(sections, order, path.name, out)
        _read_wireframe(sections, n_landmarks, out)
    return out


def import_morphologika_files(paths: Sequence[str | Path]) -> ImportBatchResult:
    specimens: dict[str, np.ndarray] = {}
    source_file_by_id: dict[str, str] = {}
    metadata: dict[str, dict[str, str]] = {}
    wireframe: list[list[int]] = []
    extra_rows: list[dict] = []
    batch_n_dim: int | None = None

    for raw_path in paths:
        path = Path(raw_path)
        parsed = _parse_one_file(path)
        for issue in parsed.issues:
            extra_rows.append(
                {
                    "Row": "",
                    "ID": path.stem,
                    "Status": STATUS_FATAL,
                    "Reason": issue,
                    "Missing_landmarks": "",
                    "Suggested_action": "Fix the affected specimen block in the source file and re-import.",
                }
            )
        for warning in parsed.warnings:
            extra_rows.append(
                {
                    "Row": "",
                    "ID": path.stem,
                    "Status": STATUS_WARNING,
                    "Reason": warning,
                    "Missing_landmarks": "",
                    "Suggested_action": "Compare with the original file; re-export it if specimens are missing.",
                }
            )

        if not parsed.specimens:
            continue

        if batch_n_dim is None:
            batch_n_dim = parsed.n_dim
        elif parsed.n_dim != batch_n_dim:
            extra_rows.append(
                {
                    "Row": "",
                    "ID": path.stem,
                    "Status": STATUS_FATAL,
                    "Reason": (
                        f"{path.name} declares [dimensions] {parsed.n_dim}, which does not match "
                        f"the rest of this import batch ({batch_n_dim}D)."
                    ),
                    "Missing_landmarks": "",
                    "Suggested_action": "Import files with a different dimensionality in a separate batch.",
                }
            )
            continue

        for name, coords in parsed.specimens.items():
            if name in specimens:
                extra_rows.append(
                    {
                        "Row": "",
                        "ID": name,
                        "Status": STATUS_FATAL,
                        "Reason": f"Specimen name '{name}' appears in more than one Morphologika file in this batch.",
                        "Missing_landmarks": "",
                        "Suggested_action": "Rename one of the specimens so IDs are unique, then re-import.",
                    }
                )
                continue
            specimens[name] = coords
            source_file_by_id[name] = path.name
            if name in parsed.metadata:
                metadata[name] = parsed.metadata[name]
        if parsed.wireframe and not wireframe:
            wireframe = parsed.wireframe

    n_dim = batch_n_dim or 3
    df, landmark_cols, validation = build_canonical_from_specimens(
        specimens,
        SOURCE_FORMAT_NAME,
        source_file_by_id,
        n_dim=n_dim,
        extra_validation_rows=extra_rows,
    )

    if metadata and not df.empty:
        columns = list(dict.fromkeys(label for values in metadata.values() for label in values))
        for offset, column in enumerate(c for c in columns if c not in df.columns):
            df.insert(1 + offset, column, [metadata.get(i, {}).get(column) for i in df["ID"]])

    return ImportBatchResult(
        raw=df,
        landmark_cols=landmark_cols,
        n_landmarks=len(landmark_cols) // n_dim if landmark_cols else 0,
        n_dim=n_dim,
        validation=validation,
        source_files=sorted(set(source_file_by_id.values())),
        wireframe_chains=wireframe,
    )
