import numpy as np
import pytest

from gmm_core.io.slicer_fcsv import (
    import_fcsv_files,
    lps_to_ras,
    read_fcsv,
)
from gmm_core.validation import STATUS_FATAL

HEADER = (
    "# Markups fiducial file version = 4.11\n"
    "# CoordinateSystem = {system}\n"
    "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
)


def _write(path, coords, system="LPS", labels=None):
    lines = [HEADER.format(system=system)]
    for i, (x, y, z) in enumerate(coords, start=1):
        label = labels[i - 1] if labels else f"LM_{i}"
        lines.append(f"FID_{i},{x},{y},{z},0,0,0,1,1,1,0,{label},,\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_read_fcsv_returns_coordinates_labels_and_declared_system(tmp_path):
    coords = [(1.0, 2.0, 3.0), (-4.5, 5.5, 6.5)]
    path = _write(tmp_path / "spec.fcsv", coords, system="LPS")

    points, labels, system = read_fcsv(path)
    assert np.allclose(points, coords)
    assert labels == ["LM_1", "LM_2"]
    assert system == "LPS"


def test_ras_files_are_recognised_too(tmp_path):
    path = _write(tmp_path / "spec.fcsv", [(0.0, 0.0, 0.0)], system="RAS")
    assert read_fcsv(path)[2] == "RAS"


def test_lps_to_ras_negates_x_and_y_and_is_its_own_inverse():
    coords = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    converted = lps_to_ras(coords)
    assert np.allclose(converted, [[-1.0, -2.0, 3.0], [4.0, -5.0, -6.0]])
    assert np.allclose(lps_to_ras(converted), coords)


def test_conversion_does_not_mutate_the_input():
    coords = np.array([[1.0, 2.0, 3.0]])
    original = coords.copy()
    lps_to_ras(coords)
    assert np.allclose(coords, original)


def test_import_reaches_the_canonical_schema(tmp_path):
    _write(tmp_path / "a.fcsv", [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
    _write(tmp_path / "b.fcsv", [(2.0, 0.0, 0.0), (3.0, 1.0, 1.0)])

    result = import_fcsv_files(sorted(tmp_path.glob("*.fcsv")))
    assert result.n_landmarks == 2
    assert result.n_dim == 3
    assert sorted(result.raw["ID"].tolist()) == ["a", "b"]
    assert "LM0_X" in result.landmark_cols


def test_a_malformed_line_excludes_the_whole_file_and_says_why(tmp_path):
    good = _write(tmp_path / "good.fcsv", [(0.0, 0.0, 0.0)])
    bad = tmp_path / "bad.fcsv"
    bad.write_text(
        HEADER.format(system="LPS") + "FID_1,notanumber,2,3,0,0,0,1,1,1,0,LM_1,,\n",
        encoding="utf-8",
    )

    result = import_fcsv_files([good, bad])
    assert result.raw["ID"].tolist() == ["good"]
    fatal = result.validation[result.validation["Status"] == STATUS_FATAL]
    assert "bad" in fatal["ID"].tolist()
    assert any("non-numeric" in reason for reason in fatal["Reason"])


def test_a_truncated_line_is_reported_rather_than_guessed(tmp_path):
    path = tmp_path / "short.fcsv"
    path.write_text(HEADER.format(system="LPS") + "FID_1,1.0,2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected at least"):
        read_fcsv(path)


def test_an_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.fcsv"
    path.write_text(HEADER.format(system="LPS"), encoding="utf-8")
    with pytest.raises(ValueError, match="no fiducial points"):
        read_fcsv(path)


def test_mixing_coordinate_systems_in_one_batch_is_flagged(tmp_path):
    """LPS and RAS differ by a reflection, so combining them would mirror half
    the sample - and GPA would absorb that into a rotation without complaining."""
    _write(tmp_path / "lps.fcsv", [(1.0, 2.0, 3.0)], system="LPS")
    _write(tmp_path / "ras.fcsv", [(1.0, 2.0, 3.0)], system="RAS")

    result = import_fcsv_files(sorted(tmp_path.glob("*.fcsv")))
    reasons = " ".join(result.validation["Reason"].astype(str))
    assert "mixes coordinate systems" in reasons


def test_a_consistent_batch_is_not_flagged(tmp_path):
    _write(tmp_path / "a.fcsv", [(1.0, 2.0, 3.0)], system="LPS")
    _write(tmp_path / "b.fcsv", [(4.0, 5.0, 6.0)], system="LPS")

    result = import_fcsv_files(sorted(tmp_path.glob("*.fcsv")))
    reasons = " ".join(result.validation["Reason"].astype(str))
    assert "mixes coordinate systems" not in reasons


def test_to_ras_converts_only_the_lps_files(tmp_path):
    _write(tmp_path / "lps.fcsv", [(1.0, 2.0, 3.0)], system="LPS")
    result = import_fcsv_files([tmp_path / "lps.fcsv"], to_ras=True)
    row = result.raw.iloc[0]
    assert row["LM0_X"] == pytest.approx(-1.0)
    assert row["LM0_Y"] == pytest.approx(-2.0)
    assert row["LM0_Z"] == pytest.approx(3.0)


def test_labels_fall_back_when_the_column_is_absent(tmp_path):
    path = tmp_path / "nolabel.fcsv"
    path.write_text(HEADER.format(system="LPS") + "FID_1,1.0,2.0,3.0\n", encoding="utf-8")
    _points, labels, _system = read_fcsv(path)
    assert labels == ["LM_1"]
