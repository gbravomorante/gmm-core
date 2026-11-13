import numpy as np
import pytest

from gmm_core.io.amira_landmarkascii import (
    MM_PER_UNIT,
    _parse_one_file,
    file_unit,
    import_amira_files,
)
from gmm_core.validation import STATUS_FATAL, STATUS_INFO, STATUS_WARNING

VALID_CONTENT = """# Amira Landmark ASCII
@1
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0 9.0
@2
"""


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_import_valid_files(tmp_path):
    p1 = _write(tmp_path, "spec_a.landmarkAscii", VALID_CONTENT)
    p2 = _write(tmp_path, "spec_b.landmarkAscii", VALID_CONTENT)
    result = import_amira_files([p1, p2])
    assert len(result.raw) == 2
    assert result.n_landmarks == 3
    assert result.n_dim == 3
    assert set(result.raw["ID"]) == {"spec_a", "spec_b"}
    assert (result.raw["SourceFormat"] == "Amira_landmarkAscii").all()


def test_missing_at1_marker_flagged(tmp_path):
    p1 = _write(tmp_path, "bad.landmarkAscii", "no markers here\n1 2 3\n")
    result = import_amira_files([p1])
    assert result.raw.empty
    assert (result.validation["Status"] == STATUS_FATAL).any()
    assert "@1" in " ".join(result.validation["Reason"])


def test_malformed_line_excludes_whole_file_not_padded(tmp_path):
    content = "@1\n1.0 2.0 3.0\nbad line here\n4.0 5.0 6.0\n@2\n"
    p1 = _write(tmp_path, "malformed.landmarkAscii", content)
    p2 = _write(tmp_path, "good.landmarkAscii", VALID_CONTENT)
    result = import_amira_files([p1, p2])
    # The malformed file is excluded entirely - never padded with NaN.
    assert "malformed" not in set(result.raw["ID"])
    assert "good" in set(result.raw["ID"])
    flagged_ids = set(result.validation["ID"])
    assert "malformed" in flagged_ids
    assert (
        result.validation.loc[result.validation["ID"] == "malformed", "Status"] == STATUS_FATAL
    ).all()


def test_landmark_count_mismatch_flagged_and_excluded(tmp_path):
    short_content = "@1\n1.0 2.0 3.0\n4.0 5.0 6.0\n@2\n"
    p1 = _write(tmp_path, "three_lm_a.landmarkAscii", VALID_CONTENT)
    p2 = _write(tmp_path, "three_lm_b.landmarkAscii", VALID_CONTENT)
    p3 = _write(tmp_path, "two_lm.landmarkAscii", short_content)
    result = import_amira_files([p1, p2, p3])
    assert result.n_landmarks == 3
    assert set(result.raw["ID"]) == {"three_lm_a", "three_lm_b"}
    assert "two_lm" in set(result.validation["ID"])


def test_no_usable_files_returns_empty_raw_with_validation(tmp_path):
    p1 = _write(tmp_path, "bad.landmarkAscii", "garbage\n")
    result = import_amira_files([p1])
    assert result.raw.empty
    assert not result.validation.empty


# --- declared coordinate units -----------------------------------------------------
# Some real Avizo exports declare NANOMETRES. Reading them
# as millimetres leaves GPA unharmed (scale is removed) but corrupts centroid
# size, every derived distance, and allometry - by a factor of a million.


UNIT_TEMPLATE = """# Avizo 3D ASCII 3.0

define Markers 2

Parameters {{
    Units {{
        Coordinates "{unit}"
    }}
    NumSets 1,
    ContentType "LandmarkSet"
}}

Markers {{ float[3] Coordinates }} @1

# Data section follows
@1
{a}
{b}
@2
"""


def _unit_file(tmp_path, name, unit, a, b):
    return _write(tmp_path, name, UNIT_TEMPLATE.format(unit=unit, a=a, b=b))


def test_nanometre_coordinates_are_converted_to_millimetres(tmp_path):
    path = _unit_file(
        tmp_path, "nm.landmarkAscii", "nm", "1.0e+07 2.0e+07 3.0e+07", "-1.0e+06 0.0 5.0e+06"
    )
    coords, issues = _parse_one_file(path)
    assert issues == []
    assert np.allclose(coords, [[10.0, 20.0, 30.0], [-1.0, 0.0, 5.0]])


def test_the_declared_unit_is_readable_without_parsing_the_whole_file(tmp_path):
    path = _unit_file(tmp_path, "nm.landmarkAscii", "nm", "1 2 3", "4 5 6")
    assert file_unit(path) == "nm"


def test_millimetre_files_are_left_exactly_alone(tmp_path):
    path = _unit_file(tmp_path, "mm.landmarkAscii", "mm", "1.5 2.5 3.5", "4.5 5.5 6.5")
    coords, issues = _parse_one_file(path)
    assert issues == []
    assert np.allclose(coords, [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])


def test_a_file_declaring_no_unit_is_not_rescaled(tmp_path):
    """Silence is not a claim about units, so nothing is changed - but the
    batch report says so, because a wrong assumption here is invisible."""
    path = _write(tmp_path, "plain.landmarkAscii", VALID_CONTENT)
    coords, issues = _parse_one_file(path)
    assert issues == []
    assert np.allclose(coords[0], [1.0, 2.0, 3.0])
    assert file_unit(path) == ""


@pytest.mark.parametrize("unit,factor", [("nm", 1e-6), ("um", 1e-3), ("cm", 10.0), ("m", 1000.0)])
def test_every_supported_unit_scales_correctly(tmp_path, unit, factor):
    path = _unit_file(tmp_path, f"{unit}.landmarkAscii", unit, "1 2 3", "4 5 6")
    coords, _issues = _parse_one_file(path)
    assert np.allclose(coords[0], np.array([1.0, 2.0, 3.0]) * factor)
    assert MM_PER_UNIT[unit] == factor


def test_an_unrecognised_unit_is_rejected_rather_than_guessed(tmp_path):
    path = _unit_file(tmp_path, "odd.landmarkAscii", "furlongs", "1 2 3", "4 5 6")
    coords, issues = _parse_one_file(path)
    assert len(coords) == 0
    assert "furlongs" in issues[0]


def test_the_conversion_is_reported_in_the_import_validation(tmp_path):
    path = _unit_file(tmp_path, "nm.landmarkAscii", "nm", "1e7 2e7 3e7", "4e7 5e7 6e7")
    result = import_amira_files([path])
    reasons = " ".join(result.validation["Reason"].astype(str))
    assert "'nm'" in reasons and "millimetres" in reasons
    assert STATUS_INFO in set(result.validation["Status"])


def test_a_batch_mixing_units_is_flagged(tmp_path):
    """The dangerous case: GPA aligns them happily while they sit a
    million-fold apart in size, so every allometry result would be nonsense."""
    nm = _unit_file(tmp_path, "a.landmarkAscii", "nm", "1e7 2e7 3e7", "4e7 5e7 6e7")
    mm = _unit_file(tmp_path, "b.landmarkAscii", "mm", "10 20 30", "40 50 60")
    result = import_amira_files([nm, mm])
    reasons = " ".join(result.validation["Reason"].astype(str))
    assert "mixes coordinate units" in reasons
    assert STATUS_WARNING in set(result.validation["Status"])


def test_a_consistent_batch_is_not_flagged_as_mixed(tmp_path):
    a = _unit_file(tmp_path, "a.landmarkAscii", "nm", "1e7 2e7 3e7", "4e7 5e7 6e7")
    b = _unit_file(tmp_path, "b.landmarkAscii", "nm", "2e7 3e7 4e7", "5e7 6e7 7e7")
    result = import_amira_files([a, b])
    reasons = " ".join(result.validation["Reason"].astype(str))
    assert "mixes coordinate units" not in reasons


def test_converted_files_land_at_a_plausible_anatomical_size(tmp_path):
    """A regression guard with a real-world flavour: a pelvis landmark set in
    nm must come out in tens of millimetres, not tens of millions."""
    path = _unit_file(
        tmp_path,
        "pelvis.landmarkAscii",
        "nm",
        "-1.326673400000000e+07 1.541731840000000e+08 -3.282402240000000e+08",
        "-1.491558200000000e+07 1.726954240000000e+08 -3.160196800000000e+08",
    )
    coords, _issues = _parse_one_file(path)
    assert np.abs(coords).max() < 1000.0
    assert np.allclose(coords[0], [-13.266734, 154.173184, -328.240224])


def test_a_file_shorter_than_its_declared_count_is_excluded(tmp_path):
    """Imported alone, a truncated file used to pass as a specimen with fewer landmarks."""
    from gmm_core.validation import STATUS_FATAL

    body = "\n".join(f"{i}.0 {i}.5 {i}.25" for i in range(4))
    text = f"# AmiraMesh 3D ASCII 2.0\n\ndefine Markers 5\n\nMarkers {{ float[3] Coordinates }} @1\n\n@1\n{body}\n"
    path = tmp_path / "short.landmarkAscii"
    path.write_text(text)
    result = import_amira_files([path])
    assert result.raw.empty
    fatal = result.validation[result.validation["Status"] == STATUS_FATAL]
    assert fatal["Reason"].str.contains("declares 5 landmarks").any()

    path.write_text(text.replace("define Markers 5", "define Markers 4"))
    assert len(import_amira_files([path]).raw) == 1
