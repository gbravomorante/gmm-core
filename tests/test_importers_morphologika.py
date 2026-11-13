from gmm_core.io.morphologika import import_morphologika_files
from gmm_core.validation import STATUS_FATAL

TWO_SPECIMEN_FILE = """[individuals]
2
[landmarks]
2
[dimensions]
3
[names]
spec_a
spec_b
[rawpoints]
'spec_a
1.0 2.0 3.0
4.0 5.0 6.0
'spec_b
7.0 8.0 9.0
10.0 11.0 12.0
"""

MALFORMED_BLOCK_FILE = """[landmarks]
2
[dimensions]
3
[rawpoints]
'good_one
1.0 2.0 3.0
4.0 5.0 6.0
'bad_one
1.0 2.0 3.0
not-a-number-line
"""


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_import_valid_multi_specimen_file(tmp_path):
    p1 = _write(tmp_path, "batch.txt", TWO_SPECIMEN_FILE)
    result = import_morphologika_files([p1])
    assert len(result.raw) == 2
    assert result.n_landmarks == 2
    assert result.n_dim == 3
    assert set(result.raw["ID"]) == {"spec_a", "spec_b"}
    assert list(result.raw.loc[result.raw["ID"] == "spec_a", "LM0_X"]) == [1.0]
    assert list(result.raw.loc[result.raw["ID"] == "spec_a", "LM1_Z"]) == [6.0]


def test_missing_landmarks_section_flagged(tmp_path):
    p1 = _write(tmp_path, "nosection.txt", "[rawpoints]\n'spec\n1 2 3\n")
    result = import_morphologika_files([p1])
    assert result.raw.empty
    assert (result.validation["Status"] == STATUS_FATAL).any()
    assert "[landmarks]" in " ".join(result.validation["Reason"])


def test_malformed_specimen_block_excluded_others_kept(tmp_path):
    p1 = _write(tmp_path, "mixed.txt", MALFORMED_BLOCK_FILE)
    result = import_morphologika_files([p1])
    assert "good_one" in set(result.raw["ID"])
    assert "bad_one" not in set(result.raw["ID"])
    assert "bad_one" in " ".join(result.validation["Reason"])


def test_dimension_mismatch_across_files_flagged(tmp_path):
    file_3d = """[landmarks]\n2\n[dimensions]\n3\n[rawpoints]\n'spec_3d\n1 2 3\n4 5 6\n"""
    file_2d = """[landmarks]\n2\n[dimensions]\n2\n[rawpoints]\n'spec_2d\n1 2\n4 5\n"""
    p1 = _write(tmp_path, "a.txt", file_3d)
    p2 = _write(tmp_path, "b.txt", file_2d)
    result = import_morphologika_files([p1, p2])
    assert result.n_dim == 3
    assert set(result.raw["ID"]) == {"spec_3d"}
    assert "spec_2d" not in set(result.raw["ID"])


def test_duplicate_specimen_name_within_file_excluded(tmp_path):
    content = """[landmarks]\n2\n[dimensions]\n3\n[rawpoints]\n'dup\n1 2 3\n4 5 6\n'dup\n7 8 9\n10 11 12\n"""
    p1 = _write(tmp_path, "dup.txt", content)
    result = import_morphologika_files([p1])
    assert "dup" not in set(result.raw["ID"])
    assert "duplicate" in " ".join(result.validation["Reason"]).lower()


def test_header_count_that_disagrees_with_the_blocks_is_reported(tmp_path):
    """A file declaring more individuals than it holds is how a truncated copy looks."""
    from gmm_core.validation import STATUS_WARNING

    path = tmp_path / "truncated.txt"
    path.write_text(TWO_SPECIMEN_FILE.replace("[individuals]\n2", "[individuals]\n3"))
    result = import_morphologika_files([path])
    assert len(result.raw) == 2
    warnings = result.validation[result.validation["Status"] == STATUS_WARNING]
    assert warnings["Reason"].str.contains("declares 3 individuals but 2").any()


def test_a_matching_header_count_raises_no_warning(tmp_path):
    path = tmp_path / "fine.txt"
    path.write_text(TWO_SPECIMEN_FILE)
    assert import_morphologika_files([path]).validation.empty
