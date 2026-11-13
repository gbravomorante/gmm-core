import json

from gmm_core.io.slicer_json import import_slicer_files
from gmm_core.validation import STATUS_FATAL


def _markup_json(positions):
    return json.dumps(
        {
            "markups": [
                {
                    "type": "Fiducial",
                    "controlPoints": [{"position": list(p)} for p in positions],
                }
            ],
        }
    )


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_import_valid_files(tmp_path):
    content = _markup_json([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    p1 = _write(tmp_path, "spec_a.mrk.json", content)
    p2 = _write(tmp_path, "spec_b.mrk.json", content)
    result = import_slicer_files([p1, p2])
    assert len(result.raw) == 2
    assert result.n_landmarks == 3
    assert set(result.raw["ID"]) == {"spec_a", "spec_b"}
    assert (result.raw["SourceFormat"] == "Slicer_mrk_json").all()


def test_specimen_id_strips_full_mrk_json_suffix(tmp_path):
    content = _markup_json([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    p1 = _write(tmp_path, "weird.name.mrk.json", content)
    result = import_slicer_files([p1])
    assert list(result.raw["ID"]) == ["weird.name"]


def test_invalid_json_flagged(tmp_path):
    p1 = _write(tmp_path, "broken.mrk.json", "{not valid json")
    result = import_slicer_files([p1])
    assert result.raw.empty
    assert (result.validation["Status"] == STATUS_FATAL).any()


def test_missing_markups_flagged(tmp_path):
    p1 = _write(tmp_path, "nomarkups.mrk.json", json.dumps({"foo": "bar"}))
    result = import_slicer_files([p1])
    assert result.raw.empty
    assert "markups" in " ".join(result.validation["Reason"])


def test_landmark_count_mismatch_not_silently_dropped(tmp_path):
    p1 = _write(tmp_path, "four_lm_a.mrk.json", _markup_json([[0, 0, 0]] * 4))
    p2 = _write(tmp_path, "four_lm_b.mrk.json", _markup_json([[0, 0, 0]] * 4))
    p3 = _write(tmp_path, "five_lm.mrk.json", _markup_json([[0, 0, 0]] * 5))
    result = import_slicer_files([p1, p2, p3])
    assert result.n_landmarks == 4
    assert set(result.raw["ID"]) == {"four_lm_a", "four_lm_b"}
    assert "five_lm" in set(result.validation["ID"])
