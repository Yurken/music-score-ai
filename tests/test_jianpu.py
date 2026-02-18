import pytest

music21 = pytest.importorskip("music21")

from src.jianpu import score_to_jianpu


def test_jianpu_c_major_contains_1235_and_meta():
    score = music21.stream.Score(id="jianpu-test")
    part = music21.stream.Part(id="melody")

    part.append(music21.tempo.MetronomeMark(number=120))
    part.append(music21.meter.TimeSignature("4/4"))
    part.append(music21.key.Key("C"))

    # 1/8-note grid melody in C major: 1 2 3 5
    for pitch_name in ["C4", "D4", "E4", "G4"]:
        n = music21.note.Note(pitch_name)
        n.quarterLength = 0.5
        part.append(n)

    # Fill remaining 4/4 measure with rest.
    part.append(music21.note.Rest(quarterLength=2.0))

    score.insert(0, part)

    result = score_to_jianpu(score, grid_unit_ql=0.5)

    text = result["text"]
    meta = result["meta"]

    assert "1 2 3 5" in text
    assert meta["time_signature"] == "4/4"
    assert str(meta["key"]).lower().startswith("c")
    assert meta["tempo"] == 120.0
