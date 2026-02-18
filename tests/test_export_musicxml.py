from pathlib import Path

import pytest

pytest.importorskip("music21")

from src.export import export_musicxml


def test_export_musicxml_contains_score_tag(tmp_path: Path):
    chords = [{"time": 0.0, "chord": "C"}]
    notes = [{"time": 0.0, "duration": 0.5, "midi": 48, "velocity": 80}]
    out = tmp_path / "score.musicxml"

    result = export_musicxml(chords=chords, notes=notes, tempo=120.0, output_path=out)
    assert result.exists()

    content = result.read_text(encoding="utf-8", errors="ignore")
    assert "<score-partwise" in content or "<score-timewise" in content
