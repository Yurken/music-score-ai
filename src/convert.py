"""Score format conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.errors import InvalidInputError, MissingDependencyError
from src.jianpu import score_to_jianpu

ALLOWED_SCORE_EXTENSIONS = {".musicxml", ".xml", ".midi", ".mid"}
ALLOWED_OUTPUT_FORMATS = {"musicxml", "midi", "jianpu_text"}


def require_music21():
    """Return music21 module or raise dependency error."""
    try:
        import music21  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "Missing dependency: music21. Install with `pip install music21`."
        ) from exc
    return music21


def validate_score_extension(filename: str) -> str:
    """Validate score file extension.

    Input:
    - filename: uploaded filename string.

    Output:
    - normalized extension string.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SCORE_EXTENSIONS:
        raise InvalidInputError(
            f"Unsupported score format: {suffix or '<none>'}. Allowed: musicxml/xml/midi/mid"
        )
    return suffix


def convert_score(input_path: Path, output_format: str, output_dir: Path) -> Dict[str, Any]:
    """Convert MusicXML/MIDI into target format.

    Input:
    - input_path: source score file path.
    - output_format: one of {'musicxml', 'midi', 'jianpu_text'}.
    - output_dir: destination directory.

    Output:
    - dict:
      - output_format: selected format.
      - file_path: converted file path string.
      - content: inline text content for jianpu_text, else None.
      - meta: conversion metadata for jianpu_text, else None.
    """
    fmt = output_format.lower().strip()
    if fmt not in ALLOWED_OUTPUT_FORMATS:
        raise InvalidInputError(f"output_format must be one of: {sorted(ALLOWED_OUTPUT_FORMATS)}")

    music21 = require_music21()
    output_dir.mkdir(parents=True, exist_ok=True)

    score = music21.converter.parse(str(input_path))

    if fmt == "musicxml":
        out_path = output_dir / "converted.musicxml"
        score.write("musicxml", fp=str(out_path))
        return {"output_format": fmt, "file_path": str(out_path), "content": None, "meta": None}

    if fmt == "midi":
        out_path = output_dir / "converted.mid"
        score.write("midi", fp=str(out_path))
        return {"output_format": fmt, "file_path": str(out_path), "content": None, "meta": None}

    jianpu_result = score_to_jianpu(score, grid_unit_ql=0.5)
    content = str(jianpu_result["text"])
    meta = dict(jianpu_result["meta"])

    out_path = output_dir / "converted_jianpu.txt"
    out_path.write_text(content, encoding="utf-8")
    return {"output_format": fmt, "file_path": str(out_path), "content": content, "meta": meta}
