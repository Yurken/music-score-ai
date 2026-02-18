"""FastAPI service for audio transcription and score conversion."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.audio_io import load_audio, validate_audio_extension
from src.bass import extract_bass_notes
from src.beat import extract_tempo_beats, infer_downbeats
from src.chords import estimate_chords
from src.convert import ALLOWED_OUTPUT_FORMATS, convert_score, validate_score_extension
from src.errors import InvalidInputError, MissingDependencyError
from src.export import export_bass_midi, export_musicxml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("music-score-ai")

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
INFERENCE_TIMEOUT_SECONDS = 90
BASE_OUTPUT_DIR = Path("outputs")

app = FastAPI(title="music-score-ai", version="0.1.0")


def create_job_dir() -> tuple[str, Path]:
    """Create output folder under ./outputs/{uuid}.

    Output:
    - (job_id, job_dir)
    """
    job_id = str(uuid.uuid4())
    job_dir = BASE_OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


async def save_upload_with_limit(upload_file: UploadFile, dst_path: Path, max_bytes: int) -> int:
    """Save uploaded file to disk with file-size limit.

    Input:
    - upload_file: FastAPI UploadFile.
    - dst_path: destination path.
    - max_bytes: max allowed size.

    Output:
    - total written bytes.

    Raises:
    - HTTPException(413) if file exceeds max size.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with dst_path.open("wb") as f:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail=f"File too large. Max {max_bytes} bytes")
            f.write(chunk)
    await upload_file.close()
    return total


def transcribe_audio_pipeline(audio_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Run end-to-end audio-to-score pipeline.

    Input:
    - audio_path: local mp3/wav path.
    - output_dir: destination directory for exports.

    Output JSON-like dict:
    - tempo: float BPM
    - beats: list[float seconds]
    - downbeats: list[float seconds]
    - chords: list[{start: float, end: float, label: str, confidence: float}]
    - bass_notes: list[{time: float, duration: float, midi: int, velocity: int}]
    - exports: {midi: path, musicxml: path}
    """
    y, sr = load_audio(audio_path)

    beat_result = extract_tempo_beats(y, sr)
    tempo = float(beat_result["tempo"])
    beats = list(beat_result["beats"])
    downbeats = infer_downbeats(beats, meter=4)

    chords = estimate_chords(y, sr, beats)
    bass_notes = extract_bass_notes(y, sr, beats)

    midi_path = output_dir / "bass.mid"
    musicxml_path = output_dir / "score.musicxml"
    export_bass_midi(bass_notes, tempo=tempo, output_path=midi_path)
    export_musicxml(chords=chords, notes=bass_notes, tempo=tempo, output_path=musicxml_path)

    return {
        "tempo": tempo,
        "beats": beats,
        "downbeats": downbeats,
        "chords": chords,
        "bass_notes": bass_notes,
        "exports": {"midi": str(midi_path), "musicxml": str(musicxml_path)},
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe/audio")
async def transcribe_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    try:
        suffix = validate_audio_extension(file.filename)
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id, job_dir = create_job_dir()
    audio_path = job_dir / f"input{suffix}"
    await save_upload_with_limit(file, audio_path, MAX_FILE_SIZE_BYTES)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(transcribe_audio_pipeline, audio_path, job_dir),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Transcription timed out after {INFERENCE_TIMEOUT_SECONDS} seconds",
        ) from exc
    except MissingDependencyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

    return {"job_id": job_id, **result}


@app.post("/convert/score")
async def convert_score_api(
    file: UploadFile = File(...),
    output_format: str = Form(...),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    try:
        suffix = validate_score_extension(file.filename)
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    fmt = output_format.lower().strip()
    if fmt not in ALLOWED_OUTPUT_FORMATS:
        raise HTTPException(status_code=400, detail=f"output_format must be one of {sorted(ALLOWED_OUTPUT_FORMATS)}")

    job_id, job_dir = create_job_dir()
    input_path = job_dir / f"input{suffix}"
    await save_upload_with_limit(file, input_path, MAX_FILE_SIZE_BYTES)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(convert_score, input_path, fmt, job_dir),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Conversion timed out after {INFERENCE_TIMEOUT_SECONDS} seconds",
        ) from exc
    except MissingDependencyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Score conversion failed")
        raise HTTPException(status_code=500, detail=f"Score conversion failed: {exc}") from exc

    return {"job_id": job_id, **result}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
