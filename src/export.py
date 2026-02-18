"""Export helpers for MIDI and MusicXML.

All note/chord event times are in seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.errors import MissingDependencyError


def require_mido():
    """Return mido module or raise dependency error."""
    try:
        import mido  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError("Missing dependency: mido. Install with `pip install mido`.") from exc
    return mido


def require_music21():
    """Return music21 module or raise dependency error."""
    try:
        import music21  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "Missing dependency: music21. Install with `pip install music21`."
        ) from exc
    return music21


def export_bass_midi(notes: List[Dict[str, object]], tempo: float, output_path: Path) -> Path:
    """Export bass notes to a single-track MIDI file.

    Input:
    - notes: list of dict events with keys {time, duration, midi, velocity}; times in seconds.
    - tempo: BPM.
    - output_path: target MIDI path.

    Output:
    - output_path (Path) after writing.
    """
    mido = require_mido()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ticks_per_beat = 480
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    bpm = max(1.0, float(tempo) if tempo else 120.0)
    sec_per_beat = 60.0 / bpm
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))

    notes_sorted = sorted(notes, key=lambda n: float(n["time"]))
    cursor_time = 0.0
    for note in notes_sorted:
        start = float(note["time"])
        duration = max(0.05, float(note["duration"]))
        midi_note = int(note["midi"])
        velocity = int(note.get("velocity", 84))

        delta_sec = max(0.0, start - cursor_time)
        delta_ticks = int(round(delta_sec / sec_per_beat * ticks_per_beat))
        track.append(mido.Message("note_on", note=midi_note, velocity=velocity, time=delta_ticks))

        note_ticks = int(round(duration / sec_per_beat * ticks_per_beat))
        note_ticks = max(1, note_ticks)
        track.append(mido.Message("note_off", note=midi_note, velocity=0, time=note_ticks))

        cursor_time = start + duration

    midi.save(str(output_path))
    return output_path


def export_musicxml(
    chords: List[Dict[str, object]],
    notes: List[Dict[str, object]],
    tempo: float,
    output_path: Path,
) -> Path:
    """Export chords and bass notes to MusicXML.

    Input:
    - chords: list of chord events. Supports:
      - legacy: {time: float seconds, chord: str}
      - new: {start: float, end: float, label: str, confidence: float}
    - notes: list of {time: float seconds, duration: float seconds, midi: int, velocity: int}.
    - tempo: BPM.
    - output_path: target MusicXML file path.

    Output:
    - output_path (Path) after writing.
    """
    music21 = require_music21()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpm = max(1.0, float(tempo) if tempo else 120.0)

    score = music21.stream.Score(id="music-score-ai")

    chord_part = music21.stream.Part(id="Chords")
    chord_part.insert(0, music21.tempo.MetronomeMark(number=bpm))

    sec_to_ql = bpm / 60.0
    for event in chords:
        label = str(event.get("label", event.get("chord", "N")))
        if label == "N":
            continue
        sec = float(event.get("start", event.get("time", 0.0)))
        offset = sec * sec_to_ql
        try:
            symbol = music21.harmony.ChordSymbol(label)
            chord_part.insert(offset, symbol)
        except Exception:
            # Ignore malformed labels instead of failing the whole export.
            continue

    bass_part = music21.stream.Part(id="Bass")
    bass_part.insert(0, music21.clef.BassClef())
    bass_part.insert(0, music21.tempo.MetronomeMark(number=bpm))

    for note_event in notes:
        sec = float(note_event["time"])
        dur_sec = max(0.05, float(note_event["duration"]))
        midi_note = int(note_event["midi"])

        n = music21.note.Note(midi_note)
        n.quarterLength = dur_sec * sec_to_ql
        bass_part.insert(sec * sec_to_ql, n)

    score.insert(0, chord_part)
    score.insert(0, bass_part)
    score.write("musicxml", fp=str(output_path))
    return output_path
