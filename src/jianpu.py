"""Jianpu conversion helpers (MVP).

This module converts MusicXML/MIDI-parsed music21 scores into single-line jianpu text.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE_INTERVALS = [0, 2, 3, 5, 7, 8, 10]


@dataclass
class MelodyCandidate:
    """Candidate melody stream stats for stream selection."""

    name: str
    stream: Any
    avg_midi: float
    density: float
    note_count: int
    score: float


def _first(iterable: Iterable[Any]) -> Optional[Any]:
    for item in iterable:
        return item
    return None


def _safe_flatten(stream_obj: Any) -> Any:
    """Flatten stream with best-effort compatibility across music21 versions."""
    if hasattr(stream_obj, "flatten"):
        return stream_obj.flatten()
    return stream_obj.flat


def _extract_primary_pitch(element: Any) -> Optional[Any]:
    """Return a representative pitch object for note/chord events."""
    if getattr(element, "isRest", False):
        return None
    if getattr(element, "isNote", False):
        return getattr(element, "pitch", None)
    if getattr(element, "isChord", False):
        pitches = list(getattr(element, "pitches", []))
        if not pitches:
            return None
        return max(pitches, key=lambda p: int(getattr(p, "midi", 0) or 0))
    return None


def _stream_note_metrics(stream_obj: Any) -> Tuple[int, float, float]:
    """Compute note count, average MIDI, and note density per quarter length."""
    flat = _safe_flatten(stream_obj)
    notes = list(getattr(flat, "notes", []))
    midi_values: List[int] = []

    for element in notes:
        pitch = _extract_primary_pitch(element)
        if pitch is None:
            continue
        midi_values.append(int(getattr(pitch, "midi", 60) or 60))

    note_count = len(midi_values)
    avg_midi = float(mean(midi_values)) if midi_values else 0.0
    highest_time = float(getattr(flat, "highestTime", 0.0) or 0.0)
    density = float(note_count) / max(1.0, highest_time)
    return note_count, avg_midi, density


def _iter_candidate_streams(score: Any) -> List[Tuple[str, Any]]:
    """Yield candidate part/voice streams for melody selection."""
    candidates: List[Tuple[str, Any]] = []

    parts = list(getattr(score, "parts", []))
    if not parts:
        return [("score", score)]

    for part_idx, part in enumerate(parts, start=1):
        part_name = str(getattr(part, "id", "") or f"part_{part_idx}")
        candidates.append((part_name, part))

        voices = list(part.recurse().getElementsByClass("Voice"))
        for voice_idx, voice in enumerate(voices, start=1):
            candidates.append((f"{part_name}:voice_{voice_idx}", voice))

    return candidates


def select_melody_stream(score: Any) -> Tuple[Any, str]:
    """Select melody stream using high-pitch + moderate-density heuristic.

    Input:
    - score: music21 score object.

    Output:
    - (selected_stream, source_name)
    """
    stream_candidates = _iter_candidate_streams(score)
    ranked: List[MelodyCandidate] = []

    target_density = 1.5
    for name, stream_obj in stream_candidates:
        note_count, avg_midi, density = _stream_note_metrics(stream_obj)
        if note_count == 0:
            continue

        density_penalty = abs(density - target_density) * 8.0
        sparse_penalty = 8.0 if note_count < 3 else 0.0
        rank_score = avg_midi - density_penalty - sparse_penalty
        ranked.append(
            MelodyCandidate(
                name=name,
                stream=stream_obj,
                avg_midi=avg_midi,
                density=density,
                note_count=note_count,
                score=rank_score,
            )
        )

    if not ranked:
        return score, "score"

    ranked.sort(key=lambda c: c.score, reverse=True)
    winner = ranked[0]
    return winner.stream, winner.name


def _extract_tempo(score: Any) -> Optional[float]:
    marks = list(_safe_flatten(score).getElementsByClass("MetronomeMark"))
    if not marks:
        return None
    number = getattr(marks[0], "number", None)
    return float(number) if isinstance(number, (int, float)) else None


def _extract_time_signature(score: Any, melody_stream: Any) -> str:
    ts = _first(_safe_flatten(melody_stream).getElementsByClass("TimeSignature"))
    if ts is None:
        ts = _first(_safe_flatten(score).getElementsByClass("TimeSignature"))
    return str(getattr(ts, "ratioString", "4/4") or "4/4")


def _extract_key(score: Any, melody_stream: Any) -> Tuple[int, str, str]:
    """Return (tonic_pitch_class, mode, key_name)."""
    key_obj = _first(_safe_flatten(melody_stream).getElementsByClass("Key"))
    if key_obj is None:
        key_obj = _first(_safe_flatten(score).getElementsByClass("Key"))

    if key_obj is None:
        key_signature = _first(_safe_flatten(melody_stream).getElementsByClass("KeySignature"))
        if key_signature is None:
            key_signature = _first(_safe_flatten(score).getElementsByClass("KeySignature"))
        if key_signature is not None and hasattr(key_signature, "asKey"):
            try:
                key_obj = key_signature.asKey("major")
            except Exception:
                key_obj = None

    if key_obj is None:
        return 0, "major", "C major"

    tonic = getattr(key_obj, "tonic", None)
    mode = str(getattr(key_obj, "mode", "major") or "major").lower()
    tonic_pc = int(getattr(tonic, "pitchClass", 0) or 0)
    tonic_name = str(getattr(tonic, "name", "C") or "C")
    if mode not in {"major", "minor"}:
        mode = "major"
    return tonic_pc, mode, f"{tonic_name} {mode}"


def _measure_quarter_length(time_signature: str) -> float:
    """Compute measure duration in quarterLength from ratio string."""
    try:
        numerator_text, denominator_text = time_signature.split("/")
        numerator = int(numerator_text)
        denominator = int(denominator_text)
        if numerator <= 0 or denominator <= 0:
            raise ValueError("invalid time signature")
        return float(numerator) * (4.0 / float(denominator))
    except Exception:
        return 4.0


def _closest_scale_degree(relative_pc: int, mode: str) -> Tuple[int, int]:
    """Map relative pitch-class to (degree_idx, accidental_steps)."""
    scale = MAJOR_SCALE_INTERVALS if mode == "major" else MINOR_SCALE_INTERVALS

    best_degree = 0
    best_delta = 0
    best_abs = 99

    for degree_idx, interval in enumerate(scale):
        delta = relative_pc - interval
        while delta > 6:
            delta -= 12
        while delta < -6:
            delta += 12

        abs_delta = abs(delta)
        if abs_delta < best_abs:
            best_abs = abs_delta
            best_degree = degree_idx
            best_delta = delta

    if best_delta > 1:
        best_delta = 1
    if best_delta < -1:
        best_delta = -1
    return best_degree, best_delta


def pitch_to_jianpu_token(pitch: Any, tonic_pc: int, mode: str, reference_octave: int = 4) -> str:
    """Convert one pitch into jianpu token.

    Input:
    - pitch: music21 pitch object.
    - tonic_pc: tonic pitch class [0..11].
    - mode: 'major' or 'minor'.
    - reference_octave: no-mark octave center (default: 4).

    Output:
    - token like '1', '3#', '5.', '6,'
    """
    pitch_class = int(getattr(pitch, "pitchClass", 0) or 0)
    octave = int(getattr(pitch, "octave", reference_octave) or reference_octave)
    midi = int(getattr(pitch, "midi", 60) or 60)

    relative_pc = (pitch_class - tonic_pc) % 12
    degree_idx, accidental_steps = _closest_scale_degree(relative_pc, mode)

    scale = MAJOR_SCALE_INTERVALS if mode == "major" else MINOR_SCALE_INTERVALS
    tonic_midi = 12 * (reference_octave + 1) + tonic_pc
    degree_base_midi = tonic_midi + scale[degree_idx]
    octave_shift = int(round((midi - degree_base_midi) / 12.0))

    degree_text = str(degree_idx + 1)
    accidental_text = "#" if accidental_steps > 0 else ("b" if accidental_steps < 0 else "")
    octave_text = "." * max(0, octave_shift) + "," * max(0, -octave_shift)
    return f"{degree_text}{accidental_text}{octave_text}"


def _strip_ties(stream_obj: Any) -> Any:
    """Merge tie chains into single events when possible."""
    flat = _safe_flatten(stream_obj)
    if not hasattr(flat, "stripTies"):
        return flat

    try:
        merged = flat.stripTies(inPlace=False)
        return merged if merged is not None else flat
    except TypeError:
        try:
            merged = flat.stripTies()
            return merged if merged is not None else flat
        except Exception:
            return flat
    except Exception:
        return flat


def _collect_events(melody_stream: Any, tonic_pc: int, mode: str) -> List[Dict[str, Any]]:
    """Collect monophonic note/rest events with offset and duration in quarterLength."""
    merged = _strip_ties(melody_stream)
    flat = _safe_flatten(merged)
    elements = sorted(list(flat.notesAndRests), key=lambda el: float(getattr(el, "offset", 0.0) or 0.0))

    events: List[Dict[str, Any]] = []
    for element in elements:
        start = float(getattr(element, "offset", 0.0) or 0.0)
        duration = float(getattr(element, "quarterLength", 0.0) or 0.0)
        if duration <= 0:
            continue

        if getattr(element, "isRest", False):
            token = "0"
            is_rest = True
        else:
            pitch = _extract_primary_pitch(element)
            if pitch is None:
                continue
            token = pitch_to_jianpu_token(pitch, tonic_pc=tonic_pc, mode=mode)
            is_rest = False

        events.append({"start": start, "duration": duration, "token": token, "is_rest": is_rest})

    return events


def _render_measures(
    events: Sequence[Dict[str, Any]],
    measure_ql: float,
    grid_unit_ql: float,
) -> str:
    """Render events into jianpu text with one measure per line."""
    if measure_ql <= 0:
        measure_ql = 4.0
    if grid_unit_ql <= 0:
        grid_unit_ql = 0.5

    total_ql = max((float(e["start"]) + float(e["duration"]) for e in events), default=measure_ql)
    measure_count = max(1, int(ceil(total_ql / measure_ql)))
    steps_per_measure = max(1, int(round(measure_ql / grid_unit_ql)))
    total_steps = measure_count * steps_per_measure

    cells = ["0"] * total_steps

    for event in sorted(events, key=lambda e: float(e["start"])):
        start_step = int(round(float(event["start"]) / grid_unit_ql))
        dur_steps = max(1, int(round(float(event["duration"]) / grid_unit_ql)))
        if start_step >= total_steps:
            continue

        end_step = min(total_steps, start_step + dur_steps)
        cells[start_step] = str(event["token"])
        for idx in range(start_step + 1, end_step):
            cells[idx] = "-"

    lines: List[str] = []
    for measure_idx in range(measure_count):
        left = measure_idx * steps_per_measure
        right = left + steps_per_measure
        body = " ".join(cells[left:right])
        lines.append(f"| {body} |")

    return "\n".join(lines) + "\n"


def score_to_jianpu(score: Any, grid_unit_ql: float = 0.5) -> Dict[str, Any]:
    """Convert score to single-part jianpu text and metadata.

    Input:
    - score: music21 score parsed from MIDI/MusicXML.
    - grid_unit_ql: quantization unit in quarterLength (0.5 means 1/8 note).

    Output:
    - dict:
      - text: jianpu plain text, one measure per line.
      - meta: {tempo, time_signature, key, melody_source, grid_unit_ql}
    """
    melody_stream, melody_source = select_melody_stream(score)

    tempo = _extract_tempo(score)
    time_signature = _extract_time_signature(score, melody_stream)
    tonic_pc, mode, key_name = _extract_key(score, melody_stream)

    events = _collect_events(melody_stream, tonic_pc=tonic_pc, mode=mode)
    measure_ql = _measure_quarter_length(time_signature)
    text = _render_measures(events=events, measure_ql=measure_ql, grid_unit_ql=grid_unit_ql)

    return {
        "text": text,
        "meta": {
            "tempo": tempo,
            "time_signature": time_signature,
            "key": key_name,
            "melody_source": melody_source,
            "grid_unit_ql": float(grid_unit_ql),
        },
    }
