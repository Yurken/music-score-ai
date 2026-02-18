"""Two-stage chord recognition with Chordino-first strategy.

Priority:
1) Use Chordino (Vamp) when plugin is available.
2) Fallback to pure Python chroma-template recognition with smoothing.

All times are in seconds.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.audio_io import require_librosa

NOTE_NAMES: List[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_SIM_THRESHOLD = 0.35


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize vector; return zeros if near-silent."""
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return np.zeros_like(vec)
    return vec / norm


def _build_chord_templates() -> List[Dict[str, Any]]:
    """Build template bank for maj/min/7 in all 12 keys."""
    templates: List[Dict[str, Any]] = []
    quality_defs = {
        "": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
        "m": np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
        "7": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=np.float32),
    }
    for root_idx, root_name in enumerate(NOTE_NAMES):
        for suffix, base in quality_defs.items():
            templates.append({"label": f"{root_name}{suffix}", "vector": _normalize(np.roll(base, root_idx))})
    return templates


CHORD_TEMPLATES = _build_chord_templates()


def normalize_chord_label(label: str) -> str:
    """Normalize different chord label styles into {root, root+'m', root+'7'} or N.

    Examples:
    - 'C:maj' -> 'C'
    - 'A:min' -> 'Am'
    - 'G:7' -> 'G7'
    - 'N' -> 'N'
    """
    raw = (label or "").strip()
    if not raw:
        return "N"

    raw = raw.split("/")[0]  # drop slash bass
    if raw.upper() in {"N", "X"}:
        return "N"

    m = re.match(r"^([A-G](?:#|b)?)(?::?([a-zA-Z0-9+()-]*))?$", raw)
    if not m:
        return raw

    root = m.group(1)
    quality = (m.group(2) or "").lower()

    if quality in {"", "maj", "major", "maj7", "6", "add9"}:
        return root
    if quality in {"m", "min", "minor", "min7", "m7", "m6"}:
        return f"{root}m"
    if quality in {"7", "dom7", "dominant7", "9"}:
        return f"{root}7"

    # If quality is unknown, keep root as a safe simplification.
    return root


def classify_chroma_frame(chroma_vec: np.ndarray) -> Tuple[str, float]:
    """Classify one chroma frame into chord label and confidence.

    Input:
    - chroma_vec: shape (12,), one frame of chroma energies.

    Output:
    - (label, confidence)
      - label: chord label string (e.g. 'C', 'Am', 'G7') or 'N'.
      - confidence: float in [0, 1].
    """
    chroma = _normalize(chroma_vec.astype(np.float32))
    if not np.any(chroma):
        return "N", 0.0

    scores = np.array([float(np.dot(chroma, tpl["vector"])) for tpl in CHORD_TEMPLATES], dtype=np.float32)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    second_score = float(np.partition(scores, -2)[-2]) if scores.size > 1 else 0.0

    if best_score < CHORD_SIM_THRESHOLD:
        return "N", max(0.0, min(1.0, best_score))

    margin = max(0.0, best_score - second_score)
    confidence = max(0.0, min(1.0, (0.65 * best_score) + (0.35 * min(1.0, margin * 3.0))))
    return str(CHORD_TEMPLATES[best_idx]["label"]), confidence


def estimate_key_center(chroma: np.ndarray) -> str:
    """Estimate rough key center using Krumhansl-Schmuckler profiles.

    Input:
    - chroma: np.ndarray shape (12, n_frames)

    Output:
    - key string like 'C major' or 'A minor'.
    """
    if chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] == 0:
        return "C major"

    pitch_energy = _normalize(np.sum(chroma, axis=1).astype(np.float32))
    major_profile = _normalize(np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]))
    minor_profile = _normalize(np.array([6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]))

    best_key = "C major"
    best_score = -1.0
    for i, root in enumerate(NOTE_NAMES):
        maj_score = float(np.dot(pitch_energy, np.roll(major_profile, i)))
        if maj_score > best_score:
            best_score = maj_score
            best_key = f"{root} major"

        min_score = float(np.dot(pitch_energy, np.roll(minor_profile, i)))
        if min_score > best_score:
            best_score = min_score
            best_key = f"{root} minor"

    return best_key


def _majority_vote_smoothing(labels: Sequence[str], window_size: int = 5) -> List[str]:
    """Apply local majority voting to reduce frame-level label jitter."""
    if not labels:
        return []
    if window_size <= 1:
        return list(labels)

    half = window_size // 2
    result: List[str] = []
    for idx in range(len(labels)):
        left = max(0, idx - half)
        right = min(len(labels), idx + half + 1)
        window = labels[left:right]
        winner = Counter(window).most_common(1)[0][0]
        result.append(winner)
    return result


def _merge_adjacent_same_label(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge adjacent segments with identical labels."""
    if not segments:
        return []

    merged: List[Dict[str, Any]] = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if str(prev["label"]) == str(seg["label"]):
            prev_duration = float(prev["end"]) - float(prev["start"])
            seg_duration = float(seg["end"]) - float(seg["start"])
            total = max(1e-9, prev_duration + seg_duration)
            prev["end"] = float(seg["end"])
            prev["confidence"] = (
                float(prev["confidence"]) * prev_duration + float(seg["confidence"]) * seg_duration
            ) / total
        else:
            merged.append(seg.copy())
    return merged


def _merge_short_segments(segments: List[Dict[str, Any]], min_duration: float = 0.35) -> List[Dict[str, Any]]:
    """Merge too-short segments into neighboring segments."""
    if len(segments) <= 1:
        return segments

    merged = [seg.copy() for seg in segments]
    idx = 0
    while idx < len(merged):
        dur = float(merged[idx]["end"]) - float(merged[idx]["start"])
        if dur >= min_duration or len(merged) == 1:
            idx += 1
            continue

        if idx == 0:
            merged[1]["start"] = float(merged[0]["start"])
            merged[1]["confidence"] = float(max(merged[1]["confidence"], merged[0]["confidence"]))
            del merged[0]
            continue

        if idx == len(merged) - 1:
            merged[-2]["end"] = float(merged[-1]["end"])
            merged[-2]["confidence"] = float(max(merged[-2]["confidence"], merged[-1]["confidence"]))
            del merged[-1]
            idx = max(0, len(merged) - 1)
            continue

        left = merged[idx - 1]
        right = merged[idx + 1]
        current = merged[idx]

        # Prefer neighbor with same label, otherwise higher confidence.
        if str(left["label"]) == str(current["label"]):
            choose_left = True
        elif str(right["label"]) == str(current["label"]):
            choose_left = False
        else:
            choose_left = float(left["confidence"]) >= float(right["confidence"])

        if choose_left:
            left["end"] = float(current["end"])
            left["confidence"] = float(max(left["confidence"], current["confidence"]))
            del merged[idx]
            idx = max(0, idx - 1)
        else:
            right["start"] = float(current["start"])
            right["confidence"] = float(max(right["confidence"], current["confidence"]))
            del merged[idx]

    return _merge_adjacent_same_label(merged)


def _frames_to_segments(
    labels: Sequence[str],
    confidences: Sequence[float],
    frame_times: np.ndarray,
    min_duration: float,
) -> List[Dict[str, Any]]:
    """Convert frame-level labels to contiguous chord segments."""
    if not labels:
        return []

    if frame_times.ndim != 1 or frame_times.size != len(labels):
        raise ValueError("frame_times must be 1D and have same length as labels")

    if frame_times.size == 1:
        frame_step = 0.1
    else:
        diffs = np.diff(frame_times)
        frame_step = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.1

    segments: List[Dict[str, Any]] = []
    start_idx = 0
    current_label = str(labels[0])

    for idx in range(1, len(labels) + 1):
        boundary = idx == len(labels) or str(labels[idx]) != current_label
        if not boundary:
            continue

        start = float(frame_times[start_idx])
        if idx < len(labels):
            end = float(frame_times[idx])
        else:
            end = float(frame_times[-1] + frame_step)
        if end <= start:
            end = start + frame_step

        seg_conf = float(np.mean(confidences[start_idx:idx])) if idx > start_idx else 0.0
        segments.append({"start": start, "end": end, "label": current_label, "confidence": seg_conf})

        if idx < len(labels):
            start_idx = idx
            current_label = str(labels[idx])

    return _merge_short_segments(_merge_adjacent_same_label(segments), min_duration=min_duration)


def infer_chords_from_chroma(
    chroma: np.ndarray,
    frame_times: np.ndarray,
    min_segment_duration: float = 0.35,
    smooth_window: int = 5,
) -> List[Dict[str, Any]]:
    """Infer chord segments from chroma features.

    Input:
    - chroma: np.ndarray shape (12, n_frames)
    - frame_times: np.ndarray shape (n_frames,), each frame timestamp in seconds
    - min_segment_duration: minimal kept segment duration in seconds
    - smooth_window: majority-vote window size for label smoothing

    Output:
    - list[dict] with unified format:
      [{"start": float, "end": float, "label": str, "confidence": float}, ...]
    """
    if chroma.ndim != 2 or chroma.shape[0] != 12:
        raise ValueError("chroma must have shape (12, n_frames)")
    if frame_times.ndim != 1 or chroma.shape[1] != frame_times.shape[0]:
        raise ValueError("frame_times length must match chroma frame count")
    if frame_times.size == 0:
        return []

    frame_labels: List[str] = []
    frame_confidences: List[float] = []
    for frame_idx in range(chroma.shape[1]):
        label, conf = classify_chroma_frame(chroma[:, frame_idx])
        frame_labels.append(label)
        frame_confidences.append(float(conf))

    smooth_labels = _majority_vote_smoothing(frame_labels, window_size=max(1, int(smooth_window)))
    segments = _frames_to_segments(
        labels=smooth_labels,
        confidences=frame_confidences,
        frame_times=frame_times,
        min_duration=max(0.05, float(min_segment_duration)),
    )
    return segments


def _extract_chordino_items(raw_output: Any) -> List[Dict[str, Any]]:
    """Extract list-style events from vamp.collect output."""
    if isinstance(raw_output, dict):
        if isinstance(raw_output.get("list"), list):
            return [item for item in raw_output["list"] if isinstance(item, dict)]
        for value in raw_output.values():
            if isinstance(value, dict) and isinstance(value.get("list"), list):
                return [item for item in value["list"] if isinstance(item, dict)]
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return [item for item in value if isinstance(item, dict)]
    if isinstance(raw_output, list) and (not raw_output or isinstance(raw_output[0], dict)):
        return [item for item in raw_output if isinstance(item, dict)]
    return []


def _parse_chordino_output(raw_output: Any) -> List[Dict[str, Any]]:
    """Parse Chordino output into unified chord segment format."""
    items = _extract_chordino_items(raw_output)
    if not items:
        return []

    parsed: List[Dict[str, Any]] = []
    for item in items:
        start = float(item.get("timestamp", item.get("time", 0.0)))
        duration = float(item.get("duration", 0.0) or 0.0)

        values = item.get("values")
        label: str = "N"
        confidence = 1.0

        if isinstance(values, list) and values:
            label = normalize_chord_label(str(values[0]))
            if len(values) > 1 and isinstance(values[1], (float, int)):
                confidence = float(values[1])
        elif isinstance(values, str):
            label = normalize_chord_label(values)
        elif isinstance(item.get("label"), str):
            label = normalize_chord_label(str(item["label"]))

        parsed.append({"start": start, "duration": duration, "label": label, "confidence": confidence})

    # Ensure temporal end boundaries.
    result: List[Dict[str, Any]] = []
    for idx, event in enumerate(parsed):
        start = float(event["start"])
        duration = float(event["duration"])
        if duration > 0:
            end = start + duration
        elif idx + 1 < len(parsed):
            end = float(parsed[idx + 1]["start"])
        else:
            end = start + 0.5
        if end <= start:
            end = start + 0.1
        result.append(
            {
                "start": start,
                "end": end,
                "label": str(event["label"]),
                "confidence": float(max(0.0, min(1.0, event["confidence"]))),
            }
        )

    return _merge_adjacent_same_label(result)


def _find_chordino_plugin_id(vamp_module: Any) -> Optional[str]:
    """Discover chordino plugin identifier from vamp host."""
    try:
        plugins = vamp_module.list_plugins()
    except Exception:
        return None

    keys: List[str] = []
    if isinstance(plugins, dict):
        keys = [str(k) for k in plugins.keys()]
    elif isinstance(plugins, list):
        keys = [str(k) for k in plugins]

    preferred = ["nnls-chroma:chordino", "chordino:chordino"]
    for candidate in preferred:
        if candidate in keys:
            return candidate

    for key in keys:
        lowered = key.lower()
        if "chordino" in lowered:
            return key

    return None


def try_chordino(y: np.ndarray, sr: int) -> Optional[List[Dict[str, Any]]]:
    """Try Chordino-based recognition if Vamp + plugin are available.

    Input:
    - y: mono waveform array
    - sr: sample rate

    Output:
    - chord segments in unified format, or None if unavailable/failed.
    """
    try:
        import vamp  # type: ignore
    except Exception:
        return None

    plugin_id = _find_chordino_plugin_id(vamp)
    if not plugin_id:
        return None

    audio = np.ascontiguousarray(y.astype(np.float32))
    outputs_to_try: List[Optional[str]] = [None, "chordino", "chord", "chords"]

    for output_name in outputs_to_try:
        try:
            if output_name is None:
                raw = vamp.collect(audio, sr, plugin_id)
            else:
                raw = vamp.collect(audio, sr, plugin_id, output=output_name)
        except Exception:
            continue

        parsed = _parse_chordino_output(raw)
        if parsed:
            return _merge_short_segments(parsed, min_duration=0.2)

    return None


def _extract_fallback_chroma(y: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Compute fallback chroma and frame times with librosa."""
    librosa = require_librosa()

    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    return chroma, frame_times


def estimate_chords(
    y: np.ndarray,
    sr: int,
    beats: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Estimate chord segments from audio.

    Strategy:
    1) Try Chordino (Vamp plugin) for higher quality labels.
    2) Fallback to pure Python chroma-template recognizer.

    Input:
    - y: mono waveform array
    - sr: sample rate
    - beats: optional beat timestamps (seconds), accepted for API compatibility

    Output:
    - unified chord list:
      [{"start": float, "end": float, "label": str, "confidence": float}, ...]
    """
    _ = beats  # currently unused; kept for backwards compatibility.
    if y.size == 0:
        return []

    chordino_segments = try_chordino(y=y, sr=sr)
    if chordino_segments:
        return chordino_segments

    chroma, frame_times = _extract_fallback_chroma(y=y, sr=sr)
    return infer_chords_from_chroma(chroma=chroma, frame_times=frame_times)
