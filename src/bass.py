"""Bass-line extraction and beat quantization.

All note timing is in seconds.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.audio_io import require_librosa


def _merge_adjacent_notes(notes: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not notes:
        return []

    merged = [notes[0].copy()]
    for note in notes[1:]:
        prev = merged[-1]
        prev_end = float(prev["time"]) + float(prev["duration"])
        if (
            int(prev["midi"]) == int(note["midi"])
            and abs(float(note["time"]) - prev_end) < 1e-4
        ):
            prev["duration"] = float(prev["duration"]) + float(note["duration"])
        else:
            merged.append(note.copy())
    return merged


def extract_bass_notes(y: np.ndarray, sr: int, beats: List[float]) -> List[Dict[str, object]]:
    """Extract a rough bass-note sequence quantized to beat grid.

    Input:
    - y: Mono waveform array.
    - sr: Sample rate.
    - beats: Beat timestamps in seconds.

    Output:
    - list[dict], each note has:
      - time: float note start in seconds.
      - duration: float note duration in seconds.
      - midi: int MIDI note number.
      - velocity: int MIDI velocity.
    """
    if not beats:
        return []

    librosa = require_librosa()

    y_harm, _ = librosa.effects.hpss(y)
    n_fft = 4096
    hop_length = 512

    stft_mag = np.abs(librosa.stft(y_harm, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frame_times = librosa.frames_to_time(np.arange(stft_mag.shape[1]), sr=sr, hop_length=hop_length)

    bass_band = (freqs >= 30.0) & (freqs <= 220.0)
    bass_spec = stft_mag[bass_band, :]
    bass_freqs = freqs[bass_band]

    if bass_spec.size == 0:
        return []

    notes: List[Dict[str, object]] = []

    for idx, start in enumerate(beats):
        end = beats[idx + 1] if idx + 1 < len(beats) else start + 0.5
        if end <= start:
            end = start + 0.25

        mask = (frame_times >= start) & (frame_times < end)
        if not np.any(mask):
            frame_idx = int(np.argmin(np.abs(frame_times - start)))
            energy = bass_spec[:, frame_idx]
        else:
            energy = bass_spec[:, mask].mean(axis=1)

        peak_idx = int(np.argmax(energy))
        peak_energy = float(energy[peak_idx])
        if peak_energy <= 1e-4:
            continue

        freq = float(bass_freqs[peak_idx])
        midi = int(round(float(librosa.hz_to_midi(freq))))
        midi = max(24, min(72, midi))

        notes.append(
            {
                "time": float(start),
                "duration": float(end - start),
                "midi": midi,
                "velocity": 84,
            }
        )

    return _merge_adjacent_notes(notes)
