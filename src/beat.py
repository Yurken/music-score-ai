"""Beat and tempo estimation.

All returned times are seconds.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.audio_io import require_librosa


def extract_tempo_beats(y: np.ndarray, sr: int) -> Dict[str, object]:
    """Estimate tempo and beat timestamps.

    Input:
    - y: Mono waveform (1D numpy array).
    - sr: Sample rate.

    Output:
    - dict with:
      - tempo: float BPM.
      - beats: list[float], beat times in seconds.
    """
    librosa = require_librosa()
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    return {"tempo": float(tempo), "beats": [float(t) for t in beat_times]}


def infer_downbeats(beats: List[float], meter: int = 4) -> List[float]:
    """Infer downbeats from beat grid using fixed meter.

    Input:
    - beats: list of beat timestamps in seconds.
    - meter: beats per bar (default=4).

    Output:
    - list[float], downbeat timestamps in seconds.
    """
    if meter <= 0:
        raise ValueError("meter must be positive")
    return [float(t) for idx, t in enumerate(beats) if idx % meter == 0]
