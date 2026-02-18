"""Audio I/O helpers.

All time values used by downstream modules are in seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.errors import InvalidInputError, MissingDependencyError

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav"}


def require_librosa():
    """Return the `librosa` module or raise a friendly dependency error."""
    try:
        import librosa  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "Missing dependency: librosa. Install with `pip install librosa soundfile`."
        ) from exc
    return librosa


def validate_audio_extension(filename: str) -> str:
    """Validate uploaded audio filename.

    Input:
    - filename: Original file name string.

    Output:
    - File extension string in lowercase (e.g. '.wav').

    Raises:
    - InvalidInputError if extension is not mp3/wav.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise InvalidInputError(
            f"Unsupported audio format: {suffix or '<none>'}. Allowed: mp3, wav"
        )
    return suffix


def load_audio(file_path: Path, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Load audio as mono waveform.

    Input:
    - file_path: Audio file path.
    - sr: Target sample rate.

    Output:
    - (y, sr):
      - y: 1D float32 numpy array waveform.
      - sr: Sample rate integer.

    Notes:
    - Time axis is represented by index / sr in seconds.
    """
    librosa = require_librosa()
    y, loaded_sr = librosa.load(str(file_path), sr=sr, mono=True)
    return y.astype(np.float32), int(loaded_sr)
