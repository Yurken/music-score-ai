import numpy as np

from src.chords import infer_chords_from_chroma


def _chord_block(root: int, intervals: list[int], frames: int) -> np.ndarray:
    block = np.zeros((12, frames), dtype=float)
    for semitone in intervals:
        block[(root + semitone) % 12, :] = 1.0
    return block


def test_infer_chords_from_artificial_chroma_sequence():
    # C (C-E-G) -> Am (A-C-E) -> G7 (G-B-D-F)
    c_major = _chord_block(root=0, intervals=[0, 4, 7], frames=4)
    a_minor = _chord_block(root=9, intervals=[0, 3, 7], frames=4)
    g7 = _chord_block(root=7, intervals=[0, 4, 7, 10], frames=4)

    chroma = np.concatenate([c_major, a_minor, g7], axis=1)
    frame_times = np.arange(chroma.shape[1], dtype=float) * 0.1

    chords = infer_chords_from_chroma(chroma, frame_times, min_segment_duration=0.2, smooth_window=1)

    labels = [event["label"] for event in chords]
    assert labels[:3] == ["C", "Am", "G7"]
    assert all(set(event.keys()) == {"start", "end", "label", "confidence"} for event in chords)
    assert all(0.0 <= float(event["confidence"]) <= 1.0 for event in chords)
