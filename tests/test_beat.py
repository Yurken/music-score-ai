from src.beat import infer_downbeats


def test_infer_downbeats_every_4_beats():
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    downbeats = infer_downbeats(beats, meter=4)
    assert downbeats == [0.0, 2.0]
