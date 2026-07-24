"""audio.py — the pure-numpy half (playback itself needs a PipeWire sink)."""

import numpy as np

from syrinx_engine import audio


def _pcm(samples):
    return np.asarray(samples, dtype=np.float32).tobytes()


def test_duration_of_counts_four_bytes_per_sample():
    assert audio.duration_of(_pcm(np.zeros(24_000)), 24_000) == 1.0
    assert audio.duration_of(_pcm(np.zeros(12_000)), 24_000) == 0.5


def test_duration_of_zero_rate_is_not_a_zero_division():
    assert audio.duration_of(_pcm(np.zeros(100)), 0) == 0.0


def test_envelope_returns_exactly_n_bars():
    pcm = _pcm(np.sin(np.linspace(0, 100, 10_000)))
    assert len(audio.envelope(pcm)) == 300
    assert len(audio.envelope(pcm, n=17)) == 17


def test_envelope_is_normalized_to_peak_one():
    # a quiet signal still tops out at 1.0 — the bars are relative, not absolute
    bars = audio.envelope(_pcm(np.sin(np.linspace(0, 50, 5_000)) * 0.01), n=64)
    assert max(bars) == 1.0
    assert all(0.0 <= b <= 1.0 for b in bars)


def test_envelope_of_silence_and_of_nothing_is_all_zeros():
    assert audio.envelope(_pcm(np.zeros(5_000)), n=32) == [0.0] * 32
    assert audio.envelope(b"", n=32) == [0.0] * 32


def test_envelope_handles_fewer_samples_than_bars():
    # more bars than samples: the empty buckets report 0.0 rather than crashing
    bars = audio.envelope(_pcm(np.linspace(0.1, 1.0, 10)), n=50)
    assert len(bars) == 50
    assert max(bars) == 1.0
