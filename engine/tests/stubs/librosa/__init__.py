"""Minimal librosa stand-in — the worker only loads mono wavs and resamples."""

import numpy as np
import soundfile as sf


def load(path, sr=None, mono=True):
    data, rate = sf.read(path, dtype="float32", always_2d=False)
    if mono and getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(rate)


def resample(y, orig_sr=None, target_sr=None):
    if not orig_sr or not target_sr or orig_sr == target_sr:
        return np.asarray(y, dtype=np.float32)
    n = int(round(len(y) * target_sr / orig_sr))
    idx = np.linspace(0, len(y) - 1, n)
    return np.interp(idx, np.arange(len(y)), y).astype(np.float32)
