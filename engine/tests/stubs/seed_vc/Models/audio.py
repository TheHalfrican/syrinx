"""Fake seed_vc.Models.audio — the keyword set the worker's _audio_data uses."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AudioData:
    samples: Any
    mel_chunks: Any = None
    duration: float = 0.0
    samples_count: int = 0
    sample_rate: int = 0
    metadata: Any = None
