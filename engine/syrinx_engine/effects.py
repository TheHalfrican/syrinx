"""Audio post-processing effects — Spotify pedalboard DSP.

Effects are JSON-serializable chains (list of {type, enabled, params} dicts),
mirroring Voicebox's registry/preset model so chains stay portable. Only the
built-in presets are exposed for now; a user chain editor can reuse the same
registry later.

pedalboard is imported lazily: without it the engine still runs and effects
degrade to a no-op (with a logged warning).
"""

import logging

import numpy as np

log = logging.getLogger("syrinx.engine.effects")

# Param definitions kept for a future chain-editor UI; apply() only needs
# defaults. (type -> pedalboard class name, params -> defaults)
_REGISTRY = {
    "chorus": ("Chorus", {"rate_hz": 1.0, "depth": 0.5, "feedback": 0.0,
                          "centre_delay_ms": 7.0, "mix": 0.5}),
    "reverb": ("Reverb", {"room_size": 0.5, "damping": 0.5, "wet_level": 0.33,
                          "dry_level": 0.4, "width": 1.0}),
    "delay": ("Delay", {"delay_seconds": 0.3, "feedback": 0.3, "mix": 0.3}),
    "compressor": ("Compressor", {"threshold_db": -20.0, "ratio": 4.0,
                                  "attack_ms": 10.0, "release_ms": 100.0}),
    "gain": ("Gain", {"gain_db": 0.0}),
    "highpass": ("HighpassFilter", {"cutoff_frequency_hz": 80.0}),
    "lowpass": ("LowpassFilter", {"cutoff_frequency_hz": 8000.0}),
    "pitch_shift": ("PitchShift", {"semitones": 0.0}),
}

# Voicebox's built-in presets, verbatim chains (backend/utils/effects.py).
PRESETS = {
    "robotic": {
        "name": "Robotic",
        "description": "Metallic robotic voice (flanger with slow LFO and high feedback)",
        "chain": [
            {"type": "chorus", "params": {"rate_hz": 0.2, "depth": 1.0, "feedback": 0.35,
                                          "centre_delay_ms": 7.0, "mix": 0.5}},
        ],
    },
    "radio": {
        "name": "Radio",
        "description": "Thin AM-radio voice with band-pass filtering and light compression",
        "chain": [
            {"type": "highpass", "params": {"cutoff_frequency_hz": 300.0}},
            {"type": "lowpass", "params": {"cutoff_frequency_hz": 3500.0}},
            {"type": "compressor", "params": {"threshold_db": -15.0, "ratio": 6.0,
                                              "attack_ms": 5.0, "release_ms": 50.0}},
            {"type": "gain", "params": {"gain_db": 6.0}},
        ],
    },
    "echo_chamber": {
        "name": "Echo Chamber",
        "description": "Spacious reverb with trailing echo",
        "chain": [
            {"type": "reverb", "params": {"room_size": 0.85, "damping": 0.3,
                                          "wet_level": 0.45, "dry_level": 0.55, "width": 1.0}},
            {"type": "delay", "params": {"delay_seconds": 0.25, "feedback": 0.3, "mix": 0.2}},
        ],
    },
    "deep_voice": {
        "name": "Deep Voice",
        "description": "Lower pitch with added warmth",
        "chain": [
            {"type": "pitch_shift", "params": {"semitones": -3.0}},
            {"type": "lowpass", "params": {"cutoff_frequency_hz": 6000.0}},
            {"type": "compressor", "params": {"threshold_db": -18.0, "ratio": 3.0,
                                              "attack_ms": 10.0, "release_ms": 150.0}},
        ],
    },
}


def list_presets() -> list:
    """[{id, name, description}] for the UI dropdowns."""
    return [
        {"id": pid, "name": p["name"], "description": p["description"]}
        for pid, p in PRESETS.items()
    ]


def preset_name(preset_id: str) -> str:
    p = PRESETS.get(preset_id)
    return p["name"] if p else ""


def _build_board(chain: list):
    import pedalboard

    plugins = []
    for effect in chain:
        if not effect.get("enabled", True):
            continue
        cls_name, defaults = _REGISTRY[effect["type"]]
        params = {**defaults, **effect.get("params", {})}
        plugins.append(getattr(pedalboard, cls_name)(**params))
    return pedalboard.Pedalboard(plugins)


def apply_chain(pcm: bytes, sample_rate: int, chain: list) -> bytes:
    """Run float32 mono PCM through an effects chain; returns processed PCM."""
    if not chain or not pcm:
        return pcm
    try:
        board = _build_board(chain)
    except ImportError:
        log.warning("pedalboard not installed — effects skipped")
        return pcm
    audio = np.frombuffer(pcm, dtype=np.float32)
    processed = board(audio[np.newaxis, :], sample_rate)[0]
    return np.ascontiguousarray(processed, dtype=np.float32).tobytes()


def apply_preset(pcm: bytes, sample_rate: int, preset_id: str) -> bytes:
    """Apply a built-in preset by id; unknown/empty ids are a no-op."""
    p = PRESETS.get(preset_id)
    if not p:
        return pcm
    return apply_chain(pcm, sample_rate, p["chain"])


def load_wav(path: str) -> tuple:
    """Read a WAV as (float32 mono PCM bytes, sample_rate)."""
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    return np.ascontiguousarray(data, dtype=np.float32).tobytes(), int(rate)
