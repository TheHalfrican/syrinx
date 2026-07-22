"""Audio post-processing effects — Spotify pedalboard DSP.

Effects are JSON-serializable chains (list of {type, enabled, params} dicts),
mirroring Voicebox's registry/preset model so chains stay portable. Only the
built-in presets are exposed for now; a user chain editor can reuse the same
registry later.

pedalboard is imported lazily: without it the engine still runs and effects
degrade to a no-op (with a logged warning).
"""

import json
import logging
import sqlite3
import time
import uuid

import numpy as np

from .profiles import _data_dir

log = logging.getLogger("syrinx.engine.effects")

# Full effect definitions (Voicebox's EFFECT_REGISTRY, verbatim ranges):
# type -> {cls: pedalboard class name, label, description, params:
#          {name: {default, min, max, step, description}}}
REGISTRY = {
    "chorus": {
        "cls": "Chorus",
        "label": "Chorus / Flanger",
        "description": "Flanger-style modulation with short delays.",
        "params": {
            "rate_hz": {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01, "description": "LFO speed (Hz)"},
            "depth": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "description": "Modulation depth"},
            "feedback": {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01, "description": "Feedback amount"},
            "centre_delay_ms": {"default": 7.0, "min": 0.5, "max": 50.0, "step": 0.5, "description": "Centre delay (ms)"},
            "mix": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "description": "Wet/dry mix"},
        },
    },
    "reverb": {
        "cls": "Reverb",
        "label": "Reverb",
        "description": "Room reverberation.",
        "params": {
            "room_size": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "description": "Room size"},
            "damping": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "description": "High frequency damping"},
            "wet_level": {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01, "description": "Wet level"},
            "dry_level": {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "description": "Dry level"},
            "width": {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "description": "Stereo width"},
        },
    },
    "delay": {
        "cls": "Delay",
        "label": "Delay",
        "description": "Echo / delay line.",
        "params": {
            "delay_seconds": {"default": 0.3, "min": 0.01, "max": 2.0, "step": 0.01, "description": "Delay time (s)"},
            "feedback": {"default": 0.3, "min": 0.0, "max": 0.95, "step": 0.01, "description": "Feedback amount"},
            "mix": {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "description": "Wet/dry mix"},
        },
    },
    "compressor": {
        "cls": "Compressor",
        "label": "Compressor",
        "description": "Dynamic range compression for consistent loudness.",
        "params": {
            "threshold_db": {"default": -20.0, "min": -60.0, "max": 0.0, "step": 0.5, "description": "Threshold (dB)"},
            "ratio": {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1, "description": "Compression ratio"},
            "attack_ms": {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1, "description": "Attack time (ms)"},
            "release_ms": {"default": 100.0, "min": 10.0, "max": 1000.0, "step": 1.0, "description": "Release time (ms)"},
        },
    },
    "gain": {
        "cls": "Gain",
        "label": "Gain",
        "description": "Volume adjustment in decibels.",
        "params": {
            "gain_db": {"default": 0.0, "min": -40.0, "max": 40.0, "step": 0.5, "description": "Gain (dB)"},
        },
    },
    "highpass": {
        "cls": "HighpassFilter",
        "label": "High-Pass Filter",
        "description": "Removes frequencies below the cutoff.",
        "params": {
            "cutoff_frequency_hz": {"default": 80.0, "min": 20.0, "max": 8000.0, "step": 1.0, "description": "Cutoff frequency (Hz)"},
        },
    },
    "lowpass": {
        "cls": "LowpassFilter",
        "label": "Low-Pass Filter",
        "description": "Removes frequencies above the cutoff.",
        "params": {
            "cutoff_frequency_hz": {"default": 8000.0, "min": 200.0, "max": 20000.0, "step": 1.0, "description": "Cutoff frequency (Hz)"},
        },
    },
    "pitch_shift": {
        "cls": "PitchShift",
        "label": "Pitch Shift",
        "description": "Shift pitch up or down by semitones.",
        "params": {
            "semitones": {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5, "description": "Semitones to shift"},
        },
    },
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


def list_presets(store: "EffectPresetStore | None" = None) -> list:
    """[{id, name, description, builtin}] — builtins first, then user presets."""
    out = [
        {"id": pid, "name": p["name"], "description": p["description"], "builtin": True}
        for pid, p in PRESETS.items()
    ]
    if store is not None:
        out += [
            {"id": p["id"], "name": p["name"], "description": p["description"], "builtin": False}
            for p in store.list()
        ]
    return out


def list_effects() -> list:
    """Effect definitions for the chain-editor UI (params as an ordered list)."""
    return [
        {
            "id": eid,
            "label": e["label"],
            "description": e["description"],
            "params": [
                {"name": name, **meta} for name, meta in e["params"].items()
            ],
        }
        for eid, e in REGISTRY.items()
    ]


def validate_chain(chain) -> str | None:
    """None if the chain is valid, else an error message (Voicebox's rules)."""
    if not isinstance(chain, list):
        return "chain must be a list"
    for i, effect in enumerate(chain):
        if not isinstance(effect, dict):
            return f"effect at index {i} must be a dict"
        etype = effect.get("type")
        if etype not in REGISTRY:
            return f"unknown effect type {etype!r} at index {i}"
        params = effect.get("params", {})
        if not isinstance(params, dict):
            return f"effect {etype!r}: params must be a dict"
        for name, value in params.items():
            meta = REGISTRY[etype]["params"].get(name)
            if meta is None:
                return f"effect {etype!r}: unknown param {name!r}"
            if not isinstance(value, (int, float)):
                return f"effect {etype!r}: param {name!r} must be a number"
    return None


class EffectPresetStore:
    """User effect presets — named chains in SQLite (table ``effect_presets``).

    Built-ins live in code (PRESETS) and cannot be edited or deleted;
    user presets are full CRUD, names unique across both.
    """

    def __init__(self) -> None:
        self._dir = _data_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db = str(self._dir / "syrinx.db")
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS effect_presets(
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT DEFAULT '',
                    chain TEXT NOT NULL,
                    created_at REAL
                );
                """
            )

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._db)
        c.row_factory = sqlite3.Row
        return c

    def _row(self, r: sqlite3.Row) -> dict:
        return {
            "id": r["id"],
            "name": r["name"],
            "description": r["description"] or "",
            "chain": json.loads(r["chain"]),
            "builtin": False,
        }

    def list(self) -> list:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM effect_presets ORDER BY name").fetchall()
            return [self._row(r) for r in rows]

    def get(self, pid: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM effect_presets WHERE id=?", (pid,)).fetchone()
            return self._row(r) if r else None

    def create(self, name: str, description: str, chain: list) -> str:
        """New preset id, or "" on invalid chain / duplicate / builtin name."""
        name = name.strip()
        if not name or any(p["name"].lower() == name.lower() for p in PRESETS.values()):
            return ""
        if validate_chain(chain) is not None:
            return ""
        pid = uuid.uuid4().hex[:12]
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO effect_presets(id,name,description,chain,created_at) "
                    "VALUES(?,?,?,?,?)",
                    (pid, name, description, json.dumps(chain), time.time()),
                )
        except sqlite3.IntegrityError:
            return ""
        return pid

    def update(self, pid: str, name: str, description: str, chain: list) -> bool:
        if validate_chain(chain) is not None or not name.strip():
            return False
        try:
            with self._conn() as c:
                cur = c.execute(
                    "UPDATE effect_presets SET name=?, description=?, chain=? WHERE id=?",
                    (name.strip(), description, json.dumps(chain), pid),
                )
                return cur.rowcount > 0
        except sqlite3.IntegrityError:
            return False

    def delete(self, pid: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM effect_presets WHERE id=?", (pid,)).rowcount > 0


def resolve_preset(preset_id: str, store: "EffectPresetStore | None" = None) -> dict | None:
    """Preset dict for a builtin OR user preset id (None if unknown)."""
    p = PRESETS.get(preset_id)
    if p is not None:
        return {"id": preset_id, "name": p["name"], "description": p["description"],
                "chain": p["chain"], "builtin": True}
    return store.get(preset_id) if store is not None else None


def preset_name(preset_id: str, store: "EffectPresetStore | None" = None) -> str:
    p = resolve_preset(preset_id, store)
    return p["name"] if p else ""


def _build_board(chain: list):
    import pedalboard

    plugins = []
    for effect in chain:
        if not effect.get("enabled", True):
            continue
        spec = REGISTRY[effect["type"]]
        defaults = {name: meta["default"] for name, meta in spec["params"].items()}
        params = {**defaults, **effect.get("params", {})}
        # clamp to the registry ranges so a hand-edited chain can't explode
        for name, meta in spec["params"].items():
            params[name] = max(meta["min"], min(meta["max"], params[name]))
        plugins.append(getattr(pedalboard, spec["cls"])(**params))
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


def apply_preset(
    pcm: bytes, sample_rate: int, preset_id: str,
    store: "EffectPresetStore | None" = None,
) -> bytes:
    """Apply a builtin or user preset by id; unknown/empty ids are a no-op."""
    p = resolve_preset(preset_id, store)
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
