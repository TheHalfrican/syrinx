"""Tiny persisted key-value settings for the engine (Settings tab knobs).

Values live in $SYRINX_DATA_DIR/engine-settings.json and win over their
environment-variable fallbacks — the env vars stay as deploy-time defaults
(and keep older run scripts working), the Settings tab writes here.

Module-level singleton: backends read through `value()` without any
plumbing, the D-Bus service writes through `set_value()` — one process,
one store.
"""

import json
import logging
from pathlib import Path

from .profiles import _data_dir

log = logging.getLogger("syrinx.engine.settings")

_FILE = None
_CACHE: dict = {}


def _path() -> Path:
    return _data_dir() / "engine-settings.json"


def _load() -> None:
    global _FILE, _CACHE
    p = _path()
    if _FILE == p:
        return
    _FILE = p
    try:
        _CACHE = json.loads(p.read_text())
        if not isinstance(_CACHE, dict):
            _CACHE = {}
    except Exception:  # noqa: BLE001
        _CACHE = {}


def value(key: str, default=None):
    """Current value for *key* (None-safe; caller applies env fallbacks)."""
    _load()
    return _CACHE.get(key, default)


def set_value(key: str, val) -> None:
    _load()
    if val is None:
        _CACHE.pop(key, None)
    else:
        _CACHE[key] = val
    try:
        _path().write_text(json.dumps(_CACHE, indent=2))
    except Exception:  # noqa: BLE001
        log.exception("save engine-settings.json failed")


def all_values() -> dict:
    _load()
    return dict(_CACHE)
