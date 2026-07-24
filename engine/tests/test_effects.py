"""effects.py — the registry, chain validation and user-preset CRUD.

_build_board / apply_chain are deliberately untested: they need pedalboard,
which is not part of the CI dependency contract (the DSP itself is verified
by ear on the GPU box).
"""

import numpy as np
import pytest

from syrinx_engine import effects
from syrinx_engine.effects import (
    EffectPresetStore,
    PRESETS,
    list_effects,
    list_presets,
    preset_name,
    resolve_preset,
    validate_chain,
)


def one_effect(eid=None):
    """A legit single-effect chain built from the registry's own defaults."""
    e = next(e for e in list_effects() if eid is None or e["id"] == eid)
    return [{"type": e["id"], "params": {p["name"]: p["default"] for p in e["params"]}}]


# --- load_wav ------------------------------------------------------------


def test_load_wav_returns_float32_mono_and_the_rate(make_wav):
    path = make_wav("tone.wav", secs=0.5, rate=16_000, amp=0.5)
    pcm, rate = effects.load_wav(str(path))
    assert rate == 16_000
    data = np.frombuffer(pcm, dtype=np.float32)
    assert data.size == 8_000
    assert 0.45 < float(np.abs(data).max()) <= 0.5


def test_load_wav_takes_the_left_channel_of_a_stereo_file(tmp_path):
    import soundfile as sf

    path = tmp_path / "stereo.wav"
    left = np.full(1000, 0.25, dtype=np.float32)
    right = np.full(1000, -0.75, dtype=np.float32)
    sf.write(str(path), np.stack([left, right], axis=1), 24_000, subtype="FLOAT")
    pcm, rate = effects.load_wav(str(path))
    data = np.frombuffer(pcm, dtype=np.float32)
    assert rate == 24_000 and data.size == 1000
    assert np.allclose(data, 0.25)


def test_load_wav_raises_on_an_unreadable_path(tmp_path):
    with pytest.raises(Exception):
        effects.load_wav(str(tmp_path / "nope.wav"))


# --- registry / validation ----------------------------------------------


def test_list_effects_exposes_params_as_an_ordered_list():
    defs = list_effects()
    assert {e["id"] for e in defs} == set(effects.REGISTRY)
    gain = next(e for e in defs if e["id"] == "gain")
    assert gain["params"][0]["name"] == "gain_db"
    assert {"default", "min", "max", "step", "description"} <= set(gain["params"][0])


def test_list_presets_puts_builtins_first():
    store = EffectPresetStore()
    store.create("Mine", "", one_effect())
    rows = list_presets(store)
    assert [r["builtin"] for r in rows[: len(PRESETS)]] == [True] * len(PRESETS)
    assert rows[-1]["name"] == "Mine"
    assert all(r["builtin"] for r in list_presets())


def test_validate_chain_accepts_a_real_chain():
    assert validate_chain(one_effect()) is None
    assert validate_chain([]) is None
    for p in PRESETS.values():
        assert validate_chain(p["chain"]) is None


@pytest.mark.parametrize(
    "chain, fragment",
    [
        ("not a list", "must be a list"),
        ([42], "must be a dict"),
        ([{"type": "warp_drive"}], "unknown effect type"),
        ([{"type": "gain", "params": []}], "params must be a dict"),
        ([{"type": "gain", "params": {"nope": 1}}], "unknown param"),
        ([{"type": "gain", "params": {"gain_db": "loud"}}], "must be a number"),
    ],
)
def test_validate_chain_explains_what_is_wrong(chain, fragment):
    err = validate_chain(chain)
    assert err is not None and fragment in err


# --- resolve / name ------------------------------------------------------


def test_resolve_preset_finds_builtins_and_user_presets():
    store = EffectPresetStore()
    builtin = resolve_preset("robotic")
    assert builtin["builtin"] is True and builtin["chain"]
    pid = store.create("Mine", "desc", one_effect())
    assert resolve_preset(pid, store)["name"] == "Mine"


def test_resolve_preset_of_an_unknown_id_is_none():
    assert resolve_preset("nope") is None
    assert resolve_preset("nope", EffectPresetStore()) is None
    assert resolve_preset("") is None


def test_preset_name_is_empty_for_unknown_ids():
    assert preset_name("robotic") == "Robotic"
    assert preset_name("nope") == ""


# --- user preset CRUD ----------------------------------------------------


def test_create_list_get_update_delete():
    store = EffectPresetStore()
    assert store.list() == []

    pid = store.create("My Chain", "notes", one_effect("reverb"))
    assert pid
    got = store.get(pid)
    assert got["name"] == "My Chain"
    assert got["description"] == "notes"
    assert got["chain"][0]["type"] == "reverb"
    assert got["builtin"] is False
    assert [p["id"] for p in store.list()] == [pid]

    assert store.update(pid, "Renamed", "new notes", one_effect("delay")) is True
    assert store.get(pid)["name"] == "Renamed"
    assert store.get(pid)["chain"][0]["type"] == "delay"

    assert store.delete(pid) is True
    assert store.get(pid) is None
    assert store.delete(pid) is False


def test_create_refuses_blank_names_builtin_names_and_bad_chains():
    store = EffectPresetStore()
    assert store.create("  ", "", one_effect()) == ""
    assert store.create("Robotic", "", one_effect()) == ""  # collides with a builtin
    assert store.create("robotic", "", one_effect()) == ""  # case-insensitively too
    assert store.create("Fine", "", [{"type": "warp_drive"}]) == ""


def test_create_refuses_duplicate_user_names():
    store = EffectPresetStore()
    assert store.create("Twice", "", one_effect())
    assert store.create("Twice", "", one_effect()) == ""


def test_update_refuses_bad_chains_blank_names_and_unknown_ids():
    store = EffectPresetStore()
    pid = store.create("Keep", "", one_effect())
    assert store.update(pid, "Keep", "", [{"type": "warp_drive"}]) is False
    assert store.update(pid, "   ", "", one_effect()) is False
    assert store.update("nope", "Whatever", "", one_effect()) is False
    assert store.get(pid)["name"] == "Keep"


def test_update_to_a_taken_name_fails_on_the_unique_index():
    store = EffectPresetStore()
    store.create("First", "", one_effect())
    second = store.create("Second", "", one_effect())
    assert store.update(second, "First", "", one_effect()) is False


def test_apply_preset_with_an_unknown_id_is_a_no_op():
    pcm = np.zeros(100, dtype=np.float32).tobytes()
    assert effects.apply_preset(pcm, 24_000, "nope") is pcm


def test_apply_chain_with_an_empty_chain_or_empty_pcm_is_a_no_op():
    pcm = np.zeros(100, dtype=np.float32).tobytes()
    assert effects.apply_chain(pcm, 24_000, []) is pcm
    assert effects.apply_chain(b"", 24_000, one_effect()) == b""
