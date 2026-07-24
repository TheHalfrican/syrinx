"""settings.py — the ⚙ tab's persisted key/value store.

The module keeps a process-wide cache keyed on the resolved file path, so the
reload test proves values come off disk and not out of that cache.
"""

import importlib
import json

from syrinx_engine import settings


def test_value_returns_the_default_when_unset():
    assert settings.value("nope") is None
    assert settings.value("nope", 42) == 42


def test_set_value_round_trips():
    settings.set_value("seedvc_steps", 33)
    assert settings.value("seedvc_steps") == 33
    settings.set_value("vc_max_secs", 12.5)
    assert settings.value("vc_max_secs") == 12.5


def test_set_value_none_clears_the_key():
    settings.set_value("seedvc_steps", 33)
    settings.set_value("seedvc_steps", None)
    assert settings.value("seedvc_steps") is None
    assert settings.value("seedvc_steps", 25) == 25
    settings.set_value("never_set", None)  # clearing an absent key is fine


def test_all_values_is_a_copy():
    settings.set_value("a", 1)
    settings.set_value("b", "two")
    vals = settings.all_values()
    assert vals == {"a": 1, "b": "two"}
    vals["a"] = 999
    assert settings.value("a") == 1


def test_values_are_written_to_the_data_dir(isolated_env):
    settings.set_value("seedvc_steps", 7)
    on_disk = json.loads((isolated_env / "engine-settings.json").read_text())
    assert on_disk == {"seedvc_steps": 7}


def test_values_survive_a_module_reload(isolated_env):
    """A reload drops the in-process cache — the values must come back off disk."""
    settings.set_value("seedvc_steps", 9)
    settings.set_value("vc_max_secs", 60.0)

    reloaded = importlib.reload(settings)
    assert reloaded._CACHE == {}  # cache really was dropped
    assert reloaded.value("seedvc_steps") == 9
    assert reloaded.all_values() == {"seedvc_steps": 9, "vc_max_secs": 60.0}


def test_a_corrupt_settings_file_degrades_to_empty(isolated_env):
    (isolated_env / "engine-settings.json").write_text("{ not json")
    reloaded = importlib.reload(settings)
    assert reloaded.all_values() == {}


def test_an_unwritable_data_dir_is_logged_not_raised(monkeypatch, tmp_path):
    """A read-only data dir must not take the engine down mid-setting."""
    monkeypatch.setenv("SYRINX_DATA_DIR", str(tmp_path / "does-not-exist"))
    importlib.reload(settings)
    settings.set_value("seedvc_steps", 5)
    assert settings.value("seedvc_steps") == 5  # kept in memory for this run


def test_a_non_dict_settings_file_degrades_to_empty(isolated_env):
    (isolated_env / "engine-settings.json").write_text("[1, 2, 3]")
    reloaded = importlib.reload(settings)
    assert reloaded.all_values() == {}
