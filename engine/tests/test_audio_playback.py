"""audio.play — the block loop, driven against a fake PortAudio stream.

sounddevice isn't part of the CI dependency contract (and a real stream needs
a PipeWire sink), so a stand-in module goes into sys.modules: the loop, the
stop/pause/seek controls, the volume ramp and the level/progress callbacks are
all engine logic and get tested for real.
"""

import asyncio
import sys
import types

import numpy as np
import pytest

from syrinx_engine import audio


@pytest.fixture
def sd(fake_sd):
    return fake_sd


class Ctl:
    def __init__(self, stop=False, paused=False, seek=None):
        self.stop = stop
        self.paused = paused
        self.seek = seek


def pcm(n, val=0.5):
    return np.full(n, val, dtype=np.float32).tobytes()


# --- the ordinary path ---------------------------------------------------


def test_play_writes_every_frame_and_reports_level_and_progress(sd):
    levels, progress = [], []
    asyncio.run(audio.play(pcm(4096), 24_000,
                           on_level=levels.append, on_progress=progress.append))
    stream = sd.made[0]
    assert stream.started and stream.stopped and stream.closed
    assert stream.frames == 4096
    assert progress[-1] == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in progress)
    assert levels and all(abs(v - 0.5) < 1e-5 for v in levels)  # RMS of a DC 0.5 block


def test_empty_pcm_never_opens_a_stream(sd):
    asyncio.run(audio.play(b"", 24_000))
    assert sd.made == []


def test_volume_is_applied_per_block(sd):
    asyncio.run(audio.play(pcm(2048), 24_000, volume=lambda: 0.25))
    written = np.concatenate(sd.made[0].written)
    assert np.allclose(written, 0.125)


def test_volume_out_of_range_is_clamped(sd):
    asyncio.run(audio.play(pcm(1024), 24_000, volume=lambda: 9.0))
    assert np.allclose(np.concatenate(sd.made[0].written), 0.5)
    asyncio.run(audio.play(pcm(1024), 24_000, volume=lambda: -3.0))
    assert np.allclose(np.concatenate(sd.made[1].written), 0.0)


# --- the controls --------------------------------------------------------


def test_stop_ends_playback_early_but_still_tears_the_stream_down(sd):
    ctl = Ctl(stop=True)
    asyncio.run(audio.play(pcm(48_000), 24_000, ctl))
    stream = sd.made[0]
    assert stream.frames == 0
    assert stream.stopped and stream.closed


def test_a_pause_holds_the_loop_until_it_is_released(sd):
    class Paused(Ctl):
        reads = 0

        @property
        def paused(self):
            # unpause after a few polls — the loop must not have written yet
            Paused.reads += 1
            return Paused.reads <= 3

        @paused.setter
        def paused(self, _v):
            pass

    asyncio.run(audio.play(pcm(2048), 24_000, Paused()))
    assert Paused.reads > 3
    assert sd.made[0].frames == 2048


def test_seek_jumps_the_read_position(sd):
    ctl = Ctl(seek=0.5)
    asyncio.run(audio.play(pcm(4096), 24_000, ctl))
    assert sd.made[0].frames == 2048  # started halfway in
    assert ctl.seek is None  # consumed, not re-applied every block


def test_a_stream_that_will_not_open_is_a_warning_not_a_crash(monkeypatch):
    def boom(**_kw):
        raise RuntimeError("no device")

    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(OutputStream=boom))
    asyncio.run(audio.play(pcm(1024), 24_000))  # returns quietly


def test_no_sounddevice_at_all_skips_playback(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    asyncio.run(audio.play(pcm(1024), 24_000))


# --- cancellation --------------------------------------------------------


def test_cancelling_stops_the_thread_instead_of_closing_the_stream_under_it(sd):
    """Cancel must flag ctl and drain the worker — closing a PortAudio stream
    from another thread while write() is blocked is a use-after-free."""

    async def go():
        ctl = Ctl()
        task = asyncio.create_task(audio.play(pcm(24_000 * 20), 24_000, ctl))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return ctl

    ctl = asyncio.run(go())
    assert ctl.stop is True
    stream = sd.made[0]
    assert stream.stopped and stream.closed  # torn down by its owning thread


def test_play_without_a_ctl_makes_its_own(sd):
    asyncio.run(audio.play(pcm(1024), 24_000))
    assert sd.made[0].frames == 1024
