"""seedvc_worker.py — the JSON line protocol, exercised as a subprocess.

The worker dup2's stdout at import time (to keep HF download chatter off the
protocol channel), so importing it in-process would hijack pytest's stdout.
It is therefore driven exactly as the backend drives it: spawn it, write one
request line, read one reply line.

seed_vc/librosa/torch come from tests/stubs — the real seed-vc is GPL and
lives only in .venv-seedvc, and the stubs pin the numbers so the peak guard
is actually observable.
"""

import json
import queue
import subprocess
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

WORKER = Path(__file__).resolve().parents[1] / "syrinx_engine" / "seedvc_worker.py"
STUBS = Path(__file__).resolve().parent / "stubs"
EXPECTED_RATE = 22050  # the stub state's .sr


def write_wav(path, secs=0.5, rate=16_000):
    n = int(secs * rate)
    t = np.linspace(0, secs, n, endpoint=False)
    pcm = (np.sin(2 * np.pi * 220 * t) * 0.4 * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def readline(proc, timeout=60.0):
    """Read one protocol line, or fail loudly instead of hanging the suite."""
    box = queue.Queue(maxsize=1)
    threading.Thread(target=lambda: box.put(proc.stdout.readline()), daemon=True).start()
    try:
        return box.get(timeout=timeout)
    except queue.Empty:
        pytest.fail(f"seedvc worker produced no reply within {timeout}s")


@pytest.fixture
def worker(tmp_path):
    """The worker as a subprocess, stubs on PYTHONPATH, always reaped."""
    log = tmp_path / "worker.err"
    with open(log, "wb") as errf:
        proc = subprocess.Popen(
            [sys.executable, str(WORKER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf,
            cwd=str(tmp_path), text=True, bufsize=1,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(STUBS),
                 "HOME": str(tmp_path), "TMPDIR": str(tmp_path)},
        )
    try:
        yield proc
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def request(proc, req):
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    return json.loads(readline(proc))


def test_a_conversion_request_replies_with_peak_safe_float32(worker, tmp_path):
    src = write_wav(tmp_path / "src.wav", secs=0.5)
    tgt = write_wav(tmp_path / "tgt.wav", secs=0.5)

    reply = request(worker, {"id": 1, "source": str(src), "target": str(tgt), "steps": 2})

    assert reply["ok"] is True
    assert reply["id"] == 1
    assert reply["rate"] == EXPECTED_RATE

    audio = np.fromfile(reply["raw"], dtype=np.float32)
    assert audio.size > 0
    peak = float(np.abs(audio).max())
    # the stub hands back a 1.7 full-scale block; every reply path is
    # peak-normalized because playback and the saved WAV hard-clip past ±1.0
    assert 0.9 < peak <= 0.99 + 1e-6


def test_a_missing_source_replies_with_an_error_not_a_crash(worker, tmp_path):
    tgt = write_wav(tmp_path / "tgt.wav", secs=0.2)

    reply = request(worker, {"id": 7, "source": str(tmp_path / "gone.wav"),
                             "target": str(tgt), "steps": 2})

    assert reply["ok"] is False
    assert reply["id"] == 7
    assert reply["error"]
    assert worker.poll() is None  # the worker survives to serve the next request


def test_the_worker_keeps_serving_after_a_failure(worker, tmp_path):
    src = write_wav(tmp_path / "src.wav", secs=0.3)
    tgt = write_wav(tmp_path / "tgt.wav", secs=0.3)

    assert request(worker, {"id": 1, "source": "/nope.wav", "target": str(tgt)})["ok"] is False
    ok = request(worker, {"id": 2, "source": str(src), "target": str(tgt), "steps": 2})
    assert ok["ok"] is True and ok["id"] == 2


def test_blank_lines_are_ignored(worker, tmp_path):
    src = write_wav(tmp_path / "src.wav", secs=0.2)
    tgt = write_wav(tmp_path / "tgt.wav", secs=0.2)
    worker.stdin.write("\n   \n")
    worker.stdin.flush()
    assert request(worker, {"id": 3, "source": str(src), "target": str(tgt)})["ok"] is True
