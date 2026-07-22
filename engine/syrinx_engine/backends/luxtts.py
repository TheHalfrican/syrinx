"""LuxTTS backend — proxies to an isolated worker process (.venv-luxtts).

Zipvoice/LuxTTS pins conflicting deps (transformers<=4.57, torchaudio 2.11), so
it lives in a separate venv rather than polluting the main engine. This backend
spawns `luxtts_worker.py` under that venv's python and talks to it over a
stdin/stdout JSON line protocol; audio returns as a raw float32 PCM temp file.
CPU-friendly zero-shot cloning — the one cloning engine usable on this box.
"""

import asyncio
import json
import logging
from pathlib import Path

from . import detect_device

log = logging.getLogger("syrinx.engine.tts.luxtts")

_HERE = Path(__file__).resolve()
_ENGINE_DIR = _HERE.parents[2]  # .../engine
_LUX_PY = _ENGINE_DIR / ".venv-luxtts" / "bin" / "python"
_WORKER = _HERE.parents[1] / "luxtts_worker.py"  # .../engine/syrinx_engine/luxtts_worker.py
_STDERR_LOG = Path.home() / ".cache" / "syrinx-luxtts.log"


class LuxTTSBackend:
    supports_cloning = True

    def __init__(self) -> None:
        self.device = detect_device()
        self.model_size = "default"
        self._proc = None
        self._lock = asyncio.Lock()
        self._req_id = 0

    async def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        if not _LUX_PY.exists():
            raise RuntimeError("LuxTTS not installed (.venv-luxtts missing)")
        _STDERR_LOG.parent.mkdir(parents=True, exist_ok=True)
        errfile = open(_STDERR_LOG, "ab")
        self._proc = await asyncio.create_subprocess_exec(
            str(_LUX_PY), str(_WORKER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=errfile,
        )
        log.info("LuxTTS worker started (pid %s)", self._proc.pid)

    async def load(self) -> None:
        await self._ensure_worker()

    async def list_voices(self) -> list:
        return []  # cloning-only, no presets

    async def synthesize(self, text: str, voice_id: str) -> tuple:
        raise ValueError("LuxTTS has no preset voices")

    async def synthesize_profile(self, profile, text: str, instruct: str = "") -> tuple:
        if not profile.samples:
            raise ValueError(f"profile {profile.id} has no reference samples")
        sample = profile.samples[0].audio_path
        async with self._lock:
            await self._ensure_worker()
            self._req_id += 1
            rid = self._req_id
            payload = json.dumps({"id": rid, "sample": str(sample), "text": text}) + "\n"
            self._proc.stdin.write(payload.encode())
            await self._proc.stdin.drain()
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    self._proc = None
                    raise RuntimeError("LuxTTS worker exited (see ~/.cache/syrinx-luxtts.log)")
                try:
                    resp = json.loads(line.decode())
                    break
                except json.JSONDecodeError:
                    # stray library print that escaped the worker's redirect
                    log.debug("luxtts worker noise: %s", line.decode(errors="replace").rstrip())
            if not resp.get("ok"):
                raise RuntimeError(f"LuxTTS synth failed: {resp.get('error')}")
            raw_path = Path(resp["raw"])
            rate = int(resp["rate"])
        pcm = raw_path.read_bytes()
        raw_path.unlink(missing_ok=True)
        log.info("synthesize_profile [luxtts] (%s): %r", profile.id, text[:60])
        return pcm, rate

    def invalidate_profile(self, profile_id: str) -> None:
        pass  # the worker caches prompts by sample path
