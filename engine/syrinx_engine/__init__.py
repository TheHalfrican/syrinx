"""Syrinx engine — the ML inference service.

Loads Qwen3-TTS (torch) and whisper.cpp (STT), plays audio via PipeWire, and
exposes everything on the D-Bus session bus as ``sh.syrinx.Engine1``.
"""

__version__ = "0.1.0"
