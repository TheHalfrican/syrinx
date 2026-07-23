"""Sentence-boundary text chunking for long TTS generations.

Ported from Voicebox's ``utils/chunked_tts.py``: long text is split at
natural boundaries (sentence end → clause → whitespace → hard cut, never
inside a ``[tag]``), synthesized per-chunk, and joined with a short
crossfade. Caps the synthesis sequence length — LuxTTS's flow-matching
memory grows steeply with target duration (an unchunked 2-minute text
OOM-killed the worker on the 15 GB dev box). Wired into every backend
(LuxTTS, Qwen, Kokoro); do the same in Chatterbox/TADA when those land.

Short text (≤ max chunk chars) is left untouched by the splitter, so
callers get a zero-overhead single-shot fast path.
"""

import asyncio
import os
import re

import numpy as np

# Default chunk size in characters (Voicebox's default).
DEFAULT_MAX_CHUNK_CHARS = 800

# Common abbreviations that should NOT be treated as sentence endings.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
        "inc", "ltd", "corp", "dept", "est", "approx", "vs", "etc",
        "e.g", "i.e", "a.m", "p.m", "u.s", "u.s.a", "u.k",
    }
)

# Paralinguistic tags (Chatterbox Turbo's ``[laugh]`` etc.) are atomic.
_PARA_TAG_RE = re.compile(r"\[[^\]]*\]")


def max_chunk_chars() -> int:
    """Chunk cap, overridable via SYRINX_TTS_CHUNK_CHARS."""
    try:
        return int(os.environ.get("SYRINX_TTS_CHUNK_CHARS", DEFAULT_MAX_CHUNK_CHARS))
    except ValueError:
        return DEFAULT_MAX_CHUNK_CHARS


def split_text_into_chunks(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Split *text* at natural boundaries into chunks of at most *max_chars*.

    Priority: sentence end (``.!?`` not after an abbreviation/decimal, plus
    CJK ``。！？``) → clause boundary (``;:,—``) → whitespace → hard cut.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        segment = remaining[:max_chars]
        split_pos = _find_last_sentence_end(segment)
        if split_pos == -1:
            split_pos = _find_last_clause_boundary(segment)
        if split_pos == -1:
            split_pos = segment.rfind(" ")
        if split_pos == -1:
            split_pos = _safe_hard_cut(segment, max_chars)

        chunk = remaining[: split_pos + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos + 1 :]

    return chunks


def _find_last_sentence_end(text: str) -> int:
    best = -1
    for m in re.finditer(r"[.!?](?:\s|$)", text):
        pos = m.start()
        if text[pos] == ".":
            # walk back over the preceding word to rule out abbreviations
            word_start = pos - 1
            while word_start >= 0 and text[word_start].isalpha():
                word_start -= 1
            word = text[word_start + 1 : pos].lower()
            if word in _ABBREVIATIONS:
                continue
            if word_start >= 0 and text[word_start].isdigit():  # decimal number
                continue
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    for m in re.finditer(r"[。！？]", text):
        if m.start() > best:
            best = m.start()
    return best


def _find_last_clause_boundary(text: str) -> int:
    best = -1
    for m in re.finditer(r"[;:,—](?:\s|$)", text):
        if _inside_bracket_tag(text, m.start()):
            continue
        best = m.start()
    return best


def _inside_bracket_tag(text: str, pos: int) -> bool:
    for m in _PARA_TAG_RE.finditer(text):
        if m.start() < pos < m.end():
            return True
    return False


def _safe_hard_cut(segment: str, max_chars: int) -> int:
    cut = max_chars - 1
    for m in _PARA_TAG_RE.finditer(segment):
        if m.start() < cut < m.end():
            return m.start() - 1 if m.start() > 0 else cut
    return cut


async def synthesize_chunked(gen_fn, text: str, *, log, label: str) -> tuple:
    """Shared chunk loop: split → per-chunk *gen_fn* in a thread → crossfade.

    *gen_fn(chunk_text)* is a sync callable returning ``(np.float32 audio,
    sample_rate)``; per-chunk trims belong inside it. Returns
    ``(pcm_bytes, rate)``.
    """
    chunks = split_text_into_chunks(text, max_chunk_chars())
    if len(chunks) <= 1:
        audio, rate = await asyncio.to_thread(gen_fn, text)
        return audio.tobytes(), rate
    log.info("%s: %d chars -> %d chunks", label, len(text), len(chunks))
    parts: list = []
    rate = 24_000
    for i, chunk in enumerate(chunks, 1):
        log.info("%s chunk %d/%d (%d chars)", label, i, len(chunks), len(chunk))
        audio, rate = await asyncio.to_thread(gen_fn, chunk)
        parts.append(audio)
    return crossfade_concat(parts, rate).tobytes(), rate


def trim_tts_output(
    audio: np.ndarray,
    sample_rate: int = 24_000,
    frame_ms: int = 20,
    silence_threshold_db: float = -40.0,
    min_silence_ms: int = 200,
    max_internal_silence_ms: int = 1000,
    fade_ms: int = 30,
) -> np.ndarray:
    """Trim trailing silence and post-silence hallucination from TTS output.

    Chatterbox sometimes produces ``[speech][silence][hallucinated noise]``.
    Cut at the first internal silence gap longer than *max_internal_silence_ms*,
    trim trailing silence, and apply a short cosine fade-out. Ported from
    Voicebox's ``utils/audio.py`` for per-chunk use with the Chatterbox engines.
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return audio

    n_frames = len(audio) // frame_len
    threshold_linear = 10 ** (silence_threshold_db / 20)
    framed = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(framed.astype(np.float64) ** 2, axis=1))
    is_speech = rms >= threshold_linear

    first_speech = 0
    for i, s in enumerate(is_speech):
        if s:
            first_speech = max(0, i - 1)  # keep 1 frame padding
            break

    # walk forward from first speech; cut at long internal silence gaps
    max_silence_frames = int(max_internal_silence_ms / frame_ms)
    consecutive_silence = 0
    cut_frame = n_frames
    for i in range(first_speech, n_frames):
        if is_speech[i]:
            consecutive_silence = 0
        else:
            consecutive_silence += 1
            if consecutive_silence >= max_silence_frames:
                cut_frame = i - consecutive_silence + 1
                break

    # trim trailing silence from the cut point, keeping a short tail
    min_silence_frames = int(min_silence_ms / frame_ms)
    end_frame = cut_frame
    while end_frame > first_speech and not is_speech[end_frame - 1]:
        end_frame -= 1
    end_frame = min(end_frame + min_silence_frames, cut_frame)

    start_sample = first_speech * frame_len
    end_sample = min(end_frame * frame_len, len(audio))
    trimmed = np.array(audio[start_sample:end_sample], dtype=np.float32, copy=True)

    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples > 0 and len(trimmed) > fade_samples:
        fade = np.cos(np.linspace(0, np.pi / 2, fade_samples, dtype=np.float32)) ** 2
        trimmed[-fade_samples:] *= fade
    return trimmed


def crossfade_concat(
    chunks: list[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = 50,
) -> np.ndarray:
    """Concatenate float32 mono chunks with a short crossfade (no clicks)."""
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    result = np.array(chunks[0], dtype=np.float32, copy=True)

    for chunk in chunks[1:]:
        if len(chunk) == 0:
            continue
        overlap = min(crossfade_samples, len(result), len(chunk))
        if overlap > 0:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            result[-overlap:] = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
            result = np.concatenate([result, chunk[overlap:]])
        else:
            result = np.concatenate([result, chunk])

    return result
