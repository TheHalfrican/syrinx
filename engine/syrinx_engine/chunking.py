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
