"""chunking.py — text splitting, crossfade joins, trailing-silence trim."""

import asyncio

import numpy as np
import pytest

from syrinx_engine import chunking
from syrinx_engine.chunking import (
    crossfade_concat,
    max_chunk_chars,
    split_text_into_chunks,
    trim_tts_output,
)

RATE = 24_000


# --- split_text_into_chunks ---------------------------------------------


def test_short_text_is_returned_untouched():
    # the zero-overhead fast path: no splitting work for ordinary sentences
    assert split_text_into_chunks("Hello there.") == ["Hello there."]
    assert split_text_into_chunks("  padded  ") == ["padded"]


def test_empty_text_yields_no_chunks():
    assert split_text_into_chunks("") == []
    assert split_text_into_chunks("   \n  ") == []


def test_long_text_respects_the_cap_and_keeps_every_word():
    text = " ".join(f"Sentence number {i} runs on for a little while." for i in range(60))
    chunks = split_text_into_chunks(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_splits_land_on_sentence_ends_when_it_can():
    text = " ".join(["Alpha bravo charlie delta echo."] * 40)
    chunks = split_text_into_chunks(text, max_chars=120)
    assert len(chunks) > 1
    assert all(c.endswith(".") for c in chunks)


def test_a_period_after_a_digit_is_a_decimal_not_a_sentence_end():
    # "pi is 3.14" must not split mid-number — same rule keeps "sentence 39."
    # from ending a chunk, which is why the sentence tests use words
    text = "Value 3.14 holds. " + "Padding words keep this text over the cap. " * 5
    chunks = split_text_into_chunks(text, max_chars=90)
    assert not any(c.endswith("3.") for c in chunks)


def test_abbreviations_are_not_sentence_ends():
    # "Dr." must not end a chunk — the real boundary is the sentence after it
    text = ("Dr. Smith met Mr. Jones at St. Marks and they talked for ages about "
            "nothing. ") + "Filler words follow here to push past the cap. " * 6
    chunks = split_text_into_chunks(text, max_chars=110)
    assert not any(c.endswith(("Dr.", "Mr.", "St.")) for c in chunks)


def test_falls_back_to_clause_then_whitespace_then_a_hard_cut():
    clause = ("alpha bravo charlie delta echo foxtrot golf hotel india, "
              "juliett kilo lima mike november oscar papa quebec romeo sierra")
    assert split_text_into_chunks(clause, max_chars=70)[0].endswith(",")

    spaced = " ".join(["word"] * 40)  # no punctuation at all
    assert all(len(c) <= 50 for c in split_text_into_chunks(spaced, max_chars=50))

    solid = "x" * 300  # no boundary of any kind -> hard cut
    chunks = split_text_into_chunks(solid, max_chars=100)
    assert [len(c) for c in chunks] == [100, 100, 100]
    assert "".join(chunks) == solid


def test_bracket_tags_are_never_split_across_a_boundary():
    """A '.' inside a [tag] is not a sentence end — cutting there would hand
    the engine half a paralinguistic tag."""
    text = ("This is the first sentence. Now a tag [uh. huh] follows, and then "
            "a good deal more text arrives to force a second chunk.")
    chunks = split_text_into_chunks(text, max_chars=50)
    assert len(chunks) > 1
    for c in chunks:
        assert c.count("[") == c.count("]")
    assert sum("[uh. huh]" in c for c in chunks) == 1


def test_a_comma_inside_a_tag_is_not_a_clause_boundary():
    """Same protection one rung down: with no sentence end in reach the
    splitter falls to clauses, and the comma in "[uh, huh]" must not count."""
    text = "alpha bravo, charlie delta echo foxtrot [uh, huh] golf hotel india juliett kilo"
    chunks = split_text_into_chunks(text, max_chars=50)
    assert chunks[0] == "alpha bravo,"  # the earlier, real comma
    assert sum("[uh, huh]" in c for c in chunks) == 1


def test_a_hard_cut_backs_off_to_before_the_tag():
    """No sentence end, no clause, no whitespace — the hard cut still refuses
    to land inside a [tag], so it cuts just before it instead."""
    text = "a" * 30 + "[laugh]" + "b" * 30
    chunks = split_text_into_chunks(text, max_chars=37)
    assert chunks[0] == "a" * 30
    assert chunks[1].startswith("[laugh]")


def test_a_digit_before_the_period_reads_as_a_numbered_item_not_a_sentence():
    # "Step 39." is treated like a decimal, so the split falls through to
    # whitespace — documented here because it shapes every other split test
    # the differential: identical text, digits vs a word before the period
    digits = split_text_into_chunks("Step 39. " * 20, max_chars=60)
    words = split_text_into_chunks("Step nine. " * 20, max_chars=60)
    assert all(c.endswith(".") for c in words)  # real sentence ends
    assert digits[0].endswith("Step")  # fell through to a whitespace cut


def test_cjk_sentence_enders_split_too():
    text = "".join(f"这是第{i}个句子。" for i in range(40))
    chunks = split_text_into_chunks(text, max_chars=100)
    assert len(chunks) > 1
    assert all(c.endswith("。") for c in chunks)


def test_max_chunk_chars_reads_the_env_and_survives_junk(monkeypatch):
    monkeypatch.delenv("SYRINX_TTS_CHUNK_CHARS", raising=False)
    assert max_chunk_chars() == chunking.DEFAULT_MAX_CHUNK_CHARS
    monkeypatch.setenv("SYRINX_TTS_CHUNK_CHARS", "250")
    assert max_chunk_chars() == 250
    monkeypatch.setenv("SYRINX_TTS_CHUNK_CHARS", "not-a-number")
    assert max_chunk_chars() == chunking.DEFAULT_MAX_CHUNK_CHARS


# --- crossfade_concat ----------------------------------------------------


def _ramp(n, val=1.0):
    return np.full(n, val, dtype=np.float32)


def test_crossfade_length_is_sum_minus_the_overlaps():
    xf_ms = 50
    overlap = int(RATE * xf_ms / 1000)
    parts = [_ramp(RATE), _ramp(RATE // 2), _ramp(RATE)]
    out = crossfade_concat(parts, RATE, crossfade_ms=xf_ms)
    assert len(out) == sum(len(p) for p in parts) - (len(parts) - 1) * overlap
    assert out.dtype == np.float32


def test_single_chunk_is_passed_straight_through():
    one = _ramp(100)
    assert crossfade_concat([one], RATE) is one


def test_empty_chunk_list_gives_an_empty_array():
    out = crossfade_concat([], RATE)
    assert out.size == 0 and out.dtype == np.float32


def test_zero_length_chunks_are_skipped():
    parts = [_ramp(1000), np.array([], dtype=np.float32), _ramp(1000)]
    out = crossfade_concat(parts, RATE, crossfade_ms=10)
    assert len(out) == 2000 - int(RATE * 10 / 1000)


def test_the_join_is_a_smooth_equal_power_ramp_not_a_click():
    a, b = _ramp(2000, 1.0), _ramp(2000, 1.0)
    out = crossfade_concat([a, b], RATE, crossfade_ms=20)
    # constant-amplitude inputs stay constant through a linear crossfade
    assert np.allclose(out, 1.0, atol=1e-5)


def test_a_zero_crossfade_is_a_plain_concatenation():
    out = crossfade_concat([_ramp(10), _ramp(10, 0.5)], RATE, crossfade_ms=0)
    assert len(out) == 20
    assert out[10] == pytest.approx(0.5)


# --- trim_tts_output -----------------------------------------------------


def _sine(secs, rate=RATE, amp=0.5, freq=220.0):
    t = np.linspace(0, secs, int(secs * rate), endpoint=False, dtype=np.float32)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def test_trailing_silence_is_trimmed():
    audio = np.concatenate([_sine(1.0), np.zeros(RATE, dtype=np.float32)])
    out = trim_tts_output(audio, RATE)
    assert len(out) < len(audio)
    # ~the speech portion survives (frame-quantized to 20 ms)
    assert len(out) == pytest.approx(RATE, abs=int(0.05 * RATE))


def test_post_silence_hallucination_is_cut_off():
    """[speech][long silence][noise] -> only the speech comes back."""
    audio = np.concatenate([
        _sine(0.8),
        np.zeros(int(1.5 * RATE), dtype=np.float32),
        _sine(0.8, freq=3000.0),  # the hallucinated tail
    ])
    out = trim_tts_output(audio, RATE)
    assert len(out) == pytest.approx(0.8 * RATE, abs=int(0.1 * RATE))


def test_speech_with_no_trailing_silence_is_left_alone():
    audio = _sine(1.0)
    out = trim_tts_output(audio, RATE)
    assert len(out) == pytest.approx(len(audio), abs=int(0.05 * RATE))


def test_a_fade_out_is_applied_to_the_tail():
    audio = np.concatenate([_sine(1.0), np.zeros(RATE, dtype=np.float32)])
    out = trim_tts_output(audio, RATE, fade_ms=30)
    assert abs(float(out[-1])) < 1e-3  # cosine²-faded to silence


def test_short_trailing_silence_is_walked_back_but_a_tail_is_kept():
    """Silence too short to read as a hallucination gap still gets trimmed —
    down to the min_silence_ms tail that keeps the ending from sounding cut."""
    audio = np.concatenate([_sine(1.0), np.zeros(int(0.5 * RATE), dtype=np.float32)])
    out = trim_tts_output(audio, RATE, min_silence_ms=200)
    assert len(out) < len(audio)
    assert len(out) == pytest.approx(1.2 * RATE, abs=int(0.05 * RATE))


def test_audio_shorter_than_one_frame_is_returned_as_is():
    tiny = _sine(0.005)  # < 20 ms
    assert trim_tts_output(tiny, RATE) is tiny


# --- synthesize_chunked --------------------------------------------------


def test_synthesize_chunked_takes_the_single_shot_path(monkeypatch):
    """Short text must not pay the chunk loop's cost — one gen_fn call, and
    the text goes through whole rather than as chunks[0]."""
    seen = []

    def gen(text):
        seen.append(text)
        return np.full(1000, 0.5, dtype=np.float32), RATE

    pcm, rate = asyncio.run(
        chunking.synthesize_chunked(gen, "  short text  ", log=_Log(), label="test")
    )
    assert seen == ["  short text  "]
    assert rate == RATE
    assert len(np.frombuffer(pcm, dtype=np.float32)) == 1000


def test_synthesize_chunked_crossfades_the_pieces(monkeypatch):
    monkeypatch.setenv("SYRINX_TTS_CHUNK_CHARS", "40")
    seen = []

    def gen(text):
        seen.append(text)
        return np.full(RATE // 2, 0.5, dtype=np.float32), RATE

    text = " ".join(["Alpha bravo charlie delta echo."] * 6)
    pcm, rate = asyncio.run(chunking.synthesize_chunked(gen, text, log=_Log(), label="test"))
    assert len(seen) > 1
    overlap = int(RATE * 50 / 1000)
    expected = len(seen) * (RATE // 2) - (len(seen) - 1) * overlap
    assert len(np.frombuffer(pcm, dtype=np.float32)) == expected


class _Log:
    """The `log` synthesize_chunked writes progress through."""

    def __init__(self):
        self.lines = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args)
