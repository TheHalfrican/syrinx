"""Personality LLM — compose & rewrite via a local Qwen3 instruct model.

Loaded in-process with `transformers` (CPU float32 here; auto-CUDA float16 on the
4090), lazily on first use so engine boot stays fast. Character-driven: both
features need a voice profile with a `personality`. Prompt templates are ported
from Voicebox's personality service.

Model: $SYRINX_LLM_MODEL (a key of _MODELS, or a raw HF repo). Default Qwen3-1.7B.
"""

import asyncio
import logging
import os

log = logging.getLogger("syrinx.engine.llm")

_MODELS = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",
    "4B": "Qwen/Qwen3-4B",
}

# --- prompt templates (ported verbatim from Voicebox) -----------------------

_CHARACTER_FRAMING = """You are roleplaying a specific character described below. Stay fully in character in everything you produce.

Rules that apply to every response:
- Do not break character. Do not explain what you are doing, refuse, apologize, greet the user, or acknowledge being an AI or assistant.
- Do not narrate action ("*smiles*", "(leans back)") or stage directions. Produce speech only.
- Do not wrap the output in quotes, code fences, or labels. Output the character's words and nothing else.
- Match the character's register — if they are curt, be curt; if they ramble, ramble; if they swear, swear."""

_COMPOSE_TASK = """Task: Produce one short utterance — one or two sentences at most — that this character might say right now, unprompted. A remark, an observation, a thought out loud. No greeting, no addressing anyone by name, no "Well, …" or "So, …" opener unless it fits the character naturally. Just a natural line of speech."""

_COMPOSE_PROMPTED_TASK = """Task: The user's message is a brief note, topic, or situation. Produce one short utterance — one or two sentences at most — that this character would say about it or in response to it. Do not answer as an assistant and do not repeat the note back; speak only as the character. Output only the character's words."""

_REWRITE_TASK = """Task: The user's next message is a piece of text. Restate every idea in it using your character's voice — keep the meaning, change the wording. Do not add new ideas, do not drop any, do not reply to the text. Output only the restated version."""


def _build_system_prompt(personality: str, task: str) -> str:
    return f"{_CHARACTER_FRAMING}\n\nCharacter description:\n{personality.strip()}\n\n{task}"


class PersonalityLLM:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self.model_size = os.environ.get("SYRINX_LLM_MODEL", "1.7B")
        self._device = "cpu"

    async def load(self) -> None:
        if self._model is not None:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        repo = _MODELS.get(self.model_size, self.model_size)  # allow a raw repo id too
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        log.info("loading LLM %s on %s (first run downloads weights)...", repo, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(repo)
        self._model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype)
        self._model.to(self._device)
        self._model.eval()
        log.info("LLM loaded")

    async def _generate(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str:
        await self.load()

        def _run() -> str:
            import torch

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            kwargs = {"max_new_tokens": max_tokens}
            if temperature and temperature > 0:
                kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
            else:
                kwargs.update(do_sample=False)
            with torch.no_grad():
                out = self._model.generate(**inputs, **kwargs)
            new_ids = out[0][inputs.input_ids.shape[1] :]
            return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        text = await asyncio.to_thread(_run)
        log.info("llm out: %r", text[:80])
        return text

    async def compose(self, personality: str, prompt: str = "") -> str:
        """An in-character line. Guided by `prompt` if given, else unprompted."""
        prompt = prompt.strip()
        if prompt:
            system = _build_system_prompt(personality, _COMPOSE_PROMPTED_TASK)
            return await self._generate(system, prompt, max_tokens=256, temperature=0.85)
        system = _build_system_prompt(personality, _COMPOSE_TASK)
        return await self._generate(system, "Speak.", max_tokens=256, temperature=0.9)

    async def rewrite(self, personality: str, text: str) -> str:
        """Restate `text` in the character's voice, meaning preserved (temp 0.3)."""
        system = _build_system_prompt(personality, _REWRITE_TASK)
        return await self._generate(system, text, max_tokens=1024, temperature=0.3)
