"""Canonical prompt tokenization — the SINGLE source of truth shared by activation
extraction (``acsl.extract``) and live serving (``server.py``).

Why this module exists: the pooled last-token POSITION must be identical between
how the cache/directions are built and how the live gate scores. Previously
``acsl.extract`` used the tokenizer's default ``enable_thinking`` (True for Qwen3 →
last token ``assistant\\n``) while ``server.py`` passed ``enable_thinking=False``
(last token after an injected ``<think></think>``, +4 tokens). The server therefore
read a different position than its directions were built at, sign-flipping
near-boundary gate scores (STATUS.md "CRITICAL INSTRUMENT BUG", 2026-07-05).

Route ALL prompt tokenization through :func:`build_prompt_inputs` so the setting can
never drift between build and serve again.
"""
from __future__ import annotations

DEFAULT_MAX_LENGTH = 1024


def build_prompt_inputs(tokenizer, prompt: str, device=None, *,
                        enable_thinking: bool = False,
                        max_length: int = DEFAULT_MAX_LENGTH,
                        chat_template: bool = True) -> dict:
    """Wrap one user ``prompt`` in the model's chat template + generation prompt and
    tokenize it. The pooled last token is the end-of-instruction position the model
    would act from.

    ``enable_thinking`` defaults to **False** — the documented ACSL design (thinking
    OFF for the prompt-encoding stages). Both extraction and serving MUST pass the
    same value; default False keeps them aligned.

    Returns a dict of tensors (batch size 1), moved to ``device`` if given.
    """
    if chat_template and getattr(tokenizer, "chat_template", None):
        tmpl_kwargs = dict(add_generation_prompt=True, tokenize=False)
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                enable_thinking=enable_thinking, **tmpl_kwargs)
        except TypeError:
            # This tokenizer's template doesn't accept enable_thinking; render plainly.
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], **tmpl_kwargs)
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length, add_special_tokens=False)
    else:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=max_length)
    enc = dict(enc)
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc
