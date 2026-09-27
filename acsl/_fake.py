"""A tiny *fake* frozen CausalLM for offline smoke/unit tests.

It mirrors the parts of a HF model ACSL relies on:
  * ``model.model.layers[L].self_attn`` returning ``(attn_output, attn_weights)``
  * a callable ``model(input_ids=..., attention_mask=...)`` forward
  * a minimal tokenizer with ``__call__(text, return_tensors='pt', ...)``

No weights are downloaded and no network is touched. Parameters are frozen
exactly like the real loader, so the frozen-base invariant holds here too.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class FakeTokenizer:
    """Deterministic char-hash tokenizer (no external files)."""

    def __init__(self, vocab_size: int = 256, max_length: int = 64):
        self.vocab_size = vocab_size
        self.model_max_length = max_length
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0

    def __call__(self, text, return_tensors=None, truncation=True, max_length=None, **kw):
        max_length = max_length or self.model_max_length
        ids = [1 + (ord(c) % (self.vocab_size - 1)) for c in str(text)][:max_length]
        if not ids:
            ids = [1]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        return out  # already a dict of tensors (return_tensors='pt'-style)


class _FakeAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, hidden_states, output_attentions: bool = False, **kw):
        out = self.proj(hidden_states)
        attn = None
        if output_attentions:
            b, s, _ = hidden_states.shape
            attn = torch.full((b, self.n_heads, s, s), 1.0 / max(s, 1))
        return out, attn


class _FakeLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d_model)
        self.self_attn = _FakeAttn(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.post_attention_layernorm = nn.LayerNorm(d_model)

    def forward(self, x, output_attentions: bool = False, **kw):
        normed = self.input_layernorm(x)
        attn_out, _ = self.self_attn(normed, output_attentions=output_attentions)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _FakeInner(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_FakeLayer(d_model) for _ in range(n_layers)])

    def forward(self, input_ids, attention_mask=None, output_attentions: bool = False,
                output_hidden_states: bool = False, **kw):
        h = self.embed_tokens(input_ids)
        all_hidden = [h] if output_hidden_states else None
        for layer in self.layers:
            h = layer(h, output_attentions=output_attentions)
            if output_hidden_states:
                all_hidden.append(h)
        if output_hidden_states:
            return h, tuple(all_hidden)
        return h


class FakeCausalLM(nn.Module):
    """Has ``.model.layers[L].self_attn``; frozen + eval by construction."""

    def __init__(self, vocab_size: int = 256, d_model: int = 64, n_layers: int = 6):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            _attn_implementation="eager",
        )
        self.model = _FakeInner(vocab_size, d_model, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, output_attentions: bool = False,
                output_hidden_states: bool = False, **kw):
        if output_hidden_states:
            h, all_hidden = self.model(
                input_ids, attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,
            )
            return SimpleNamespace(
                logits=self.lm_head(h), last_hidden_state=h,
                hidden_states=all_hidden,
            )
        h = self.model(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
        return SimpleNamespace(logits=self.lm_head(h), last_hidden_state=h)


def load_fake_model(vocab_size: int = 256, d_model: int = 64, n_layers: int = 6, device=None):
    """Return ``(model, tokenizer)`` mirroring ``load_frozen_model`` — frozen, eval.

    Moves to the best available device (cuda/DirectML/cpu) unless one is given.
    """
    from .model import pick_device

    device = device if device is not None else pick_device()
    torch.manual_seed(0)
    model = FakeCausalLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)
    model.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.to(device)
    return model, FakeTokenizer(vocab_size=vocab_size)
