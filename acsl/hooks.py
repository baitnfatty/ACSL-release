"""Attention-sublayer tap.

``AttnTap`` registers a forward hook on ``model.model.layers[L].self_attn`` and
captures, per forward pass:

  * ``o_L`` — the attention sublayer output, shape ``(batch, seq, d_model)``.
  * ``h_L`` — the hidden states fed *into* the attention sublayer (the residual
    stream after this layer's input norm), same shape; used by the head/loop.
  * ``A_L`` — the attention pattern (only if ``capture_pattern=True``; requires
    eager attention). ``None`` otherwise.

Eager attention is mandatory: with FlashAttention/SDPA the attention sublayer
returns no pattern and the hooked tensors differ. See ``model.load_frozen_model``.
"""

from __future__ import annotations

import torch


def _get_attn_module(model, layer: int):
    """Locate the self-attention module for a decoder layer.

    Targets ``model.model.layers[layer].self_attn`` (Qwen2/Llama/Mistral-style).
    A couple of fallbacks are tried for other naming conventions.
    """
    for path in (
        ("model", "layers"),       # Qwen2 / Llama / Mistral
        ("transformer", "h"),      # GPT-2 / GPT-NeoX style
        ("gpt_neox", "layers"),
    ):
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if not ok:
            continue
        layers = obj
        block = layers[layer]
        for attn_attr in ("self_attn", "attention", "attn"):
            if hasattr(block, attn_attr):
                return getattr(block, attn_attr)
    raise AttributeError(
        "could not locate self-attention module; expected model.model.layers[L].self_attn"
    )


class AttnTap:
    """Context manager that taps one layer's attention sublayer.

    Usage::

        with AttnTap(model, layer=12, capture_pattern=False) as tap:
            model(**inputs)
            o = tap.o_L            # (batch, seq, d_model)
            h = tap.h_L            # (batch, seq, d_model)
    """

    def __init__(self, model, layer: int, capture_pattern: bool = False):
        self.model = model
        self.layer = int(layer)
        self.capture_pattern = bool(capture_pattern)

        self._module = _get_attn_module(model, self.layer)
        self._fwd_handle = None
        self._pre_handle = None

        self._o_L: torch.Tensor | None = None
        self._h_L: torch.Tensor | None = None
        self._A_L: torch.Tensor | None = None
        self.n_calls = 0  # number of times the hook fired this lifetime

    # -- hook bodies -------------------------------------------------------
    def _pre_hook(self, module, args, kwargs):
        # Ask the eager attention to also return its pattern.
        if self.capture_pattern:
            kwargs = dict(kwargs)
            kwargs["output_attentions"] = True
        return args, kwargs

    def _fwd_hook(self, module, inputs, output):
        self.n_calls += 1
        # h_L: the hidden states fed into the attention sublayer.
        if isinstance(inputs, (tuple, list)) and len(inputs) > 0 and torch.is_tensor(inputs[0]):
            self._h_L = inputs[0].detach()
        else:
            self._h_L = None

        # o_L / A_L: attention returns either a bare tensor or a tuple whose
        # first element is the output and (optionally) second the pattern.
        if torch.is_tensor(output):
            self._o_L = output.detach()
            self._A_L = None
        else:
            self._o_L = output[0].detach()
            attn = output[1] if len(output) > 1 else None
            self._A_L = attn.detach() if torch.is_tensor(attn) else None

    # -- context management ------------------------------------------------
    def __enter__(self) -> "AttnTap":
        self.reset()
        if self.capture_pattern:
            self._pre_handle = self._module.register_forward_pre_hook(
                self._pre_hook, with_kwargs=True
            )
        self._fwd_handle = self._module.register_forward_hook(self._fwd_hook)
        return self

    def __exit__(self, *exc) -> bool:
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if self._pre_handle is not None:
            self._pre_handle.remove()
            self._pre_handle = None
        return False  # do not suppress exceptions

    def reset(self) -> None:
        self._o_L = self._h_L = self._A_L = None
        self.n_calls = 0

    # -- captured tensors --------------------------------------------------
    @property
    def o_L(self) -> torch.Tensor:
        if self._o_L is None:
            raise RuntimeError("o_L not captured yet; run a forward pass inside the AttnTap context")
        return self._o_L

    @property
    def h_L(self) -> torch.Tensor:
        if self._h_L is None:
            raise RuntimeError("h_L not captured yet; run a forward pass inside the AttnTap context")
        return self._h_L

    @property
    def A_L(self):
        """Attention pattern, or None when capture_pattern is off / unavailable."""
        return self._A_L
