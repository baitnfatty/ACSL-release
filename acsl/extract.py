"""Stream residual-stream activations to disk (v3: all layers, one forward pass).

For each example we run one frozen forward pass with ``output_hidden_states=True``
and write a pooled feature vector per layer.  Nothing is held in VRAM beyond the
current example: tensors are moved to CPU float32 and saved as
``{out_dir}/{split}/{layer}/{example_id}.npy`` (numpy memmap-friendly), with a
per-split ``index.jsonl`` mapping example_id -> label.

Pooling defaults to the last non-pad token (the position whose hidden state the
head/loop will read at decode time).

v3 change: uses ``output_hidden_states=True`` to capture the full residual-stream
stack in one forward pass per example, instead of N_layers forward passes with
per-layer hooks.  The hidden_states tuple from HF returns post-block residual
activations (index 0 = embeddings, 1..n_layers = after each transformer block).
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np
import torch

from .model import pick_device


def _pool(t: torch.Tensor, attention_mask: torch.Tensor | None, how: str) -> torch.Tensor:
    """Pool (batch, seq, d) -> (batch, d)."""
    if how == "last":
        if attention_mask is None:
            return t[:, -1, :]
        lengths = attention_mask.long().sum(dim=1) - 1
        idx = lengths.clamp(min=0)
        return t[torch.arange(t.shape[0], device=t.device), idx, :]
    if how == "mean":
        if attention_mask is None:
            return t.mean(dim=1)
        m = attention_mask.unsqueeze(-1).to(t.dtype)
        return (t * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    raise ValueError(f"unknown pool {how!r}; use 'last' or 'mean'")


def _tokenize(tokenizer, prompt: str, device, max_length: int, chat_template: bool = False,
              enable_thinking: bool = False):
    """Tokenize a prompt via the shared canonical path (:mod:`acsl.tokenization`), so
    the pooled last-token position matches live serving exactly. ``enable_thinking``
    defaults to False (documented ACSL design)."""
    from .tokenization import build_prompt_inputs
    return build_prompt_inputs(tokenizer, prompt, device,
                               enable_thinking=enable_thinking,
                               max_length=max_length, chat_template=chat_template)


def smoke_check(model, tokenizer, device=None, max_length=512, n_prompts=5):
    """Verify output_hidden_states returns the full residual stack on this device.

    Runs a handful of forward passes and checks:
      1. hidden_states is a tuple of length n_layers + 1
      2. Each element has shape [batch, seq, d_model]
      3. No errors or silent CPU fallback on DirectML

    Returns (n_layers, d_model) on success; raises on failure.
    """
    device = device or pick_device()
    n_layers_expected = model.config.num_hidden_layers
    prompts = [f"Test prompt number {i} for smoke check." for i in range(n_prompts)]

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = _tokenize(tokenizer, prompt, device, max_length)
            out = model(**inputs, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states

            if not isinstance(hs, (tuple, list)):
                raise RuntimeError(
                    f"output_hidden_states returned {type(hs).__name__}, expected tuple"
                )
            if len(hs) != n_layers_expected + 1:
                raise RuntimeError(
                    f"expected {n_layers_expected + 1} hidden states (embed + {n_layers_expected} layers), "
                    f"got {len(hs)}"
                )

            d_model = hs[1].shape[-1]
            for layer_idx, h in enumerate(hs[1:], start=1):
                if h.ndim != 3:
                    raise RuntimeError(
                        f"hidden_states[{layer_idx}] has {h.ndim} dims, expected 3 (batch, seq, d_model)"
                    )
                if h.shape[-1] != d_model:
                    raise RuntimeError(
                        f"hidden_states[{layer_idx}] d_model={h.shape[-1]}, expected {d_model}"
                    )

            del out, hs

    print(f"[smoke] OK: {n_prompts} prompts, {n_layers_expected} layers, d_model={d_model}")
    return n_layers_expected, d_model


def extract_activations(
    model,
    loader: Iterable,
    layers,
    out_dir: str,
    tokenizer=None,
    pool: str = "last",
    device: torch.device | None = None,
    max_length: int = 512,
    chat_template: bool = False,
    enable_thinking: bool = False,
) -> dict:
    """Extract residual-stream activations for every example, all layers at once.

    One forward pass per example with ``output_hidden_states=True``.  Saves pooled
    (last-token) activations per layer to disk in the same format as v1/v2:
    ``{out_dir}/{split}/{layer}/{example_id}.npy``.

    ``layers`` is a list of layer indices to save (0-based, post-block).  Pass
    ``None`` to save all layers (determined from model config).

    Returns a manifest dict (counts per split, paths, layer info).
    """
    device = device or pick_device()
    n_model_layers = model.config.num_hidden_layers

    if layers is None:
        layers = list(range(n_model_layers))
    else:
        layers = [int(L) for L in layers if int(L) < n_model_layers]

    os.makedirs(out_dir, exist_ok=True)

    counts: dict[str, int] = {}
    index_files: dict[str, object] = {}

    def _index(split: str):
        if split not in index_files:
            d = os.path.join(out_dir, split)
            os.makedirs(d, exist_ok=True)
            index_files[split] = open(os.path.join(d, "index.jsonl"), "w", encoding="utf-8")
        return index_files[split]

    try:
        with torch.no_grad():
            examples_buf = []
            for ex in loader:
                split = getattr(ex, "split", "default")
                ex_id = str(getattr(ex, "id"))
                label = int(getattr(ex, "label", -1))

                if getattr(ex, "input_ids", None) is not None:
                    inputs = {
                        "input_ids": torch.as_tensor(ex.input_ids).reshape(1, -1)
                    }
                    am = getattr(ex, "attention_mask", None)
                    if am is not None:
                        inputs["attention_mask"] = torch.as_tensor(am).reshape(1, -1)
                else:
                    if tokenizer is None:
                        raise ValueError("loader yields text prompts but no tokenizer was passed")
                    inputs = _tokenize(tokenizer, ex.prompt, "cpu", max_length,
                                       chat_template=chat_template, enable_thinking=enable_thinking)

                examples_buf.append((ex_id, split, label, inputs))

            total = len(examples_buf)
            print(f"[extract] {total} examples buffered, extracting {len(layers)} layers per forward pass")

            # Write index entries
            for ex_id, split, label, _ in examples_buf:
                _index(split).write(
                    json.dumps({"id": ex_id, "label": label}) + "\n"
                )
                counts[split] = counts.get(split, 0) + 1

            # Ensure layer directories exist for all splits
            for split in counts:
                for L in layers:
                    os.makedirs(os.path.join(out_dir, split, str(L)), exist_ok=True)

            # One forward pass per example — all layers extracted at once
            done = 0
            for ex_id, split, label, inputs_cpu in examples_buf:
                inputs = {k: v.to(device) for k, v in inputs_cpu.items()}
                attn_mask = inputs.get("attention_mask")

                out = model(**inputs, output_hidden_states=True, use_cache=False)
                hidden_states = out.hidden_states  # tuple of (n_layers+1,)

                for L in layers:
                    h = hidden_states[L + 1]  # +1: index 0 is embeddings
                    vec = _pool(h, attn_mask, pool)[0].to("cpu", torch.float32).numpy()
                    np.save(os.path.join(out_dir, split, str(L), f"{ex_id}.npy"), vec)

                del out, hidden_states, inputs
                done += 1
                if done % 50 == 0:
                    print(f"[extract] {done}/{total}")

            print(f"[extract] {done}/{total} done")

    finally:
        for f in index_files.values():
            f.close()

    manifest = {
        "out_dir": out_dir,
        "layers": layers,
        "which": "residual_stream",
        "pool": pool,
        "counts": counts,
        "n_model_layers": n_model_layers,
    }
    with open(os.path.join(out_dir, "extract_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_cached(out_dir: str, split: str, layer: int):
    """Load cached features and labels for a (split, layer): returns (X, y, ids)."""
    layer_dir = os.path.join(out_dir, split, str(layer))
    index_path = os.path.join(out_dir, split, "index.jsonl")
    labels: dict[str, int] = {}
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            labels[str(rec["id"])] = int(rec["label"])

    X, y, ids = [], [], []
    for ex_id, lab in labels.items():
        path = os.path.join(layer_dir, f"{ex_id}.npy")
        if not os.path.exists(path):
            continue
        X.append(np.load(path))
        y.append(lab)
        ids.append(ex_id)
    if not X:
        return np.empty((0, 0), np.float32), np.empty((0,), np.int64), []
    return np.stack(X).astype(np.float32), np.asarray(y, np.int64), ids
