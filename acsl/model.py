"""Frozen base-model loading.

Hard rules enforced here:
  * eager attention (so we can hook the attention sublayer output o_L and,
    optionally, the attention pattern A_L). No FlashAttention / SDPA.
  * every base parameter has requires_grad_(False) — we NEVER backprop through
    the transformer. Only the SecurityHead is trained, elsewhere.
  * model.eval(); device-agnostic (cuda==AMD on ROCm, else cpu).
"""

from __future__ import annotations

import torch


def _directml_device():
    """Return a DirectML device (preferring an AMD adapter) or None.

    DirectML (torch-directml) runs PyTorch on any DirectX-12 GPU on Windows/WSL,
    including RDNA2 (RX 6900 XT) that ROCm-on-WSL won't execute compute for.
    """
    try:
        import torch_directml
    except Exception:
        return None
    if not torch_directml.is_available() or torch_directml.device_count() == 0:
        return None
    idx = 0
    for i in range(torch_directml.device_count()):
        if any(k in torch_directml.device_name(i) for k in ("Radeon", "AMD", "6900")):
            idx = i
            break
    return torch_directml.device(idx)


def is_directml(device) -> bool:
    """True if device is a DirectML device (torch type 'privateuseone')."""
    return getattr(device, "type", None) == "privateuseone"


def pick_device() -> torch.device:
    """Best available device: cuda (incl. ROCm/AMD) > DirectML (any DX12 GPU) > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    dml = _directml_device()
    if dml is not None:
        return dml
    return torch.device("cpu")


_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def resolve_dtype(dtype) -> torch.dtype:
    """Accept a torch.dtype or a string name; return a torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        try:
            return _DTYPES[dtype.lower()]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"unknown dtype {dtype!r}; choose one of {sorted(_DTYPES)}") from exc
    raise TypeError(f"dtype must be a str or torch.dtype, got {type(dtype)!r}")


def load_frozen_model(name: str, dtype=torch.float16, device: torch.device | None = None):
    """Load a frozen instruct model for activation tapping.

    Returns ``(model, tokenizer)``. The model uses ``attn_implementation='eager'``,
    has every parameter frozen (``requires_grad_(False)``), and is in ``eval()``
    mode on the selected device.
    """
    # Imported lazily so the rest of the package (metrics, etc.) imports without
    # transformers present.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = resolve_dtype(dtype)
    device = device or pick_device()
    if is_directml(device) and torch_dtype == torch.bfloat16:
        # DirectML doesn't support bf16; fall back to fp16.
        torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch_dtype,
        attn_implementation="eager",  # REQUIRED: hookable attention sublayer.
    )

    # Freeze the base. No gradients ever flow through the transformer.
    model.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(False)

    model.eval()
    model.to(device)

    # Sanity: assert eager attention actually took effect.
    impl = getattr(model.config, "_attn_implementation", None)
    if impl not in (None, "eager"):
        raise RuntimeError(
            f"expected eager attention, got {impl!r}; ACSL must hook the attention sublayer"
        )

    return model, tokenizer


def assert_frozen(model) -> None:
    """Raise if any base parameter is trainable. Cheap invariant check."""
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise AssertionError(f"base model is not frozen; trainable params: {trainable[:5]} ...")
