"""The Attention-Coupled Security Loop.

At layer L: judge the hidden state with the head, steer it along the (negated)
caution direction proportionally to risk, re-judge, and repeat up to K times or
until the risk read stabilizes (``|Δr| < eps``). The adjusted ``h_L`` is what the
forward pass would continue with.

Fail closed (hard constraint 6): ``safe`` is True only when the loop *converged*
AND the final risk is at/below ``thr``. If it did not converge, or risk stays
above threshold, ``safe=False`` and the policy must force the safe action.
"""

from __future__ import annotations

import torch

from .head import risk_logit


def _gain(r: torch.Tensor, g_max: float) -> torch.Tensor:
    """Bounded steering gain in [0, g_max): only push when risk is positive."""
    return g_max * torch.sigmoid(r) * (r > 0).to(r.dtype)


def security_loop(h_L, head, u, g_max: float = 8.0, K: int = 4, eps: float = 1e-2, thr: float = 0.0, benign_index: int = 0, temperature: float = 1.0):
    """Run the deliberation loop on a single layer's hidden state.

    Parameters
    ----------
    h_L : Tensor (d_model,) or (n, d_model)
        Hidden state(s) read at the tapped layer.
    head : SecurityHead
        Frozen-base classifier producing per-category risk logits.
    u : Tensor (d_model,)
        Unit caution direction (mean_harm - mean_safe), normalized. Steering is
        along ``-u`` (toward safe).
    g_max, K, eps, thr : loop hyperparameters (see configs/default.yaml).

    Returns
    -------
    (h_adjusted, r_star, safe)
        ``h_adjusted`` has the input's shape; ``r_star`` is the final (max, i.e.
        conservative) risk log-odds as a float; ``safe`` is the fail-closed flag.
    """
    was_1d = h_L.dim() == 1
    h = h_L.unsqueeze(0) if was_1d else h_L
    h = h.clone().float()
    u = u.to(h.dtype)
    u_unit = u / torch.linalg.vector_norm(u).clamp(min=1e-12)

    def _risk(hh):
        with torch.no_grad():
            return risk_logit(head(hh), benign_index=benign_index, temperature=temperature)  # (n,)

    r = _risk(h)
    r_star = float(r.max())
    converged = False

    for _ in range(int(K)):
        g = _gain(r, g_max).unsqueeze(-1)           # (n, 1)
        h = h - g * u_unit                          # steer toward safe along -u
        r_new = _risk(h)
        delta = float((r_new - r).abs().max())
        r = r_new
        r_star = float(r.max())
        if delta < eps:
            converged = True
            break

    # Fail closed: safe only if the loop settled AND risk is at/below threshold.
    safe = bool(converged and (r_star <= thr))

    h_out = h.squeeze(0) if was_1d else h
    return h_out.to(h_L.dtype), r_star, safe


def _get_decoder_layer(model, layer: int):
    """Locate the full decoder layer block (not just self_attn)."""
    for path in (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ):
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj[layer]
    raise AttributeError("could not locate decoder layer block")


class ResidualStreamHook:
    """Couple the loop into the forward pass at the RESIDUAL STREAM level.

    Hooks the full decoder layer block (not just self_attn), so the head
    receives the same residual-stream vectors it was trained on from
    output_hidden_states. The hook reads the layer's output hidden state
    at the last token position, runs security_loop, and writes the
    adjusted vector back.
    """

    def __init__(self, model, layer, head, u, g_max=8.0, K=4, eps=1e-2, thr=0.0, benign_index=0, temperature=1.0):
        self._module = _get_decoder_layer(model, int(layer))
        self.head = head
        self.u = u
        self.params = dict(g_max=g_max, K=K, eps=eps, thr=thr, benign_index=benign_index, temperature=temperature)
        self._handle = None
        self.last_r_star: float | None = None
        self.last_safe: bool | None = None

    def _hook(self, module, inputs, output):
        # Decoder layers return either a bare tensor or a tuple (hidden_states, ...)
        bare = torch.is_tensor(output)
        h = output if bare else output[0]
        last = h[:, -1, :]  # (batch, d_model)
        adj, r_star, safe = security_loop(last, self.head, self.u, **self.params)
        self.last_r_star, self.last_safe = r_star, safe
        new_h = h.clone()
        new_h[:, -1, :] = adj.to(new_h.dtype)
        if bare:
            return new_h
        return (new_h,) + tuple(output[1:])

    def __enter__(self):
        self._handle = self._module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


class SecurityLoopHook:
    """Couple the loop INTO the forward pass at the tapped attention layer.

    Registers a forward hook on ``model.model.layers[L].self_attn`` that reads the
    attention output ``o_L`` at the current last position, runs ``security_loop``,
    and writes the adjusted vector back so the forward pass continues with it.
    The most recent ``(r_star, safe)`` are exposed for the policy. Fail-closed
    behaviour is inherited from ``security_loop``.
    """

    def __init__(self, model, layer, head, u, g_max=8.0, K=4, eps=1e-2, thr=0.0, benign_index=0, temperature=1.0):
        from .hooks import _get_attn_module

        self._module = _get_attn_module(model, int(layer))
        self.head = head
        self.u = u
        self.params = dict(g_max=g_max, K=K, eps=eps, thr=thr, benign_index=benign_index, temperature=temperature)
        self._handle = None
        self.last_r_star: float | None = None
        self.last_safe: bool | None = None

    def _hook(self, module, inputs, output):
        bare = torch.is_tensor(output)
        o = output if bare else output[0]
        last = o[:, -1, :]  # (batch, d_model)
        adj, r_star, safe = security_loop(last, self.head, self.u, **self.params)
        self.last_r_star, self.last_safe = r_star, safe
        new_o = o.clone()
        new_o[:, -1, :] = adj.to(new_o.dtype)
        if bare:
            return new_o
        return (new_o,) + tuple(output[1:])

    def __enter__(self):
        self._handle = self._module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False
