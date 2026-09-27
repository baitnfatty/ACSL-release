"""The security loop must fail closed.

safe=True only when the loop converged AND final risk <= thr. Otherwise
safe=False forces the safe action via the policy. We drive synthetic high-risk
states through a hand-built linear head whose dynamics we can predict exactly.
"""

import pytest

torch = pytest.importorskip("torch")

from acsl.head import SecurityHead
from acsl.loop import security_loop
from acsl.policy import action_policy


def _linear_head(d_model, w_harm, b_harm=0.0):
    """2-category head with risk(h) = w_harm . h + b_harm (benign row = 0)."""
    head = SecurityHead(d_model, 2, hidden=None)
    W = torch.zeros(2, d_model)
    W[1] = torch.as_tensor(w_harm, dtype=torch.float32)
    head.net.weight.data.copy_(W)
    head.net.bias.data.copy_(torch.tensor([0.0, float(b_harm)]))
    head.eval()
    return head


def test_constant_high_risk_fails_closed():
    d = 8
    u = torch.zeros(d)
    u[0] = 1.0
    # weight orthogonal-to-everything (zero) + high bias => risk == 5 regardless
    head = _linear_head(d, torch.zeros(d), b_harm=5.0)
    h0 = torch.randn(d)
    h_adj, r_star, safe = security_loop(h0, head, u, g_max=8.0, K=4, eps=1e-2, thr=0.0)
    assert safe is False                       # risk stayed above threshold
    assert r_star == pytest.approx(5.0, abs=1e-4)
    assert h_adj.shape == h0.shape
    assert action_policy(r_star, safe) == "refuse_or_route"


def test_unconverged_within_K_fails_closed():
    d = 8
    u = torch.zeros(d)
    u[0] = 1.0
    head = _linear_head(d, u, b_harm=100.0)    # risk starts ~100, drops slowly
    h0 = torch.zeros(d)
    h_adj, r_star, safe = security_loop(h0, head, u, g_max=8.0, K=1, eps=1e-2, thr=0.0)
    assert safe is False                       # one step can't reach thr; not converged
    assert r_star > 0.0
    assert action_policy(r_star, safe) in ("refuse_or_route", "flag")


def test_steering_can_resolve_to_safe():
    d = 8
    u = torch.zeros(d)
    u[0] = 1.0
    head = _linear_head(d, u, b_harm=0.0)      # risk(h) = h[0]
    h0 = u.clone()                              # risk starts at +1
    h_adj, r_star, safe = security_loop(h0, head, u, g_max=8.0, K=4, eps=1e-2, thr=0.0)
    assert safe is True                         # steering drove risk below threshold
    assert r_star <= 0.0
    assert action_policy(r_star, safe) in ("answer", "steer")
