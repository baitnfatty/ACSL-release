"""AttnTap wiring tests — run against a tiny fake model, no downloads.

Verifies: o_L shape/dtype, the hook fires exactly once per forward pass, h_L is
captured, and A_L is captured only when capture_pattern=True.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from acsl.hooks import AttnTap


class FakeAttn(nn.Module):
    """Mimics a HF self_attn: returns (attn_output, attn_weights)."""

    def __init__(self, d_model: int, n_heads: int = 2):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, hidden_states, output_attentions: bool = False, **kwargs):
        out = self.proj(hidden_states)
        attn = None
        if output_attentions:
            b, s, _ = hidden_states.shape
            attn = torch.ones(b, self.n_heads, s, s)
        return out, attn


class FakeLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d_model)
        self.self_attn = FakeAttn(d_model)

    def forward(self, x, **kwargs):
        normed = self.input_layernorm(x)
        attn_out, _ = self.self_attn(normed, **kwargs)
        return x + attn_out  # residual add (new tensor, not in-place)


class FakeInner(nn.Module):
    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(d_model) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FakeModel(nn.Module):
    """Has the model.model.layers[L].self_attn structure AttnTap expects."""

    def __init__(self, d_model: int = 16, n_layers: int = 4):
        super().__init__()
        self.model = FakeInner(d_model, n_layers)

    def forward(self, x):
        return self.model(x)


@pytest.fixture
def fake():
    torch.manual_seed(0)
    return FakeModel(d_model=16, n_layers=4)


def test_o_L_shape_and_single_fire(fake):
    b, s, d = 3, 7, 16
    x = torch.randn(b, s, d)
    with AttnTap(fake, layer=2) as tap:
        fake(x)
        assert tap.o_L.shape == (b, s, d)
        assert tap.h_L.shape == (b, s, d)
        assert tap.o_L.dtype == x.dtype
        assert tap.n_calls == 1  # fired exactly once for one forward pass


def test_hook_removed_on_exit(fake):
    x = torch.randn(2, 5, 16)
    tap = AttnTap(fake, layer=1)
    with tap:
        fake(x)
    assert tap.n_calls == 1
    # After exit the hook is gone: another forward must not increment.
    fake(x)
    assert tap.n_calls == 1


def test_A_L_optional(fake):
    x = torch.randn(2, 4, 16)
    # default: pattern not captured
    with AttnTap(fake, layer=0, capture_pattern=False) as tap:
        fake(x)
        assert tap.A_L is None
    # opt-in: pattern captured with shape (batch, heads, seq, seq)
    with AttnTap(fake, layer=0, capture_pattern=True) as tap:
        fake(x)
        assert tap.A_L is not None
        assert tap.A_L.shape == (2, 2, 4, 4)


def test_o_L_before_forward_raises(fake):
    with AttnTap(fake, layer=0) as tap:
        with pytest.raises(RuntimeError):
            _ = tap.o_L
