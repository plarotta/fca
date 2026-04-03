"""Unit tests for FCA architecture modules."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from fca.config import FCAConfig
from fca.model import BottleneckMLP, FCAAttention, FCAGate, FCABranch, FCABlock, FCAGPT
from fca.losses import EMATargetTracker, lambda_schedule, compute_future_loss, compute_total_loss


@pytest.fixture
def config():
    return FCAConfig(
        block_size=64,
        vocab_size=256,
        n_layer=6,
        n_head=4,
        n_embd=128,
        fca_layers=[4, 5],
        bottleneck_dim=32,
        bottleneck_hidden_dim=64,
        fca_n_head=4,
    )


@pytest.fixture
def small_config():
    """Tiny config for fast smoke tests."""
    return FCAConfig(
        block_size=16,
        vocab_size=64,
        n_layer=4,
        n_head=2,
        n_embd=32,
        fca_layers=[2, 3],
        bottleneck_dim=8,
        bottleneck_hidden_dim=16,
        fca_n_head=2,
    )


class TestBottleneckMLP:
    def test_output_shape(self, config):
        mlp = BottleneckMLP(config)
        h = torch.randn(2, 16, config.n_embd)
        z = mlp(h)
        assert z.shape == (2, 16, config.bottleneck_dim)

    def test_gradient_flows(self, config):
        mlp = BottleneckMLP(config)
        h = torch.randn(2, 16, config.n_embd, requires_grad=True)
        z = mlp(h)
        z.sum().backward()
        assert h.grad is not None


class TestFCAAttention:
    def test_output_shape(self, config):
        attn = FCAAttention(config)
        z = torch.randn(2, 16, config.bottleneck_dim)
        h = torch.randn(2, 16, config.n_embd)
        out = attn(z, h)
        assert out.shape == (2, 16, config.n_embd)

    def test_causal(self, config):
        """Verify output at position t doesn't change when future positions are modified."""
        attn = FCAAttention(config)
        attn.eval()
        z = torch.randn(1, 8, config.bottleneck_dim)
        h = torch.randn(1, 8, config.n_embd)

        out1 = attn(z, h)

        # Modify future positions (5, 6, 7) and check position 4 is unchanged
        h2 = h.clone()
        h2[:, 5:, :] = torch.randn_like(h2[:, 5:, :])
        z2 = z.clone()
        z2[:, 5:, :] = torch.randn_like(z2[:, 5:, :])
        out2 = attn(z2, h2)

        torch.testing.assert_close(out1[:, :5, :], out2[:, :5, :])


class TestFCAGate:
    def test_output_range(self, config):
        gate = FCAGate(config)
        h = torch.randn(2, 16, config.n_embd)
        g = gate(h)
        assert g.shape == (2, 16, 1)
        assert (g >= 0).all() and (g <= 1).all()

    def test_initial_value_near_zero(self, config):
        """Gate should start nearly closed (bias initialized to -2)."""
        gate = FCAGate(config)
        h = torch.zeros(1, 1, config.n_embd)
        g = gate(h)
        assert g.item() < 0.2  # sigmoid(-2) ≈ 0.12


class TestFCABlock:
    def test_with_fca(self, config):
        block = FCABlock(config, has_fca=True)
        x = torch.randn(2, 16, config.n_embd)
        out, z, g = block(x)
        assert out.shape == (2, 16, config.n_embd)
        assert z is not None
        assert z.shape == (2, 16, config.bottleneck_dim)
        assert g is not None
        assert g.shape == (2, 16, 1)

    def test_without_fca(self, config):
        block = FCABlock(config, has_fca=False)
        x = torch.randn(2, 16, config.n_embd)
        out, z, g = block(x)
        assert out.shape == (2, 16, config.n_embd)
        assert z is None
        assert g is None


class TestFCAGPT:
    def test_forward_with_targets(self, small_config):
        model = FCAGPT(small_config)
        idx = torch.randint(0, small_config.vocab_size, (2, 16))
        targets = torch.randint(0, small_config.vocab_size, (2, 16))
        logits, ce_loss, aux = model(idx, targets)

        assert logits.shape == (2, 16, small_config.vocab_size)
        assert ce_loss is not None
        assert ce_loss.item() > 0

        # Check aux outputs
        assert len(aux['hidden_states']) == small_config.n_layer + 1  # including post-embedding
        assert set(aux['z_vectors'].keys()) == set(small_config.fca_layers)
        assert set(aux['gate_values'].keys()) == set(small_config.fca_layers)
        assert set(aux['future_preds'].keys()) == set(small_config.fca_layers)

    def test_forward_inference(self, small_config):
        model = FCAGPT(small_config)
        idx = torch.randint(0, small_config.vocab_size, (2, 16))
        logits, ce_loss, aux = model(idx)

        assert logits.shape == (2, 1, small_config.vocab_size)
        assert ce_loss is None

    def test_generate(self, small_config):
        model = FCAGPT(small_config)
        model.eval()
        idx = torch.randint(0, small_config.vocab_size, (1, 4))
        out = model.generate(idx, max_new_tokens=5)
        assert out.shape == (1, 9)

    def test_random_z_ablation(self, small_config):
        small_config.random_z = True
        model = FCAGPT(small_config)
        idx = torch.randint(0, small_config.vocab_size, (2, 16))
        targets = torch.randint(0, small_config.vocab_size, (2, 16))
        logits, ce_loss, aux = model(idx, targets)

        # Should still produce valid outputs
        assert logits.shape == (2, 16, small_config.vocab_size)
        assert ce_loss is not None


class TestLosses:
    def test_lambda_schedule(self):
        assert lambda_schedule(0, 20000, True) == 0.0
        assert lambda_schedule(10000, 20000, True) == 0.5
        assert lambda_schedule(20000, 20000, True) == 1.0
        assert lambda_schedule(30000, 20000, True) == 1.0
        # No schedule
        assert lambda_schedule(0, 20000, False) == 1.0

    def test_future_loss_computation(self, small_config):
        model = FCAGPT(small_config)
        ema = EMATargetTracker(small_config, torch.device('cpu'))

        idx = torch.randint(0, small_config.vocab_size, (2, 16))
        targets = torch.randint(0, small_config.vocab_size, (2, 16))
        _, ce_loss, aux = model(idx, targets)

        ema.update(aux['hidden_states'])

        future_loss = compute_future_loss(
            aux['future_preds'], aux['hidden_states'], ema, small_config
        )
        assert future_loss.item() >= 0
        assert future_loss.requires_grad

    def test_total_loss_integration(self, small_config):
        model = FCAGPT(small_config)
        ema = EMATargetTracker(small_config, torch.device('cpu'))

        idx = torch.randint(0, small_config.vocab_size, (2, 16))
        targets = torch.randint(0, small_config.vocab_size, (2, 16))
        _, ce_loss, aux = model(idx, targets)

        ema.update(aux['hidden_states'])

        total, ce, future, lam = compute_total_loss(
            ce_loss, aux['future_preds'], aux['hidden_states'],
            ema, small_config, step=10000,
        )
        assert total.item() > 0
        assert total.requires_grad
        # At step 10000 with 20000 warmup, lambda should be 0.5
        assert lam == pytest.approx(0.5)

    def test_ema_tracker_smoothing(self, small_config):
        """EMA target should differ from raw hidden state after multiple updates."""
        ema = EMATargetTracker(small_config, torch.device('cpu'))

        # Simulate multiple updates with different hidden states
        for _ in range(10):
            hidden_states = [torch.randn(2, 16, small_config.n_embd) for _ in range(small_config.n_layer + 1)]
            ema.update(hidden_states)

        # EMA state should exist and differ from latest raw state
        for layer_idx in small_config.fca_layers:
            target_idx = layer_idx + 1
            assert ema.ema_states[layer_idx] is not None
            raw = hidden_states[target_idx]
            ema_val = ema.ema_states[layer_idx]
            assert not torch.allclose(raw, ema_val, atol=1e-3)


class TestSmokeOverfit:
    """Smoke test: verify loss decreases when overfitting a single batch."""

    def test_overfit_single_batch(self, small_config):
        model = FCAGPT(small_config)
        ema = EMATargetTracker(small_config, torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        idx = torch.randint(0, small_config.vocab_size, (4, 16))
        targets = torch.randint(0, small_config.vocab_size, (4, 16))

        losses = []
        for step in range(50):
            _, ce_loss, aux = model(idx, targets)
            ema.update(aux['hidden_states'])
            total, _, _, _ = compute_total_loss(
                ce_loss, aux['future_preds'], aux['hidden_states'],
                ema, small_config, step=step,
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            losses.append(total.item())

        # Loss should decrease meaningfully (at least 30% reduction)
        assert losses[-1] < losses[0] * 0.7, f"Loss didn't decrease enough: {losses[0]:.4f} → {losses[-1]:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
