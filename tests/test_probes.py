"""Tests for the probing harness."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
import numpy as np

from fca.config import FCAConfig
from fca.model import FCAGPT
from probes.probe import LinearProbe, train_probe
from probes.extract import extract_hidden_states
from probes.train_probes import prepare_probe_data, run_all_probes


@pytest.fixture
def tiny_config():
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


@pytest.fixture
def tiny_model(tiny_config):
    model = FCAGPT(tiny_config)
    model.eval()
    return model


@pytest.fixture
def fake_data():
    """Fake tokenized data as a numpy array."""
    return np.random.randint(0, 64, size=(2000,), dtype=np.uint16)


class TestLinearProbe:
    def test_output_shape(self):
        probe = LinearProbe(32, 64)
        h = torch.randn(10, 32)
        logits = probe(h)
        assert logits.shape == (10, 64)

    def test_training(self):
        probe = LinearProbe(32, 64)
        h = torch.randn(500, 32)
        targets = torch.randint(0, 64, (500,))
        result = train_probe(probe, h, targets, epochs=3, batch_size=128)
        assert 'final_accuracy' in result
        assert 0.0 <= result['final_accuracy'] <= 1.0
        assert len(result['train_losses']) == 3


class TestExtraction:
    def test_extract_fcagpt(self, tiny_model, fake_data, tiny_config):
        result = extract_hidden_states(
            tiny_model, fake_data,
            block_size=tiny_config.block_size,
            n_sequences=5,
            batch_size=2,
            device='cpu',
        )
        n_tokens = 5 * tiny_config.block_size
        assert result['tokens'].shape == (n_tokens,)
        for layer_idx in range(tiny_config.n_layer + 1):
            assert layer_idx in result
            assert result[layer_idx].shape == (n_tokens, tiny_config.n_embd)


class TestProbeDataPrep:
    def test_alignment(self, tiny_model, fake_data, tiny_config):
        result = extract_hidden_states(
            tiny_model, fake_data,
            block_size=tiny_config.block_size,
            n_sequences=5,
            batch_size=2,
            device='cpu',
        )
        h, targets = prepare_probe_data(result, layer_idx=2, lookahead=2, block_size=tiny_config.block_size)
        expected_per_seq = tiny_config.block_size - 2
        assert h.shape[0] == 5 * expected_per_seq
        assert targets.shape[0] == 5 * expected_per_seq


class TestEndToEnd:
    def test_full_probe_pipeline(self, tiny_model, fake_data, tiny_config):
        """Extract states → train probes → get accuracy matrix."""
        result = extract_hidden_states(
            tiny_model, fake_data,
            block_size=tiny_config.block_size,
            n_sequences=5,
            batch_size=2,
            device='cpu',
        )
        probe_results = run_all_probes(
            result,
            n_layers=tiny_config.n_layer,
            block_size=tiny_config.block_size,
            vocab_size=tiny_config.vocab_size,
            hidden_dim=tiny_config.n_embd,
            device='cpu',
            probe_epochs=2,
            probe_batch_size=64,
        )
        matrix = probe_results['matrix']
        assert len(matrix) == tiny_config.n_layer + 1  # layers 0..n_layer
        assert len(matrix[0]) == 4  # 4 lookaheads
        for row in matrix:
            for acc in row:
                assert 0.0 <= acc <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
