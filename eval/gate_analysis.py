"""Analyze FCA gate values from a trained FCAGPT checkpoint.

Generates histograms and per-layer statistics of learned gate values g.
"""

import os
import sys
import json
import argparse
from contextlib import nullcontext

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from probes.extract import load_model
from fca.model import FCAGPT


@torch.no_grad()
def collect_gate_values(
    model: FCAGPT,
    data_path: str,
    block_size: int = 1024,
    batch_size: int = 8,
    n_batches: int = 50,
    device: str = 'cuda',
) -> dict:
    """Collect gate values across many sequences.

    Returns dict mapping layer_idx → flat tensor of all gate values.
    """
    model.eval()
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    gate_collections = {}

    for _ in range(n_batches):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix]).to(device)

        with ctx:
            _, _, aux = model(x)

        for layer_idx, g in aux['gate_values'].items():
            if layer_idx not in gate_collections:
                gate_collections[layer_idx] = []
            gate_collections[layer_idx].append(g.cpu().reshape(-1))

    return {k: torch.cat(v) for k, v in gate_collections.items()}


def analyze_gates(gate_values: dict, output_dir: str, model_name: str):
    """Compute statistics and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    stats = {}
    for layer_idx in sorted(gate_values.keys()):
        g = gate_values[layer_idx]
        stats[f"layer_{layer_idx}"] = {
            "mean": g.mean().item(),
            "std": g.std().item(),
            "median": g.median().item(),
            "min": g.min().item(),
            "max": g.max().item(),
            "pct_below_0.1": (g < 0.1).float().mean().item(),
            "pct_above_0.9": (g > 0.9).float().mean().item(),
        }

    # Print table
    print(f"\nGate statistics for {model_name}:")
    print(f"{'Layer':>8} {'Mean':>8} {'Std':>8} {'Median':>8} {'<0.1':>8} {'>0.9':>8}")
    print("-" * 56)
    for layer_idx in sorted(gate_values.keys()):
        s = stats[f"layer_{layer_idx}"]
        print(f"{layer_idx:>8} {s['mean']:>8.4f} {s['std']:>8.4f} {s['median']:>8.4f} "
              f"{s['pct_below_0.1']:>8.3f} {s['pct_above_0.9']:>8.3f}")

    # Save stats
    stats_path = os.path.join(output_dir, f'{model_name}_gate_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Plot histograms
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        layers = sorted(gate_values.keys())
        n = len(layers)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, layer_idx in zip(axes, layers):
            g = gate_values[layer_idx].numpy()
            ax.hist(g, bins=50, density=True, alpha=0.7)
            ax.set_title(f'Layer {layer_idx}')
            ax.set_xlabel('Gate value')
            ax.set_ylabel('Density')
            ax.axvline(g.mean(), color='red', linestyle='--', label=f'mean={g.mean():.3f}')
            ax.legend(fontsize=8)

        fig.suptitle(f'Gate Value Distributions — {model_name}', fontsize=14)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{model_name}_gates.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not installed, skipping plot")


def main():
    parser = argparse.ArgumentParser(description="Analyze FCA gate values")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='nanoGPT/data/openwebtext/val.bin')
    parser.add_argument('--output_dir', type=str, default='results/eval')
    parser.add_argument('--model_name', type=str, default='fca-top-third')
    parser.add_argument('--n_batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model, config = load_model(args.checkpoint, args.device)
    if not isinstance(model, FCAGPT):
        print("Error: checkpoint is not an FCAGPT model")
        sys.exit(1)

    gate_values = collect_gate_values(
        model, args.data_path,
        block_size=config.block_size,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        device=args.device,
    )
    analyze_gates(gate_values, args.output_dir, args.model_name)


if __name__ == '__main__':
    main()
