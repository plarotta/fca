"""Early Commitment / Confidence Saturation evaluation.

Measures how early a model's prediction confidence for the correct next token
reaches saturation across sequence positions. The key metric is AULC
(Area Under the Learning Curve) — higher = earlier commitment.

FCA should shift the confidence curve leftward compared to the baseline.
"""

import os
import sys
import json
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from probes.extract import load_model


@torch.no_grad()
def compute_confidence_curves(
    model,
    data_path: str,
    block_size: int = 1024,
    n_sequences: int = 1000,
    batch_size: int = 8,
    device: str = 'cuda',
) -> dict:
    """Compute per-position confidence in the correct next token.

    Returns dict with:
        'mean_confidence': (block_size,) — average P(correct) at each position
        'mean_entropy': (block_size,) — average entropy at each position
        'aulc': scalar — area under the confidence curve (normalized to [0,1])
    """
    model.eval()
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Accumulators
    conf_sum = torch.zeros(block_size)
    entropy_sum = torch.zeros(block_size)
    count = 0

    indices = torch.randint(len(data) - block_size - 1, (n_sequences,))

    for batch_start in range(0, n_sequences, batch_size):
        batch_idx = indices[batch_start:batch_start + batch_size]
        actual_bs = len(batch_idx)

        x = torch.stack([
            torch.from_numpy(data[i:i + block_size].astype(np.int64))
            for i in batch_idx
        ]).to(device)
        y = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
            for i in batch_idx
        ]).to(device)

        with ctx:
            out = model(x, y)
            if len(out) == 3:
                logits, _, _ = out
            else:
                logits, _ = out

        # logits: (B, T, vocab_size)
        probs = F.softmax(logits.float(), dim=-1)  # (B, T, V)

        # Confidence: P(correct next token) at each position
        correct_probs = probs.gather(2, y.unsqueeze(-1)).squeeze(-1)  # (B, T)
        conf_sum += correct_probs.sum(dim=0).cpu()

        # Entropy at each position
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (B, T)
        entropy_sum += entropy.sum(dim=0).cpu()

        count += actual_bs

    mean_conf = (conf_sum / count).numpy()
    mean_entropy = (entropy_sum / count).numpy()

    # AULC: area under the confidence curve, normalized by maximum possible area
    aulc = float(np.trapezoid(mean_conf, dx=1.0 / block_size))

    return {
        'mean_confidence': mean_conf.tolist(),
        'mean_entropy': mean_entropy.tolist(),
        'aulc': aulc,
        'n_sequences': count,
        'block_size': block_size,
    }


def plot_confidence_comparison(
    results_list: list,
    labels: list,
    output_path: str,
):
    """Plot confidence curves for multiple models side-by-side."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for results, label in zip(results_list, labels):
        positions = np.arange(len(results['mean_confidence']))
        ax1.plot(positions, results['mean_confidence'], label=f"{label} (AULC={results['aulc']:.4f})", alpha=0.8)
        ax2.plot(positions, results['mean_entropy'], label=label, alpha=0.8)

    ax1.set_xlabel('Sequence Position')
    ax1.set_ylabel('P(correct next token)')
    ax1.set_title('Confidence Saturation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Sequence Position')
    ax2.set_ylabel('Entropy (nats)')
    ax2.set_title('Prediction Entropy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Confidence saturation evaluation")
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True)
    parser.add_argument('--names', type=str, nargs='+', default=None)
    parser.add_argument('--data_path', type=str, default='nanoGPT/data/openwebtext/val.bin')
    parser.add_argument('--output_dir', type=str, default='results/eval')
    parser.add_argument('--n_sequences', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    names = args.names or [os.path.dirname(c).split('/')[-1] for c in args.checkpoints]

    all_results = []

    for ckpt_path, name in zip(args.checkpoints, names):
        print(f"\n=== Computing confidence curves for {name} ===")
        model, config = load_model(ckpt_path, args.device)
        block_size = config.block_size if hasattr(config, 'block_size') else 1024

        results = compute_confidence_curves(
            model, args.data_path,
            block_size=block_size,
            n_sequences=args.n_sequences,
            batch_size=args.batch_size,
            device=args.device,
        )
        results['model_name'] = name

        # Save individual results
        json_path = os.path.join(args.output_dir, f'{name}_confidence.json')
        with open(json_path, 'w') as f:
            json.dump(results, f)
        print(f"  AULC: {results['aulc']:.4f}")

        all_results.append(results)

    # Summary table
    print(f"\n{'Model':<25} {'AULC':>12}")
    print("-" * 40)
    for r in all_results:
        print(f"{r['model_name']:<25} {r['aulc']:>12.4f}")

    # Comparison plot
    if len(all_results) >= 2:
        plot_confidence_comparison(
            all_results, names,
            os.path.join(args.output_dir, 'confidence_comparison.png'),
        )


if __name__ == '__main__':
    main()
