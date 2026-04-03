"""Train and evaluate linear probes across all (layer, lookahead) pairs.

For each layer l and lookahead k, trains a linear probe to predict the token
at position t+k from the hidden state at layer l, position t.

Outputs a probe accuracy matrix and generates comparison plots.
"""

import os
import sys
import json
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from probes.probe import LinearProbe, train_probe


LOOKAHEADS = [1, 2, 4, 8]


def prepare_probe_data(
    hidden_states: dict,
    layer_idx: int,
    lookahead: int,
    block_size: int,
) -> tuple:
    """Prepare (hidden_state, target_token) pairs for a given (layer, lookahead).

    Hidden states and tokens are stored flat (n_sequences * block_size).
    We need to align position t with the token at position t+k, being careful
    not to cross sequence boundaries.

    Args:
        hidden_states: dict from extract.py with layer tensors and 'tokens'
        layer_idx: which layer's hidden states to use
        lookahead: k — predict token this many positions ahead
        block_size: sequence length (to avoid crossing boundaries)

    Returns:
        (h, targets) where h is (N_valid, hidden_dim) and targets is (N_valid,)
    """
    h_flat = hidden_states[layer_idx]  # (total_positions, hidden_dim)
    tokens_flat = hidden_states['tokens']  # (total_positions,)

    total = h_flat.shape[0]
    n_sequences = total // block_size

    # For each sequence, positions 0..block_size-1-lookahead are valid
    valid_per_seq = block_size - lookahead
    if valid_per_seq <= 0:
        raise ValueError(f"lookahead {lookahead} >= block_size {block_size}")

    valid_h = []
    valid_targets = []

    for seq_idx in range(n_sequences):
        start = seq_idx * block_size
        # Hidden states at positions 0..valid_per_seq-1
        h_seq = h_flat[start:start + valid_per_seq]
        # Target tokens at positions lookahead..block_size-1
        t_seq = tokens_flat[start + lookahead:start + block_size]
        valid_h.append(h_seq)
        valid_targets.append(t_seq)

    return torch.cat(valid_h, dim=0), torch.cat(valid_targets, dim=0)


def run_all_probes(
    hidden_states: dict,
    n_layers: int,
    block_size: int,
    vocab_size: int,
    hidden_dim: int,
    device: str = 'cpu',
    probe_lr: float = 1e-3,
    probe_epochs: int = 10,
    probe_batch_size: int = 4096,
) -> dict:
    """Train probes for all (layer, lookahead) combinations.

    Returns:
        accuracy_matrix: dict with keys like "layer_3_k_2" → accuracy float
        Also includes 'matrix' key: list of lists [n_layers+1][len(LOOKAHEADS)]
    """
    results = {}
    matrix = []

    for layer_idx in range(n_layers + 1):
        row = []
        for k in LOOKAHEADS:
            print(f"  Training probe: layer={layer_idx}, k={k}...", end=" ", flush=True)

            h, targets = prepare_probe_data(hidden_states, layer_idx, k, block_size)
            probe = LinearProbe(hidden_dim, vocab_size)
            info = train_probe(
                probe, h, targets,
                lr=probe_lr,
                epochs=probe_epochs,
                batch_size=probe_batch_size,
                device=device,
            )

            acc = info['final_accuracy']
            key = f"layer_{layer_idx}_k_{k}"
            results[key] = acc
            row.append(acc)
            print(f"accuracy={acc:.4f}")

        matrix.append(row)

    results['matrix'] = matrix
    results['layers'] = list(range(n_layers + 1))
    results['lookaheads'] = LOOKAHEADS
    return results


def plot_probe_results(results: dict, output_path: str, title: str = "Probe Accuracy"):
    """Plot probe accuracy curves: one line per lookahead k."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot generation")
        return

    matrix = np.array(results['matrix'])
    layers = results['layers']
    lookaheads = results['lookaheads']

    fig, ax = plt.subplots(figsize=(10, 6))
    for j, k in enumerate(lookaheads):
        ax.plot(layers, matrix[:, j], marker='o', label=f'k={k}')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_comparison(
    results_list: list,
    labels: list,
    output_path: str,
    title: str = "Probe Accuracy Comparison",
):
    """Overlay probe curves from multiple models for comparison.

    Args:
        results_list: list of results dicts (from run_all_probes)
        labels: list of model names
        output_path: where to save the plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot generation")
        return

    lookaheads = results_list[0]['lookaheads']
    n_lookaheads = len(lookaheads)

    fig, axes = plt.subplots(1, n_lookaheads, figsize=(5 * n_lookaheads, 5))
    if n_lookaheads == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for j, k in enumerate(lookaheads):
        ax = axes[j]
        for idx, (results, label) in enumerate(zip(results_list, labels)):
            matrix = np.array(results['matrix'])
            layers = results['layers']
            ax.plot(layers, matrix[:, j], marker='o', color=colors[idx % 10], label=label)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Top-1 Accuracy')
        ax.set_title(f'Lookahead k={k}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train linear probes on extracted hidden states")
    parser.add_argument('--hidden_states', type=str, required=True,
                        help='Path to hidden_states.pt from extract.py')
    parser.add_argument('--output_dir', type=str, default='results/probes',
                        help='Directory to save probe results')
    parser.add_argument('--model_name', type=str, default='baseline',
                        help='Name for this model (used in filenames)')
    parser.add_argument('--vocab_size', type=int, default=50304)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--probe_lr', type=float, default=1e-3)
    parser.add_argument('--probe_epochs', type=int, default=10)
    parser.add_argument('--probe_batch_size', type=int, default=4096)

    # Comparison mode
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                        help='Paths to multiple probe result JSON files for comparison plotting')
    parser.add_argument('--compare_labels', type=str, nargs='+', default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Comparison mode
    if args.compare:
        results_list = []
        for path in args.compare:
            with open(path) as f:
                results_list.append(json.load(f))
        labels = args.compare_labels or [os.path.basename(p).replace('.json', '') for p in args.compare]
        plot_comparison(
            results_list, labels,
            os.path.join(args.output_dir, 'comparison.png'),
        )
        return

    # Training mode
    print(f"Loading hidden states from {args.hidden_states}")
    hidden_states = torch.load(args.hidden_states, map_location='cpu', weights_only=False)

    print(f"Training {(args.n_layers + 1) * len(LOOKAHEADS)} probes...")
    results = run_all_probes(
        hidden_states,
        n_layers=args.n_layers,
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        device=args.device,
        probe_lr=args.probe_lr,
        probe_epochs=args.probe_epochs,
        probe_batch_size=args.probe_batch_size,
    )

    # Save results
    results_path = os.path.join(args.output_dir, f'{args.model_name}_probes.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Plot
    plot_path = os.path.join(args.output_dir, f'{args.model_name}_probes.png')
    plot_probe_results(results, plot_path, title=f"Probe Accuracy — {args.model_name}")

    # Print summary table
    print(f"\n{'':>10}", end="")
    for k in LOOKAHEADS:
        print(f"  k={k:>4}", end="")
    print()
    for i, row in enumerate(results['matrix']):
        print(f"Layer {i:>3}", end="")
        for acc in row:
            print(f"  {acc:>.4f}", end="")
        print()


if __name__ == '__main__':
    main()
