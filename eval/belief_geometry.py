"""PCA analysis of z vectors to visualize belief state geometry.

Checks whether the FCA bottleneck z develops structured geometric clusters
matching theoretical belief-state geometry, compared to raw hidden states
at the same depth in the baseline model.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from probes.extract import load_model
from fca.model import FCAGPT


@torch.no_grad()
def collect_z_vectors(
    model: FCAGPT,
    data_path: str,
    block_size: int = 1024,
    batch_size: int = 8,
    n_batches: int = 50,
    device: str = 'cuda',
) -> dict:
    """Collect z vectors from all FCA layers."""
    from contextlib import nullcontext

    model.eval()
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    z_collections = {}

    for _ in range(n_batches):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix]).to(device)

        with ctx:
            _, _, aux = model(x)

        for layer_idx, z in aux['z_vectors'].items():
            if layer_idx not in z_collections:
                z_collections[layer_idx] = []
            # Sample positions to keep memory manageable
            z_flat = z.float().cpu().reshape(-1, z.shape[-1])
            # Subsample to ~1000 vectors per batch
            n = min(1000, z_flat.shape[0])
            perm = torch.randperm(z_flat.shape[0])[:n]
            z_collections[layer_idx].append(z_flat[perm])

    return {k: torch.cat(v).numpy() for k, v in z_collections.items()}


def run_pca(z_dict: dict, n_components: int = 3) -> dict:
    """Run PCA on z vectors for each layer. Returns explained variance ratios."""
    from sklearn.decomposition import PCA

    results = {}
    for layer_idx, z in z_dict.items():
        pca = PCA(n_components=n_components)
        z_proj = pca.fit_transform(z)
        results[layer_idx] = {
            'projected': z_proj,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_explained': sum(pca.explained_variance_ratio_).item(),
        }
    return results


def plot_pca(pca_results: dict, output_dir: str, model_name: str):
    """Plot 2D PCA projections of z vectors."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    layers = sorted(pca_results.keys())
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, layers):
        proj = pca_results[layer_idx]['projected']
        var = pca_results[layer_idx]['explained_variance_ratio']
        ax.scatter(proj[:, 0], proj[:, 1], s=1, alpha=0.3)
        ax.set_xlabel(f'PC1 ({var[0]:.1%})')
        ax.set_ylabel(f'PC2 ({var[1]:.1%})')
        ax.set_title(f'Layer {layer_idx} (total: {pca_results[layer_idx]["total_explained"]:.1%})')

    fig.suptitle(f'PCA of z vectors — {model_name}', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_name}_belief_geometry.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="PCA analysis of FCA z vectors")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='nanoGPT/data/openwebtext/val.bin')
    parser.add_argument('--output_dir', type=str, default='results/eval')
    parser.add_argument('--model_name', type=str, default='fca-top-third')
    parser.add_argument('--n_batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, config = load_model(args.checkpoint, args.device)
    if not isinstance(model, FCAGPT):
        print("Error: checkpoint is not an FCAGPT model")
        sys.exit(1)

    print("Collecting z vectors...")
    z_dict = collect_z_vectors(
        model, args.data_path,
        block_size=config.block_size,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        device=args.device,
    )

    print("Running PCA...")
    pca_results = run_pca(z_dict)

    # Print summary
    for layer_idx in sorted(pca_results.keys()):
        var = pca_results[layer_idx]['explained_variance_ratio']
        total = pca_results[layer_idx]['total_explained']
        print(f"  Layer {layer_idx}: PC1={var[0]:.3f}, PC2={var[1]:.3f}, PC3={var[2]:.3f}, total={total:.3f}")

    # Save variance info
    stats = {str(k): {kk: vv for kk, vv in v.items() if kk != 'projected'} for k, v in pca_results.items()}
    stats_path = os.path.join(args.output_dir, f'{args.model_name}_belief_geometry.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    plot_pca(pca_results, args.output_dir, args.model_name)


if __name__ == '__main__':
    main()
