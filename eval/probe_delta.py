"""Compute and visualize the probe accuracy delta between FCA models and baseline.

The key success metric: at which layer depth does future token extractability
peak? FCA should shift this leftward (earlier layers) compared to baseline.
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def compute_deltas(baseline_results: dict, fca_results: dict) -> dict:
    """Compute per-(layer, lookahead) accuracy delta: FCA - baseline.

    Also computes the "threshold crossing layer" — the earliest layer where
    probe accuracy reaches 50% of the peak accuracy for that lookahead.
    """
    baseline_matrix = np.array(baseline_results['matrix'])
    fca_matrix = np.array(fca_results['matrix'])
    delta_matrix = fca_matrix - baseline_matrix
    lookaheads = baseline_results['lookaheads']

    threshold_pct = 0.5  # 50% of peak

    baseline_crossing = {}
    fca_crossing = {}

    for j, k in enumerate(lookaheads):
        baseline_peak = baseline_matrix[:, j].max()
        fca_peak = fca_matrix[:, j].max()

        baseline_threshold = baseline_peak * threshold_pct
        fca_threshold = fca_peak * threshold_pct

        # First layer crossing the threshold
        bl_cross = next((i for i, v in enumerate(baseline_matrix[:, j]) if v >= baseline_threshold), -1)
        fca_cross = next((i for i, v in enumerate(fca_matrix[:, j]) if v >= fca_threshold), -1)

        baseline_crossing[f"k={k}"] = bl_cross
        fca_crossing[f"k={k}"] = fca_cross

    return {
        'delta_matrix': delta_matrix.tolist(),
        'baseline_crossing': baseline_crossing,
        'fca_crossing': fca_crossing,
        'lookaheads': lookaheads,
    }


def print_delta_table(baseline_results: dict, fca_results: dict, fca_name: str):
    """Print a formatted delta table."""
    baseline_matrix = np.array(baseline_results['matrix'])
    fca_matrix = np.array(fca_results['matrix'])
    delta = fca_matrix - baseline_matrix
    lookaheads = baseline_results['lookaheads']

    print(f"\nAccuracy delta ({fca_name} - baseline):")
    print(f"{'Layer':>8}", end="")
    for k in lookaheads:
        print(f"  k={k:>4}", end="")
    print()
    print("-" * (8 + 8 * len(lookaheads)))
    for i in range(delta.shape[0]):
        print(f"{i:>8}", end="")
        for j in range(delta.shape[1]):
            d = delta[i, j]
            sign = "+" if d >= 0 else ""
            print(f"  {sign}{d:>.4f}", end="")
        print()

    # Threshold crossing
    deltas = compute_deltas(baseline_results, fca_results)
    print(f"\n50%-of-peak threshold crossing layer:")
    print(f"{'':>12} {'Baseline':>10} {fca_name:>10} {'Shift':>8}")
    for k in lookaheads:
        bl = deltas['baseline_crossing'][f"k={k}"]
        fc = deltas['fca_crossing'][f"k={k}"]
        shift = bl - fc if bl >= 0 and fc >= 0 else "N/A"
        print(f"  k={k:>4}     {bl:>10} {fc:>10} {str(shift):>8}")


def main():
    parser = argparse.ArgumentParser(description="Compute probe accuracy deltas")
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline probe results JSON')
    parser.add_argument('--fca', type=str, nargs='+', required=True,
                        help='Paths to FCA probe results JSON(s)')
    parser.add_argument('--fca_names', type=str, nargs='+', default=None)
    parser.add_argument('--output_dir', type=str, default='results/eval')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.baseline) as f:
        baseline = json.load(f)

    names = args.fca_names or [os.path.basename(p).replace('_probes.json', '') for p in args.fca]

    for fca_path, name in zip(args.fca, names):
        with open(fca_path) as f:
            fca = json.load(f)
        print_delta_table(baseline, fca, name)

        deltas = compute_deltas(baseline, fca)
        delta_path = os.path.join(args.output_dir, f'{name}_delta.json')
        with open(delta_path, 'w') as f:
            json.dump(deltas, f, indent=2)
        print(f"\nDelta saved to {delta_path}")


if __name__ == '__main__':
    main()
