"""Compute validation perplexity for one or more model checkpoints."""

import os
import sys
import math
import argparse
from contextlib import nullcontext

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from probes.extract import load_model


@torch.no_grad()
def compute_perplexity(
    model,
    data_path: str,
    block_size: int = 1024,
    batch_size: int = 8,
    eval_iters: int = 200,
    device: str = 'cuda',
) -> float:
    """Compute perplexity over random chunks of the data."""
    model.eval()
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    losses = []
    for _ in range(eval_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix]).to(device)

        with ctx:
            out = model(x, y)
            # Handle both GPT (logits, loss) and FCAGPT (logits, loss, aux)
            if len(out) == 3:
                _, loss, _ = out
            else:
                _, loss = out
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser(description="Compute validation perplexity")
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True)
    parser.add_argument('--names', type=str, nargs='+', default=None)
    parser.add_argument('--data_path', type=str, default='nanoGPT/data/openwebtext/val.bin')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    names = args.names or [os.path.dirname(c).split('/')[-1] for c in args.checkpoints]

    print(f"{'Model':<25} {'Perplexity':>12}")
    print("-" * 40)

    for ckpt_path, name in zip(args.checkpoints, names):
        model, config = load_model(ckpt_path, args.device)
        block_size = config.block_size if hasattr(config, 'block_size') else 1024
        ppl = compute_perplexity(
            model, args.data_path,
            block_size=block_size,
            batch_size=args.batch_size,
            eval_iters=args.eval_iters,
            device=args.device,
        )
        print(f"{name:<25} {ppl:>12.2f}")


if __name__ == '__main__':
    main()
