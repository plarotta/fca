"""Generate a tiny synthetic dataset in nanoGPT's binary format for smoke testing.

Creates train.bin and val.bin with random token IDs in the same uint16 memmap
format that nanoGPT expects. Also creates a meta.pkl with vocab_size.
"""

import os
import pickle
import argparse

import numpy as np


def generate(output_dir: str, n_train: int = 50_000, n_val: int = 5_000, vocab_size: int = 256):
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    # Add some structure: repeating patterns so models can actually learn something
    # Mix of random tokens and repeated n-gram patterns
    def make_data(n_tokens):
        data = np.zeros(n_tokens, dtype=np.uint16)
        pos = 0
        while pos < n_tokens:
            if rng.random() < 0.5:
                # Random chunk
                chunk_len = min(rng.integers(10, 100), n_tokens - pos)
                data[pos:pos + chunk_len] = rng.integers(0, vocab_size, size=chunk_len, dtype=np.uint16)
            else:
                # Repeating pattern (so probes and loss can find signal)
                pattern_len = rng.integers(2, 8)
                pattern = rng.integers(0, vocab_size, size=pattern_len, dtype=np.uint16)
                repeats = min(rng.integers(3, 20), (n_tokens - pos) // pattern_len)
                chunk = np.tile(pattern, repeats)
                chunk_len = len(chunk)
                data[pos:pos + chunk_len] = chunk
            pos += chunk_len
        return data

    train_data = make_data(n_train)
    val_data = make_data(n_val)

    train_path = os.path.join(output_dir, 'train.bin')
    val_path = os.path.join(output_dir, 'val.bin')
    meta_path = os.path.join(output_dir, 'meta.pkl')

    train_data.tofile(train_path)
    val_data.tofile(val_path)

    with open(meta_path, 'wb') as f:
        pickle.dump({'vocab_size': vocab_size}, f)

    print(f"Generated mini dataset in {output_dir}:")
    print(f"  train.bin: {n_train} tokens ({os.path.getsize(train_path)} bytes)")
    print(f"  val.bin:   {n_val} tokens ({os.path.getsize(val_path)} bytes)")
    print(f"  vocab_size: {vocab_size}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='nanoGPT/data/mini')
    parser.add_argument('--n_train', type=int, default=50_000)
    parser.add_argument('--n_val', type=int, default=5_000)
    parser.add_argument('--vocab_size', type=int, default=256)
    args = parser.parse_args()
    generate(args.output_dir, args.n_train, args.n_val, args.vocab_size)
