"""Extract hidden states from a trained model for probe training.

Works with both nanoGPT's GPT and our FCAGPT. Uses forward hooks to
capture hidden states at every layer without modifying model code.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import GPT, GPTConfig
from fca.model import FCAGPT
from fca.config import FCAConfig


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load a model from checkpoint. Auto-detects GPT vs FCAGPT."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'fca_config' in checkpoint:
        # FCAGPT model
        config = checkpoint['fca_config']
        model = FCAGPT(config)
    else:
        # Standard nanoGPT model
        model_args = checkpoint['model_args']
        config = GPTConfig(**model_args)
        model = GPT(config)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def extract_hidden_states_fcagpt(
    model: FCAGPT,
    data: np.ndarray,
    block_size: int,
    n_sequences: int,
    batch_size: int = 32,
    device: str = 'cpu',
) -> dict:
    """Extract hidden states from an FCAGPT model.

    Returns dict mapping layer_idx → tensor of shape (n_sequences * block_size, n_embd),
    plus 'tokens' → (n_sequences * block_size,) for target alignment.
    """
    all_hidden = {i: [] for i in range(model.config.n_layer + 1)}
    all_tokens = []

    indices = torch.randint(len(data) - block_size - 1, (n_sequences,))

    for batch_start in range(0, n_sequences, batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        x = torch.stack([
            torch.from_numpy(data[i:i + block_size].astype(np.int64))
            for i in batch_indices
        ]).to(device)
        # Tokens at positions 1..block_size (the "next tokens" for each position)
        tokens = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
            for i in batch_indices
        ])

        _, _, aux = model(x)

        for layer_idx, h in enumerate(aux['hidden_states']):
            all_hidden[layer_idx].append(h.cpu())
        all_tokens.append(tokens)

    result = {}
    for layer_idx in all_hidden:
        result[layer_idx] = torch.cat(all_hidden[layer_idx], dim=0).reshape(-1, model.config.n_embd)
    result['tokens'] = torch.cat(all_tokens, dim=0).reshape(-1)

    return result


@torch.no_grad()
def extract_hidden_states_gpt(
    model: GPT,
    data: np.ndarray,
    block_size: int,
    n_sequences: int,
    batch_size: int = 32,
    device: str = 'cpu',
) -> dict:
    """Extract hidden states from a standard nanoGPT model using forward hooks.

    Returns dict mapping layer_idx → tensor of shape (n_sequences * block_size, n_embd),
    plus 'tokens' for target alignment.
    """
    n_layers = model.config.n_layer
    captured = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            captured[layer_idx] = output.detach().cpu()
        return hook

    # Register hooks on each Block
    hooks = []
    for i, block in enumerate(model.transformer.h):
        # layer i+1 because index 0 is post-embedding
        hooks.append(block.register_forward_hook(make_hook(i + 1)))

    all_hidden = {i: [] for i in range(n_layers + 1)}
    all_tokens = []

    indices = torch.randint(len(data) - block_size - 1, (n_sequences,))

    for batch_start in range(0, n_sequences, batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        x = torch.stack([
            torch.from_numpy(data[i:i + block_size].astype(np.int64))
            for i in batch_indices
        ]).to(device)
        tokens = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
            for i in batch_indices
        ])

        # Forward pass — hooks capture intermediate states
        # Capture post-embedding state manually
        pos = torch.arange(0, block_size, dtype=torch.long, device=device)
        tok_emb = model.transformer.wte(x)
        pos_emb = model.transformer.wpe(pos)
        post_embed = model.transformer.drop(tok_emb + pos_emb)
        all_hidden[0].append(post_embed.cpu())

        # Run through blocks (hooks fire automatically)
        h = post_embed
        for block in model.transformer.h:
            h = block(h)

        for layer_idx in range(1, n_layers + 1):
            all_hidden[layer_idx].append(captured[layer_idx])

        all_tokens.append(tokens)
        captured.clear()

    # Cleanup hooks
    for hook in hooks:
        hook.remove()

    n_embd = model.config.n_embd
    result = {}
    for layer_idx in all_hidden:
        result[layer_idx] = torch.cat(all_hidden[layer_idx], dim=0).reshape(-1, n_embd)
    result['tokens'] = torch.cat(all_tokens, dim=0).reshape(-1)

    return result


def extract_hidden_states(model, data, block_size, n_sequences, batch_size=32, device='cpu'):
    """Dispatch to the correct extraction function based on model type."""
    if isinstance(model, FCAGPT):
        return extract_hidden_states_fcagpt(model, data, block_size, n_sequences, batch_size, device)
    else:
        return extract_hidden_states_gpt(model, data, block_size, n_sequences, batch_size, device)


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states from a trained model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='nanoGPT/data/openwebtext/val.bin',
                        help='Path to tokenized data (.bin)')
    parser.add_argument('--output_dir', type=str, default='results/hidden_states',
                        help='Directory to save extracted states')
    parser.add_argument('--n_sequences', type=int, default=1000,
                        help='Number of sequences to extract from')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, args.device)
    block_size = config.block_size if hasattr(config, 'block_size') else 1024

    print(f"Loading data from {args.data_path}")
    data = np.memmap(args.data_path, dtype=np.uint16, mode='r')

    print(f"Extracting hidden states for {args.n_sequences} sequences...")
    result = extract_hidden_states(
        model, data, block_size, args.n_sequences,
        batch_size=args.batch_size, device=args.device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'hidden_states.pt')
    print(f"Saving to {output_path}")
    torch.save(result, output_path)

    # Print summary
    n_layers = max(k for k in result if isinstance(k, int))
    n_tokens = result['tokens'].shape[0]
    print(f"Extracted {n_layers + 1} layers, {n_tokens} token positions total")


if __name__ == '__main__':
    main()
