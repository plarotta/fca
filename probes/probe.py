"""Linear probe for future token prediction from frozen hidden states.

A single affine layer that takes a hidden state at layer l, position t
and predicts the token at position t+k.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class LinearProbe(nn.Module):
    """Linear probe: hidden_state → token prediction.

    Args:
        hidden_dim: dimension of input hidden states (n_embd)
        vocab_size: number of output classes
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, hidden_dim) — hidden states at specific positions
        Returns:
            logits: (batch, vocab_size)
        """
        return self.linear(h)


def train_probe(
    probe: LinearProbe,
    hidden_states: torch.Tensor,
    target_tokens: torch.Tensor,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 4096,
    device: str = 'cpu',
) -> dict:
    """Train a linear probe on frozen hidden states.

    Args:
        probe: LinearProbe model
        hidden_states: (N, hidden_dim) — all extracted hidden states
        target_tokens: (N,) — corresponding target token ids
        lr: learning rate
        epochs: training epochs
        batch_size: mini-batch size
        device: device to train on

    Returns:
        dict with 'train_losses' and 'final_accuracy'
    """
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    N = hidden_states.shape[0]
    # 80/20 split
    split = int(0.8 * N)
    perm = torch.randperm(N)
    train_idx, val_idx = perm[:split], perm[split:]

    train_h, train_t = hidden_states[train_idx], target_tokens[train_idx]
    val_h, val_t = hidden_states[val_idx], target_tokens[val_idx]

    train_losses = []

    for epoch in range(epochs):
        probe.train()
        epoch_loss = 0.0
        n_batches = 0

        shuffle = torch.randperm(train_h.shape[0])
        train_h, train_t = train_h[shuffle], train_t[shuffle]

        for i in range(0, train_h.shape[0], batch_size):
            h_batch = train_h[i:i + batch_size].to(device)
            t_batch = train_t[i:i + batch_size].to(device)

            logits = probe(h_batch)
            loss = F.cross_entropy(logits, t_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

    # Evaluate on validation set
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, val_h.shape[0], batch_size):
            h_batch = val_h[i:i + batch_size].to(device)
            t_batch = val_t[i:i + batch_size].to(device)
            logits = probe(h_batch)
            preds = logits.argmax(dim=-1)
            correct += (preds == t_batch).sum().item()
            total += t_batch.shape[0]

    accuracy = correct / max(total, 1)
    return {'train_losses': train_losses, 'final_accuracy': accuracy}
