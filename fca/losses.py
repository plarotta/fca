"""
FCA loss functions: future hidden state prediction with EMA targets and lambda scheduling.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from fca.config import FCAConfig


class EMATargetTracker:
    """Maintains exponential moving average of hidden states for stable future loss targets.

    For each FCA layer l, tracks EMA of the hidden state at layer l+1 (the target).
    The EMA smooths out the rapidly fluctuating hidden states during early training,
    preventing the bottleneck z from chasing noisy targets.
    """

    def __init__(self, config: FCAConfig, device: torch.device):
        self.decay = config.ema_decay
        self.enabled = config.use_ema_target
        self.fca_layers = config.fca_layers
        # EMA states are lazily initialized on first update
        self.ema_states: Dict[int, Optional[torch.Tensor]] = {l: None for l in self.fca_layers}

    @torch.no_grad()
    def update(self, hidden_states: List[torch.Tensor]):
        """Update EMA states with new hidden states from the current forward pass.

        Args:
            hidden_states: list where hidden_states[i] is the output of layer i
                          (index 0 = post-embedding, index n_layer = final layer output)
        """
        if not self.enabled:
            return

        for layer_idx in self.fca_layers:
            target_idx = layer_idx + 1
            if target_idx >= len(hidden_states):
                continue

            h = hidden_states[target_idx].detach()

            if self.ema_states[layer_idx] is None:
                self.ema_states[layer_idx] = h.clone()
            else:
                self.ema_states[layer_idx].mul_(self.decay).add_(h, alpha=1.0 - self.decay)

    def get_target(self, layer_idx: int, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Get the target hidden state for a given FCA layer.

        Returns EMA-smoothed target if enabled, otherwise raw stop-grad target.
        """
        target_idx = layer_idx + 1
        if self.enabled and self.ema_states[layer_idx] is not None:
            return self.ema_states[layer_idx].detach()
        else:
            return hidden_states[target_idx].detach()


def lambda_schedule(step: int, warmup_steps: int, use_schedule: bool) -> float:
    """Compute the lambda multiplier for the future loss.

    Linear ramp from 0 to 1 over warmup_steps, then constant 1.0.
    If use_schedule is False, returns 1.0 always (no warmup).
    """
    if not use_schedule:
        return 1.0
    return min(1.0, step / max(warmup_steps, 1))


def compute_future_loss(
    future_preds: Dict[int, torch.Tensor],
    hidden_states: List[torch.Tensor],
    ema_tracker: EMATargetTracker,
    config: FCAConfig,
) -> torch.Tensor:
    """Compute the future hidden state prediction loss across all FCA layers.

    For each FCA layer l at position t:
        target = stop_grad(EMA(h_{l+1}[t + future_offset]))
        prediction = future_proj(z_l[t])
        loss = MSE(prediction, target)

    The temporal shift (future_offset) means position t's prediction is compared
    against the hidden state at position t+1 in the next layer.

    Args:
        future_preds: dict of layer_idx → projected z tensors (B, T, n_embd)
        hidden_states: list of per-layer hidden states
        ema_tracker: EMA target state manager
        config: FCA configuration

    Returns:
        Scalar MSE loss averaged across layers, positions, and dimensions.
    """
    if not future_preds:
        return torch.tensor(0.0, device=hidden_states[0].device)

    total_loss = torch.tensor(0.0, device=hidden_states[0].device)
    n_layers = 0
    offset = config.future_offset

    for layer_idx, pred in future_preds.items():
        target = ema_tracker.get_target(layer_idx, hidden_states)

        # Temporal shift: pred at position t predicts target at position t+offset
        # Trim sequences to align: pred[:, :T-offset] vs target[:, offset:]
        B, T, D = pred.shape
        if T <= offset:
            continue

        pred_aligned = pred[:, :T - offset]       # (B, T-offset, D)
        target_aligned = target[:, offset:]        # (B, T-offset, D)

        layer_loss = F.mse_loss(pred_aligned, target_aligned)
        total_loss = total_loss + layer_loss
        n_layers += 1

    if n_layers == 0:
        return total_loss

    return total_loss / n_layers


def compute_total_loss(
    ce_loss: torch.Tensor,
    future_preds: Dict[int, torch.Tensor],
    hidden_states: List[torch.Tensor],
    ema_tracker: EMATargetTracker,
    config: FCAConfig,
    step: int,
) -> tuple:
    """Compute combined loss: CE + lambda * future_loss.

    Returns:
        (total_loss, ce_loss, future_loss, current_lambda) for logging.
    """
    future_loss = compute_future_loss(future_preds, hidden_states, ema_tracker, config)
    lam = lambda_schedule(step, config.lambda_warmup_steps, config.use_lambda_schedule)
    total = ce_loss + config.future_loss_weight * lam * future_loss

    return total, ce_loss, future_loss, lam
