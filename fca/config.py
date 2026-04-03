"""FCA experiment configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


# Sparse placement presets for 12-layer GPT-2 Small
PLACEMENT_PRESETS = {
    "top_third": [8, 9, 10, 11],
    "top_quarter": [9, 10, 11],
    "last_4": [8, 9, 10, 11],  # same as top_third for 12 layers
    "all_layers": list(range(12)),
}


@dataclass
class FCAConfig:
    """Configuration for the FCA architecture on top of nanoGPT's GPTConfig."""

    # --- Base model (mirrors nanoGPT GPTConfig) ---
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    # --- FCA architecture ---
    fca_layers: List[int] = field(default_factory=lambda: [8, 9, 10, 11])
    bottleneck_dim: int = 192  # n_embd // 4
    bottleneck_hidden_dim: Optional[int] = None  # defaults to n_embd // 2
    fca_n_head: int = 12  # heads in the FCA attention branch
    fca_dropout: float = 0.0
    random_z: bool = False  # ablation: replace z with Gaussian noise

    # --- Future loss ---
    future_loss_weight: float = 1.0  # max weight after lambda ramp
    lambda_warmup_steps: int = 20000  # steps to linearly ramp lambda from 0 to 1
    use_lambda_schedule: bool = True  # False = constant weight from step 0
    ema_decay: float = 0.999  # EMA decay for target hidden states
    use_ema_target: bool = True  # False = raw stop-grad targets (no EMA)
    future_offset: int = 1  # predict hidden state this many positions ahead

    def __post_init__(self):
        if self.bottleneck_hidden_dim is None:
            self.bottleneck_hidden_dim = self.n_embd // 2
        # Validate FCA layer indices
        for l in self.fca_layers:
            assert 0 <= l < self.n_layer, f"FCA layer {l} out of range [0, {self.n_layer})"

    @classmethod
    def from_preset(cls, preset: str, **overrides):
        """Create config from a placement preset name."""
        layers = PLACEMENT_PRESETS[preset]
        return cls(fca_layers=list(layers), **overrides)
