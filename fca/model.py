"""
FCA (Future-Conditioned Attention) architecture modules.

Extends nanoGPT's GPT with a secondary per-layer attention branch conditioned
on a bottlenecked latent z, fused back via a learned sigmoid gate g.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

# Import nanoGPT components
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))
from model import GPTConfig, GPT, Block, LayerNorm, CausalSelfAttention, MLP

from fca.config import FCAConfig


class BottleneckMLP(nn.Module):
    """Projects hidden state h to bottleneck latent z.

    h ∈ R^(B, T, n_embd) → z ∈ R^(B, T, bottleneck_dim)
    Architecture: Linear → GELU → Linear
    """

    def __init__(self, config: FCAConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.bottleneck_hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(config.bottleneck_hidden_dim, config.bottleneck_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.fca_dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.fc1(h)
        z = self.gelu(z)
        z = self.fc2(z)
        z = self.dropout(z)
        return z


class FCAAttention(nn.Module):
    """Future-conditioned cross-attention.

    Queries come from bottleneck z, keys/values come from primary hidden state h.
    Uses causal masking — position t can only attend to positions <= t.
    Output projected back to n_embd dimension.
    """

    def __init__(self, config: FCAConfig):
        super().__init__()
        self.n_head = config.fca_n_head
        self.n_embd = config.n_embd
        self.bottleneck_dim = config.bottleneck_dim
        self.head_dim = config.n_embd // config.fca_n_head
        assert config.n_embd % config.fca_n_head == 0

        # Q from z (bottleneck_dim → n_embd), K/V from h (n_embd → n_embd each)
        self.q_proj = nn.Linear(config.bottleneck_dim, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.fca_dropout)
        self.resid_dropout = nn.Dropout(config.fca_dropout)
        self.dropout_p = config.fca_dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.size()

        q = self.q_proj(z).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            causal_mask = torch.tril(torch.ones(T, T, device=h.device, dtype=torch.bool))
            att = att.masked_fill(~causal_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        y = self.resid_dropout(self.out_proj(y))
        return y


class FCAGate(nn.Module):
    """Learned sigmoid gate for FCA fusion.

    g = sigmoid(Linear(h)) — scalar per position.
    Initialized to output ~0 so FCA branch starts silent.
    """

    def __init__(self, config: FCAConfig):
        super().__init__()
        self.linear = nn.Linear(config.n_embd, 1, bias=True)
        # Initialize bias to -2 so sigmoid(-2) ≈ 0.12, starting nearly closed
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -2.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(h))  # (B, T, 1)


class FCABranch(nn.Module):
    """Complete FCA branch: bottleneck → attention → gate → fusion output.

    Also exposes z for the future prediction loss.
    """

    def __init__(self, config: FCAConfig):
        super().__init__()
        self.bottleneck = BottleneckMLP(config)
        self.attention = FCAAttention(config)
        self.gate = FCAGate(config)
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.random_z = config.random_z
        self.bottleneck_dim = config.bottleneck_dim

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (fca_output to add to residual, z, gate_values)."""
        z = self.bottleneck(h)

        if self.random_z:
            z = torch.randn_like(z)

        fca_out = self.attention(z, self.ln(h))
        g = self.gate(h)
        return g * fca_out, z, g


class FCABlock(nn.Module):
    """Transformer block with optional FCA branch.

    Standard path: h = h + attn(ln1(h)); h = h + mlp(ln2(h))
    FCA augmentation: h = h + g * fca_attention(z, ln(h))
    """

    def __init__(self, config: FCAConfig, has_fca: bool):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(self._to_gpt_config(config))
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(self._to_gpt_config(config))
        self.has_fca = has_fca
        if has_fca:
            self.fca = FCABranch(config)

    @staticmethod
    def _to_gpt_config(fca_config: FCAConfig) -> GPTConfig:
        """Convert FCAConfig to GPTConfig for nanoGPT modules."""
        return GPTConfig(
            block_size=fca_config.block_size,
            vocab_size=fca_config.vocab_size,
            n_layer=fca_config.n_layer,
            n_head=fca_config.n_head,
            n_embd=fca_config.n_embd,
            dropout=fca_config.dropout,
            bias=fca_config.bias,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns (hidden_state, z_or_None, gate_or_None)."""
        # Standard transformer block
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        # FCA augmentation
        z, g = None, None
        if self.has_fca:
            fca_out, z, g = self.fca(x)
            x = x + fca_out

        return x, z, g


class FCAGPT(nn.Module):
    """GPT with Future-Conditioned Attention branches.

    Extends nanoGPT's GPT architecture. Returns hidden states at each layer
    for the future prediction loss, plus z vectors and gate values from FCA layers.
    """

    def __init__(self, config: FCAConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                FCABlock(config, has_fca=(i in config.fca_layers))
                for i in range(config.n_layer)
            ]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # Future prediction heads: project z → n_embd to predict future hidden states
        self.future_proj = nn.ModuleDict({
            str(i): nn.Linear(config.bottleneck_dim, config.n_embd, bias=config.bias)
            for i in config.fca_layers
        })

        # Init weights
        self.apply(self._init_weights)
        # Scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"FCAGPT parameters: {self.get_num_params() / 1e6:.2f}M")
        fca_params = sum(
            p.numel() for n, p in self.named_parameters()
            if 'fca' in n or 'future_proj' in n
        )
        print(f"  FCA branch parameters: {fca_params / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Returns:
            logits: (B, T, vocab_size)
            ce_loss: cross-entropy loss if targets provided, else None
            aux: dict with 'hidden_states', 'z_vectors', 'gate_values', 'future_preds'
                 for computing future loss externally
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))

        # Collect layer outputs for future loss computation
        hidden_states = [x]  # hidden_states[i] = output of layer i (0 = post-embedding)
        z_vectors = {}       # layer_idx → z tensor
        gate_values = {}     # layer_idx → gate tensor
        future_preds = {}    # layer_idx → projected z for future loss

        for i, block in enumerate(self.transformer.h):
            x, z, g = block(x)
            hidden_states.append(x)

            if z is not None:
                z_vectors[i] = z
                gate_values[i] = g
                future_preds[i] = self.future_proj[str(i)](z)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            ce_loss = None

        aux = {
            'hidden_states': hidden_states,
            'z_vectors': z_vectors,
            'gate_values': gate_values,
            'future_preds': future_preds,
        }
        return logits, ce_loss, aux

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Same logic as nanoGPT: decay 2D+ params, don't decay biases/LN."""
        import inspect

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
