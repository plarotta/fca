"""
FCA training script. Forks nanoGPT's train.py with:
  - FCAGPT model with future-conditioned attention branches
  - Future hidden state prediction loss with EMA targets
  - Lambda schedule for loss ramp-up
  - Logging of gate values and z statistics

Usage:
  python -m fca.train --config configs/fca_top_third.yaml
  python -m fca.train --fca_layers 8 9 10 11 --bottleneck_dim 192

Can also run on multiple GPUs with torchrun:
  torchrun --standalone --nproc_per_node=4 -m fca.train
"""

import os
import sys
import time
import math
import pickle
import argparse
from contextlib import nullcontext


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', '1'):
        return True
    if v.lower() in ('no', 'false', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Ensure nanoGPT is importable (for data loading utils, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nanoGPT'))

from fca.config import FCAConfig
from fca.model import FCAGPT
from fca.losses import EMATargetTracker, compute_total_loss


def parse_args():
    parser = argparse.ArgumentParser(description="FCA Training")
    # Config file (overrides defaults, CLI overrides config file)
    parser.add_argument('--config', type=str, default=None)

    # I/O
    parser.add_argument('--out_dir', type=str, default='results/fca-top-third')
    parser.add_argument('--eval_interval', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--always_save_checkpoint', type=str2bool, default=True)
    parser.add_argument('--init_from', type=str, default='scratch')
    parser.add_argument('--checkpoint_interval', type=int, default=25000,
                        help='Save numbered checkpoints at this interval')

    # Wandb
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='fca')
    parser.add_argument('--wandb_run_name', type=str, default='fca-top-third')

    # Data
    parser.add_argument('--dataset', type=str, default='openwebtext')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--block_size', type=int, default=1024)

    # Base model
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bias', type=str2bool, default=False)

    # FCA architecture
    parser.add_argument('--fca_layers', type=int, nargs='+', default=[8, 9, 10, 11])
    parser.add_argument('--bottleneck_dim', type=int, default=192)
    parser.add_argument('--fca_n_head', type=int, default=12)
    parser.add_argument('--fca_dropout', type=float, default=0.0)
    parser.add_argument('--random_z', action='store_true')

    # Future loss
    parser.add_argument('--future_loss_weight', type=float, default=1.0)
    parser.add_argument('--lambda_warmup_steps', type=int, default=20000)
    parser.add_argument('--no_lambda_schedule', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--no_ema_target', action='store_true')
    parser.add_argument('--future_offset', type=int, default=1)

    # Optimizer
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--max_iters', type=int, default=100000)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=2000)
    parser.add_argument('--lr_decay_iters', type=int, default=100000)
    parser.add_argument('--min_lr', type=float, default=6e-5)

    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--compile', type=str2bool, default=True)

    return parser.parse_args()


def build_fca_config(args) -> FCAConfig:
    return FCAConfig(
        block_size=args.block_size,
        vocab_size=50304,  # will be overridden if meta.pkl exists
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        fca_layers=args.fca_layers,
        bottleneck_dim=args.bottleneck_dim,
        fca_n_head=args.fca_n_head,
        fca_dropout=args.fca_dropout,
        random_z=args.random_z,
        future_loss_weight=args.future_loss_weight,
        lambda_warmup_steps=args.lambda_warmup_steps,
        use_lambda_schedule=not args.no_lambda_schedule,
        ema_decay=args.ema_decay,
        use_ema_target=not args.no_ema_target,
        future_offset=args.future_offset,
    )


def main():
    args = parse_args()

    # --- DDP setup ---
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert args.gradient_accumulation_steps % ddp_world_size == 0
        args.gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = args.device

    tokens_per_iter = (
        args.gradient_accumulation_steps * ddp_world_size * args.batch_size * args.block_size
    )
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    # Fall back to float32 on CPU
    dtype = args.dtype if device_type == 'cuda' else 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # --- Data ---
    data_dir = os.path.join('nanoGPT', 'data', args.dataset)

    def get_batch(split):
        fname = 'train.bin' if split == 'train' else 'val.bin'
        data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + args.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + args.block_size].astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # --- Model ---
    fca_config = build_fca_config(args)

    # Check for dataset-specific vocab size
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        fca_config.vocab_size = meta['vocab_size']
        if master_process:
            print(f"found vocab_size = {fca_config.vocab_size} (inside {meta_path})")

    iter_num = 0
    best_val_loss = 1e9

    if args.init_from == 'scratch':
        if master_process:
            print("Initializing a new FCAGPT model from scratch")
        model = FCAGPT(fca_config)
    elif args.init_from == 'resume':
        if master_process:
            print(f"Resuming training from {args.out_dir}")
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        fca_config = checkpoint['fca_config']
        model = FCAGPT(fca_config)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    model.to(device)

    # EMA tracker
    ema_tracker = EMATargetTracker(fca_config, device)

    # GradScaler for fp16 (only on CUDA)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16' and device_type == 'cuda'))

    # Optimizer
    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
    )
    if args.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None

    # Compile
    if args.compile:
        if master_process:
            print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    # DDP wrap
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    # --- Eval ---
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, ce_loss, aux = model(X, Y)
                losses[k] = ce_loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # --- LR schedule ---
    def get_lr(it):
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
        if it > args.lr_decay_iters:
            return args.min_lr
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    # --- Wandb ---
    if args.wandb_log and master_process:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # --- Training loop ---
    X, Y = get_batch('train')
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    if master_process:
        print(f"Starting training from iter {iter_num}")

    while True:
        # LR schedule
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Eval + checkpoint
        if iter_num % args.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            }
            if args.wandb_log:
                import wandb
                wandb.log(log_dict)

            if losses['val'] < best_val_loss or args.always_save_checkpoint:
                best_val_loss = min(best_val_loss, losses['val'])
                if iter_num > 0:
                    ckpt = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'fca_config': fca_config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'args': vars(args),
                    }
                    print(f"saving checkpoint to {args.out_dir}")
                    torch.save(ckpt, os.path.join(args.out_dir, 'ckpt.pt'))

        # Numbered checkpoints at fixed intervals
        if iter_num > 0 and iter_num % args.checkpoint_interval == 0 and master_process:
            ckpt = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'fca_config': fca_config,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'args': vars(args),
            }
            ckpt_path = os.path.join(args.out_dir, f'ckpt_{iter_num}.pt')
            print(f"saving numbered checkpoint to {ckpt_path}")
            torch.save(ckpt, ckpt_path)

        if iter_num == 0 and args.eval_only:
            break

        # Forward / backward with gradient accumulation
        for micro_step in range(args.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == args.gradient_accumulation_steps - 1
                )
            with ctx:
                logits, ce_loss, aux = model(X, Y)

                # Update EMA targets
                ema_tracker.update(aux['hidden_states'])

                # Compute combined loss
                total_loss, ce_loss_val, future_loss_val, current_lambda = compute_total_loss(
                    ce_loss,
                    aux['future_preds'],
                    aux['hidden_states'],
                    ema_tracker,
                    fca_config,
                    iter_num,
                )
                total_loss = total_loss / args.gradient_accumulation_steps

            X, Y = get_batch('train')
            scaler.scale(total_loss).backward()

        # Gradient clipping
        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0 and master_process:
            total_lossf = total_loss.item() * args.gradient_accumulation_steps
            ce_lossf = ce_loss_val.item()
            future_lossf = future_loss_val.item()

            # Gate statistics
            gate_stats = {}
            for layer_idx, g in aux['gate_values'].items():
                gate_stats[f'gate/{layer_idx}/mean'] = g.mean().item()
                gate_stats[f'gate/{layer_idx}/std'] = g.std().item()

            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(
                    args.batch_size * args.gradient_accumulation_steps, dt
                ) if hasattr(raw_model, 'estimate_mfu') else -1.0
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

            print(
                f"iter {iter_num}: total_loss {total_lossf:.4f}, "
                f"ce {ce_lossf:.4f}, future {future_lossf:.4f}, "
                f"lambda {current_lambda:.3f}, "
                f"time {dt * 1000:.2f}ms"
            )

            if args.wandb_log:
                import wandb
                log_dict = {
                    "iter": iter_num,
                    "train/total_loss": total_lossf,
                    "train/ce_loss": ce_lossf,
                    "train/future_loss": future_lossf,
                    "train/lambda": current_lambda,
                    "lr": lr,
                    **gate_stats,
                }
                wandb.log(log_dict)

        iter_num += 1
        local_iter_num += 1

        if iter_num > args.max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    main()
