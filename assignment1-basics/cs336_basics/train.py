import os
import time
import math
import torch
import numpy as np
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy_from_logits, load_checkpoint, save_checkpoint, clip_grad_norm
from cs336_basics.config import get_default_config
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.optimizer import AdamW, lr_cosine_schedule_with_warmup

def open_memmap_1d(path: str, np_dtype: str) -> np.memmap:
    """
    Open a 1D token memmap file. The file is assumed to be a raw binary array.
    """
    dtype = np.dtype(np_dtype)
    itemsize = dtype.itemsize
    nbytes = os.path.getsize(path)
    if nbytes % itemsize != 0:
        raise ValueError(f"File size is not divisible by dtype size: {path} ({nbytes} bytes, itemsize={itemsize})")
    length = nbytes // itemsize
    return np.memmap(path, mode="r", dtype=dtype, shape=(length,))

def torch_dtype_from_string(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("float32", "fp32"):
        return torch.float32
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype string: {s}")

def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr

@torch.no_grad()
def estimate_loss(model: torch.nn.Module, data: np.memmap, cfg) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.train.eval_batches):
        xb, yb = get_batch(
            dataset=data,
            batch_size=cfg.train.batch_size,
            context_length=cfg.data.context_length,
            device=cfg.data.device
        )
        logits = model(xb)  # (B, S, V)
        B, S, V = logits.shape
        loss = cross_entropy_from_logits(logits.reshape(B * S, V), yb.reshape(B * S))
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))

def main() -> None:
    # 1. Load configuration and set random seed
    cfg = get_default_config()

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    # 2. Optional experiment tracking (weights & biases)
    wandb = None
    if cfg.wandb.enable:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, config={
            "data": cfg.data.__dict__,
            "model": cfg.model.__dict__,
            "optim": cfg.optim.__dict__,
            "train": cfg.train.__dict__,
            "wandb": cfg.wandb.__dict__
        })

    # 3. Prepare filesystem and load datasets (memory-mapped)
    os.makedirs(os.path.dirname(cfg.train.ckpt_path) or ".", exist_ok=True)

    train_mm = open_memmap_1d(cfg.data.train_data_path, cfg.data.np_dtype)
    val_mm = open_memmap_1d(cfg.data.val_data_path, cfg.data.np_dtype)

    # 4. Create model and move it to the target device
    device = torch.device(cfg.data.device)
    model_dtype = torch_dtype_from_string(cfg.model.torch_dtype)

    d_ff = cfg.model.d_ff if cfg.model.d_ff is not None else 4 * cfg.model.d_model 

    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=d_ff,
        rope_theta=cfg.model.rope_theta,
        max_seq_len=cfg.model.max_seq_len,
        eps=cfg.model.rmsnorm_eps,
        device=device,
        dtype=model_dtype
    ).to(device)

    # 5. Create optimizer and (optionally) resume from checkpoint
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr_max,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay
    )

    start_it = 0
    if cfg.train.resume_from is not None and os.path.exists(cfg.train.resume_from):
        start_it = load_checkpoint(cfg.train.resume_from, model, optimizer)

    # 6. Training loop initialization
    best_val = float("inf")
    last_log_t = time.time()

    # 7. Main training loop
    for it in range(start_it, cfg.train.max_steps):
        # 7.1 Update learning rate according to schedule
        lr = lr_cosine_schedule_with_warmup(
            t=it, 
            alpha_max=cfg.optim.lr_max,
            alpha_min=cfg.optim.lr_min,
            T_w=cfg.optim.warmup_iters,
            T_c=cfg.optim.cosine_cycle_iters
        )
        set_optimizer_lr(optimizer, lr)

        # 7.2 Sample a batch of training data
        xb, yb = get_batch(
            train_mm,
            batch_size=cfg.train.batch_size,
            context_length=cfg.data.context_length,
            device=cfg.data.device
        )

        # 7.3 Forward pass and loss computation
        logits = model(xb)  # (B, S, V)
        B, S, V = logits.shape
        loss = cross_entropy_from_logits(logits.reshape(B * S, V), yb.reshape(B * S))

        # 7.4 Backward pass (gradient computation)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 7.5 Gradient clipping for training stability
        if cfg.optim.grad_clip > 0:
            clip_grad_norm(model.parameters(), cfg.optim.grad_clip, eps=1e-6)
        
        # 7.6 Optimizer step (parameter update)
        optimizer.step()

        # 7.7 Periodic training metrics logging
        if (it + 1) % cfg.train.log_interval == 0:
            now = time.time()
            dt = max(now - last_log_t, 1e-9)
            tok_s = (cfg.train.batch_size * cfg.data.context_length * cfg.train.log_interval) / dt
            msg = f"it={it+1} loss={loss.item():.4f} lr={lr:.3e} tok/s={tok_s:.1f}"
            print(msg)
            if wandb is not None:
                wandb.log({"train/loss": float(loss.item()), "train/lr": lr, "train/tok_s": tok_s}, step=it + 1)
            last_log_t = now
        
        # 7.8 Periodic evaluation on validation set
        if (it + 1) % cfg.train.eval_interval == 0:
            val_loss = estimate_loss(model, val_mm, cfg)
            val_ppl = float(math.exp(val_loss))
            print(f"[eval] it={it+1} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
            if wandb is not None:
                wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=it + 1)            

            # Save the best-performing checkpoint
            if val_loss < best_val:
                best_val = val_loss
                best_path = cfg.train.ckpt_path.replace(".pt", ".best.pt")
                save_checkpoint(model, optimizer, it + 1, best_path)

        # 7.9 Periodic checkpointing
        if (it + 1) % cfg.train.ckpt_interval == 0:
            save_checkpoint(model, optimizer, it + 1, cfg.train.ckpt_path)

    # 8. Final checkpoint adn cleanup 
    save_checkpoint(model, optimizer, cfg.train.max_steps, cfg.train.ckpt_path)
    if wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main()