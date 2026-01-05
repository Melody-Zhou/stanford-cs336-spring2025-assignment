import os
import math
import torch
from typing import Iterable, BinaryIO, IO, Union

def cross_entropy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Numerically-stable cross entropy from logits.

    Args:
        logits: Tensor of shape [..., vocab_size], where the last dimension is the class dimension.
        targets: Long tensor of shape [...], containing class indices in [0, vocab_size).

    Returns:
        A scalar tensor: mean negative log-likelihood over all batch elements.
    """
    if logits.ndim < 1:
        raise ValueError("logits must have at least 1 dimension [..., vocab_size].")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(f"targets shape {targets.shape} must match logits batch shape {logits.shape[:-1]}.")
    if targets.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        raise TypeError("targets must be an integer tensor of class indices.")

    # Subtract max for numerical stability
    m = logits.max(dim=-1, keepdim=True).values
    shifted = logits - m

    # Compute logsumexp over the class dimension
    lse = torch.logsumexp(shifted, dim=-1)  # shape [...]

    # Gather the logit corresponding to the true class
    idx = targets.unsqueeze(-1)  # [..., 1]
    correct = shifted.gather(dim=-1, index=idx).squeeze(-1)  # shape [...]

    # Negative log-likelihood: logsumexp - correct_logit
    nll = lse - correct

    # Return mean over all batch elements
    return nll.mean()

def clip_grad_norm(
    params: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6
) -> float:
    """
    Clip gradients in-place so that the global L2 norm does not exceed max_norm.

    Args:
        params: Iterable of parameters whose .grad will be modified in-place.
        max_norm: Maximum allowed global L2 norm.
        eps: Small constant for numerical stability.

    Returns:
        The total (pre-clipping) global L2 norm as a Python float.
    """
    if max_norm < 0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}")
    
    # Collect gradients that exist
    grads = []
    for p in params:
        if p is None:
            continue
        g = p.grad
        if g is None:
            continue
        if g.is_sparse:
            raise RuntimeError("clip_grad_norm_ does not support sparse gradients.")
        grads.append(g)
    
    if len(grads) == 0:
        return 0.0
    
    # Compute global L2 norm: sqrt(sum_i ||g_i||_2^2)
    # Use float32 accumulation for stability and consistency
    total_sq = 0.0
    for g in grads:
        total_sq += float(g.detach().float().pow(2).sum().item())
    total_norm = math.sqrt(total_sq)

    # Compute clipping coefficient
    clip_coef = float(max_norm) / (total_norm + float(eps))

    # If norm exceeds threshold, scale all grads by the same factor (in-place)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)

    return float(total_norm)

PathOrFile = Union[str, os.PathLike, BinaryIO, IO[bytes]]

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: PathOrFile
) -> None:
    """
    Save a training checkpoint containing model/optimizer state and iteration.

    Args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        iteration: Current training iteration (step).
        out: File path or a binary file-like object.
    """
    obj = {
        "iteration": int(iteration),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(obj, out)    

def load_checkpoint(
    src: PathOrFile,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    Load a training checkpoint and restore model/optimizer state.

    Args:
        src: File path or a binary file-like object.
        model: torch.nn.Module to restore into.
        optimizer: torch.optim.Optimizer to restore into.

    Returns:
        The iteration (step) stored in the checkpoint.
    """
    ckpt = torch.load(src, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must be a dict.")
    
    if "model_state_dict" not in ckpt or "optimizer_state_dict" not in ckpt or "iteration" not in ckpt:
        raise KeyError("Checkpoint dict missing required keys.")

    model.load_state_dict(ckpt["model_state_dict"])    
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt["iteration"])