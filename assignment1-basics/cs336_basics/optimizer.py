import math
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr : float = 1e-3,
        betas : tuple[float, float] = (0.9, 0.999),
        eps : float = 1e-8,
        weight_decay: float = 0.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not (0.0 <= betas[0] < 1.0) or not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)        
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")
                
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Apply decoupled weight decay
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Compute bias-corrected step size
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
    
def lr_cosine_schedule_with_warmup(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int
) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Args:
        t: Current iteration (step).
        alpha_max: Maximum learning rate.
        alpha_min: Minimum / final learning rate.
        T_w: Number of warmup iterations.
        T_c: Number of cosine annealing iterations (end of cosine cycle).

    Returns:
        Learning rate at step t.
    """
    # Warmup phase: alpha_t = (t / T_w) * alpha_max
    if T_w > 0 and t < T_w:
        return (t / T_w) * float(alpha_max)
    
    # After cosine phase: keep alpha_min
    if t > T_c:
        return float(alpha_min)

    # Cosine phase: T_w <= t <= T_c
    denom = T_c - T_w
    if denom <= 0:
        # Degenerate schedule: no valid cosine interval
        return float(alpha_min)

    frac = (t - T_w) / denom  # in [0, 1]
    return float(alpha_min) + 0.5 * (1.0 + math.cos(math.pi * frac)) * (float(alpha_max) - float(alpha_min))