import torch
from typing import Optional

def top_p_sampling(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) sampling to a probability vector.

    Args:
        probs: 1D tensor of probabilities (sum to 1).
        top_p: Cumulative probability threshold in (0, 1].
    
    Returns:
        Filtered probabilities (renormalized), same shape as probs.
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")    

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens while cumulative prob is <= top_p, but always keep at least one token.
    keep = cum <= top_p
    keep[..., 0] = True

    filtered_sorted_probs = sorted_probs * keep.to(sorted_probs.dtype)
    filtered_sorted_probs = filtered_sorted_probs / filtered_sorted_probs.sum(dim=-1, keepdim=True)

    out = torch.zeros_like(probs)
    out.scatter_(dim=-1, index=sorted_idx, src=filtered_sorted_probs)
    return out

@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    *,
    end_token_id: int,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0
) -> torch.Tensor:
    
    if prompt_ids.dim() != 1:
        raise ValueError(f"prompt_ids must be 1D (t,), got shape {tuple(prompt_ids.shape)}")
    if prompt_ids.dtype != torch.long:
        prompt_ids = prompt_ids.to(torch.long)

    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    
    model_was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    out = prompt_ids.to(device)

    # Try to read context_length if the model exposes it
    context_length: Optional[int] = getattr(model, "context_length", None)

    for _ in range(max_new_tokens):
        # Truncate to the model context window if needed
        if context_length is not None and out.numel() > context_length:
            inp = out[-context_length:]
        else:
            inp = out

        logits = model(inp.unsqueeze(0))  # (1, S, V)
        next_logits = logits[0, -1, :]    # (V,)

        # Greedy decoding if temperature == 0
        if temperature == 0.0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            if temperature < 0.0:
                raise ValueError(f"temperature must be >= 0, got {temperature}")
            
            scaled = next_logits / float(temperature)
            probs = torch.softmax(scaled, dim=-1)

            if top_p < 1.0:
                probs = top_p_sampling(probs, top_p)
            
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        
        out = torch.cat([out, torch.tensor([next_id], device=device, dtype=torch.long)], dim=0)

        if next_id == int(end_token_id):
            break

    if model_was_training:
        model.train()

    return out

if __name__ == "__main__":
    from cs336_basics.nn_utils import load_checkpoint
    from cs336_basics.config import get_default_config
    from cs336_basics.transformer_lm import TransformerLM
    from cs336_basics.tokenizer import Tokenizer
    from cs336_basics.optimizer import AdamW

    EOT = "<|endoftext|>"

    cfg = get_default_config()
    device = torch.device(cfg.data.device)

    # ---- 1) Load tokenzier (TinyStories BPE) ----
    tok = Tokenizer.from_files(
        "workspace/tinystories_bpe_vocab.pkl",
        "workspace/tinystories_bpe_merges.pkl",
        special_tokens=[EOT]
    )
    end_token_id = tok.special_id[EOT]

    # ---- 2) Build model (match training config) ----
    model_dtype = cfg.model.torch_dtype.lower()
    dtype_map = {"float32": torch.float32, "fp32": torch.float32,
                 "float16": torch.float16, "fp16": torch.float16,
                 "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    dtype = dtype_map[model_dtype]

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
        dtype=dtype,
    ).to(device)    

    # ---- 3) Create a dummy optimizer ----
    optimizer = AdamW(model.parameters())

    # --- 4) Load checkpoint weights ---
    ckpt_path = "workspace/ckpt.best.pt"
    it = load_checkpoint(ckpt_path, model, optimizer)

    # 5) Encode prompt -> generate -> decode
    prompt = "Once upon a time"
    prompt_ids = torch.tensor(tok.encode(prompt), dtype=torch.long)

    out_ids = generate(
        model,
        prompt_ids,
        end_token_id=end_token_id,
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.9,
    )

    print(tok.decode(out_ids.tolist()))