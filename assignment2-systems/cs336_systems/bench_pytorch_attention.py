import torch
import argparse
from typing import Callable, Dict, Any
from pathlib import Path

from cs336_basics.modules import scaled_dot_product_attention
from cs336_systems.utils import AttentionRow, AttentionBenchmarkReporter


def cuda_sync():
    torch.cuda.synchronize()


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # True = keep, False = masked out
    return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))


def time_forward(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor, k: torch. Tensor, v: torch.Tensor,
    iters: int
) -> float:
    # Use CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    cuda_sync()
    start.record()
    for _ in range(iters):
        cuda_sync()
        _ = fn(q, k, v)
        cuda_sync()
    end.record()
    cuda_sync()
    return start.elapsed_time(end) / iters


def time_backward(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor, k: torch. Tensor, v: torch.Tensor,
    iters: int
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0
    for _ in range(iters):
        cuda_sync()
        out = fn(q, k, v)
        loss = out.sum()
        cuda_sync()

        start.record()
        loss.backward()
        end.record()
        cuda_sync()
        total_ms += start.elapsed_time(end)

        # clear grads for next iter
        q.grad = None
        k.grad = None
        v.grad = None

    return total_ms / iters


def _cleanup_after_oom(local_vars: Dict[str, Any]) -> None:
    for name in ["q", "k", "v", "out", "loss", "mask", "fn"]:
        if name in local_vars:
            try:
                del local_vars[name]
            except Exception:
                pass
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def run_one(
    d_model: int, seq_len: int, batch: int, dtype: torch.dtype,
    warmup: int, iters: int, use_causal_mask: bool,
    do_compile: bool
) -> AttentionRow:
    device = torch.device("cuda")
    local_vars: Dict[str, Any] = {}
    impl = "compiled" if do_compile else "eager"

    try:
        q = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        local_vars.update({"q": q, "k": k, "v": v})

        mask = causal_mask(seq_len, device) if use_causal_mask else None
        local_vars["mask"] = mask

        def attn(q, k, v):
            return scaled_dot_product_attention(q, k, v, mask=mask)
            # return naive_attention(q, k, v, mask=mask)

        fn = attn
        if do_compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is not available in this PyTorch build.")
            fn = torch.compile(attn)
        local_vars["fn"] = fn

        # warmup
        for _ in range(warmup):
            cuda_sync()
            out = fn(q, k, v)
            cuda_sync()
            out.sum().backward()
            cuda_sync()
            q.grad = None
            k.grad = None
            v.grad = None
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cuda_sync()

        fwd_ms = time_forward(fn, q, k, v, iters)

        # memory snapshot
        mem_before_bwd_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        bwd_ms = time_backward(fn, q, k, v, iters)

        return AttentionRow(d_model, seq_len, fwd_ms, bwd_ms, mem_before_bwd_mb, "ok", impl=impl)

    except RuntimeError as e:
        msg =  str(e).lower()
        if "out of memory" in msg:
            # Release references, then clear cache
            _cleanup_after_oom(local_vars)
            return AttentionRow(d_model, seq_len, None, None, None, "oom", impl=impl)
        return AttentionRow(d_model, seq_len, None, None, None, f"error:{type(e).__name__}", impl=impl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="float32", choices=[ "float32", "float16", "bfloat16"])
    parser.add_argument("--no-causal", action="store_true", help="Disable causal mask (default: causal enabled)")
    parser.add_argument("--out-dir", type=str, default="runs/pytorch_attention")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for attention")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    out_dir = Path(args.out_dir)
    reporter = AttentionBenchmarkReporter(
        jsonl_path=out_dir / "metrics.jsonl",
        md_path=out_dir / "table.md",
        title="#### Attention benchmark (impl=eager or impl=compiled)",
    )

    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    for d in d_models:
        for s in seq_lens:
            torch.cuda.empty_cache()
            r = run_one(
                d_model=d,
                seq_len=s,
                batch=args.batch,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                use_causal_mask=(not args.no_causal),
                do_compile=args.compile
            )
            tag = "compiled" if args.compile else "eager"
            print(f"[{tag}] d={d:3d}, s={s:5d} -> {r.status}"
                  + ("" if r.fwd_ms is None else f", fwd={r.fwd_ms:.3f}ms, bwd={r.bwd_ms:.3f}ms, mem={r.mem_before_bwd_mb:.1f}MB"))
            reporter.append(r)

    reporter.write_markdown()


if __name__ == "__main__":
    main()
