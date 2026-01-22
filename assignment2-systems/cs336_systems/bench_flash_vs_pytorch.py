import torch
import triton.testing
from typing import Tuple, Optional
import argparse

from cs336_basics.modules import scaled_dot_product_attention
from cs336_systems.flash_triton import FlashAttention2Triton

from cs336_systems.utils import FlashBenchRow, FlashBenchmarkReporter


def make_causal_mask(n: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(n, device=device)
    return (idx[:, None] >= idx[None, :])


def pow2_list(lo: int, hi: int) -> list[int]:
    out = []
    x = lo
    while x <= hi:
        out.append(x)
        x *= 2
    return out


@torch.no_grad()
def make_inputs(
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn((1, seq_len, d_model), device=device, dtype=dtype)
    k = torch.randn((1, seq_len, d_model), device=device, dtype=dtype)
    v = torch.randn((1, seq_len, d_model), device=device, dtype=dtype)
    return q, k, v


def bench_one(
    impl: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    warmup: int,
    rep: int
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Returns: (fwd_ms, bwd_ms, e2e_ms, status)
    """
    assert q.shape[0] == 1, "batch size must be 1"

    device = q.device
    mask = make_causal_mask(q.shape[1], device) if is_causal else None

    def fwd_fn():
        if impl == "baseline":
            return scaled_dot_product_attention(q, k, v, mask=mask)
        elif impl == "flash":
            return FlashAttention2Triton.apply(q, k, v, is_causal)
        else:
            raise ValueError(f"unknow impl: {impl}")
    
    try:
        # foward benchmark
        fwd_ms = float(triton.testing.do_bench(fwd_fn, warmup=warmup, rep=rep))

        # build one graph for backward-only
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)

        if impl == "baseline":
            out = scaled_dot_product_attention(q_, k_, v_, mask=mask)
        else:
            out = FlashAttention2Triton.apply(q_, k_, v_, is_causal)

        do = torch.randn_like(out)

        def bwd_only():
            torch.autograd.grad(out, (q_, k_, v_), grad_outputs=do, retain_graph=True)
        
        bwd_ms = float(triton.testing.do_bench(bwd_only, warmup=warmup, rep=rep))

        # end-to-end benchmark (rebuild graph each rep)
        def e2e():
            qx = q.detach().requires_grad_(True)
            kx = k.detach().requires_grad_(True)
            vx = v.detach().requires_grad_(True)

            if impl == "baseline":
                oy = scaled_dot_product_attention(qx, kx, vx, mask=mask)
            else:
                oy = FlashAttention2Triton.apply(qx, kx, vx, is_causal)
            
            doy = torch.rand_like(oy)
            torch.autograd.grad(oy, (qx, kx, vx), grad_outputs=doy, retain_graph=False)

        e2e_ms = float(triton.testing.do_bench(e2e, warmup=warmup, rep=rep))

        return fwd_ms, bwd_ms, e2e_ms, "ok"
    
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None, "oom"

    except Exception as e:
        print(e)
        return None, None, None, f"error:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", type=str, default="runs/flash_bench.jsonl")
    ap.add_argument("--out-md", type=str, default="runs/flash_bench.md")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--rep", type=int, default=100)

    ap.add_argument("--seq-min", type=int, default=128)
    ap.add_argument("--seq-max", type=int, default=65536)
    ap.add_argument("--d-min", type=int, default=16)
    ap.add_argument("--d-max", type=int, default=128)

    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    device = torch.device(args.device)

    reporter = FlashBenchmarkReporter(
        args.out_jsonl,
        args.out_md,
        title="#### FlashAttention-2 (Triton) vs Baseline (PyTorch)  (batch=1, causal=True)"
    )

    seq_list = pow2_list(args.seq_min, args.seq_max)
    d_list = pow2_list(args.d_min, args.d_max)
    dtypes = [(torch.bfloat16, "bf16"), (torch.float32, "fp32")]

    for n in seq_list:
        for d in d_list:
            for dtype, dtype_name in dtypes:
                q, k, v = make_inputs(n, d, dtype=dtype, device=device)

                for impl in ["baseline", "flash"]:
                    fwd_ms, bwd_ms, e2e_ms, status = bench_one(
                        impl=impl,
                        q=q,
                        k=k,
                        v=v,
                        is_causal=True,
                        warmup=args.warmup,
                        rep=args.rep,
                    )

                    row = FlashBenchRow(
                        impl=impl,
                        dtype=dtype_name,
                        seq_len=n,
                        d_model=d,
                        fwd_ms=fwd_ms,
                        bwd_ms=bwd_ms,
                        e2e_ms=e2e_ms,
                        status=status,
                    )

                    tag = f"{impl}|{dtype_name}"
                    msg = f"[{tag:13s}] d={d:4d}, s={n:6d} -> {row.status}"

                    if row.fwd_ms is not None:
                        msg += (
                            f", fwd={row.fwd_ms:.3f}ms"
                            f", bwd={row.bwd_ms:.3f}ms"
                            f", e2e={row.e2e_ms:.3f}ms"
                        )
                    
                    print(msg)

                    reporter.append(row)

    reporter.write_markdown()
    print(reporter.render_markdown())
    print(f"\nSaved: {args.out_jsonl}  and  {args.out_md}")


if __name__ == "__main__":
    main()    