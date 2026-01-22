import torch
import triton.testing
from pathlib import Path

from cs336_systems.flash_triton import FlashAttention2Triton
from cs336_systems.utils import LeaderboardBenchRow, LeaderboardBenchmarkReporter


def bench_once(fn, warmup=200, rep=1000):
    torch.cuda.synchronize()
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def main(
    out_jsonl="runs/leaderboard_ablation.jsonl", 
    out_md="runs/leaderboard_ablation.md",
    warmup=200,
    rep=1000,
    *,
    variant="baseline"
):
    out_jsonl = Path(out_jsonl)
    out_md = Path(out_md)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    B = 1
    S = 4096
    H = 16
    Dh = 64
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn(B * H, S, Dh, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B * H, S, Dh, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B * H, S, Dh, device=device, dtype=dtype, requires_grad=True)    

    def flash(q, k, v):
        return FlashAttention2Triton.apply(q, k, v, True)

    try:
        # fwd
        def fwd():
            return flash(q, k, v)

        fwd_ms = bench_once(fwd, warmup=warmup, rep=rep)

        # bwd (fixed graph)
        o = flash(q, k, v)
        loss = o.sum()

        def bwd():
            q.grad = None
            k.grad = None
            v.grad = None
            loss.backward(retain_graph=True)

        bwd_ms = bench_once(bwd, warmup=warmup, rep=rep)

        # e2e (rebuild graph)
        def e2e():
            qx = q.detach().requires_grad_(True)
            kx = k.detach().requires_grad_(True)
            vx = v.detach().requires_grad_(True)
            oy = flash(qx, kx, vx)
            oy.sum().backward()

        e2e_ms = bench_once(e2e, warmup=warmup, rep=rep)

        status = "ok"

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        fwd_ms = bwd_ms = e2e_ms = None
        status = "oom"

    except Exception as e:
        print(e)
        fwd_ms = bwd_ms = e2e_ms = None
        status = f"error:{type(e).__name__}"

    reporter = LeaderboardBenchmarkReporter(
        out_jsonl,
        out_md,
        title="#### FlashAttention-2 leaderboard ablation (batch=1, causal=True)",
    )

    row = LeaderboardBenchRow(
        variant=variant,
        dtype="bf16",
        seq_len=S,
        n_heads=H,
        d_head=Dh,
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        e2e_ms=e2e_ms,
        status=status,
    )

    msg = f"[{variant:12s}] B={B:1d}, S={S:5d}, H={H:2d}, Dh={Dh:3d} -> {row.status}"
    if row.fwd_ms is not None:
        msg += f", fwd={row.fwd_ms:.3f}ms, bwd={row.bwd_ms:.3f}ms, e2e={row.e2e_ms:.3f}ms"
    print(msg)

    reporter.append(row)
    reporter.write_markdown()

if __name__ == "__main__":
    main()