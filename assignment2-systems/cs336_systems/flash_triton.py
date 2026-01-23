import math
import torch
import triton
import triton.language as tl

from cs336_systems.flash_pytorch import flash_bwd_recompute_impl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES: tl.constexpr,
    N_KEYS: tl.constexpr,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # program ids
    pid_q = tl.program_id(0)  # query tile id
    pid_b = tl.program_id(1)  # batch id

    # block pointers for Q/K/V/O/L tiles
    # block pointers encapsulate base, shape, strides, and OOB handling;
    # advancing them avoids re-materializing raw pointer arithmetic in the loop
    Qb = Q_ptr + pid_b * stride_qb
    Kb = K_ptr + pid_b * stride_kb
    Vb = V_ptr + pid_b * stride_vb
    Ob = O_ptr + pid_b * stride_ob
    Lb = L_ptr + pid_b * stride_lb

    Q_bp = tl.make_block_ptr(
        base=Qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_bp = tl.make_block_ptr(
        base=Kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_bp = tl.make_block_ptr(
        base=Vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_bp = tl.make_block_ptr(
        base=Ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(pid_q * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_bp = tl.make_block_ptr(
        base=Lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(pid_q * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # load the Q tile
    # keep the original dtype for the final store cast
    q_raw = tl.load(Q_bp, boundary_check=(0, 1), padding_option="zero")
    q = q_raw.to(tl.float32)

    # running state (on-chip)
    m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)  # [Bq]
    l = tl.zeros((Q_TILE_SIZE,), tl.float32)                # [Bq]
    acc = tl.zeros((Q_TILE_SIZE, D), tl.float32)            # [Bq, D]

    # iterate over K/V tile by advancing block pointers (instead of re-building raw pointers)
    K_it = K_bp
    V_it = V_bp

    # absolute query indices used only for causal masking
    q_abs = pid_q * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for kb in range(0, N_KEYS, K_TILE_SIZE):
        # load one (K_TILE_SIZE, D) tile of K and V
        k = tl.load(K_it, boundary_check=(0, 1), padding_option="zero").to(tl.float32)  # [Bk, D]
        v = tl.load(V_it, boundary_check=(0, 1), padding_option="zero")                 # [Bk, D]

        # S = q @ k^T * scale -> [Bq, Bk]
        S = tl.dot(q, tl.trans(k)) * scale  # float32

        # causal mask: keep if q_idx >= k_idx else -1e-6
        if IS_CAUSAL:
            k_abs = kb + tl.arange(0, K_TILE_SIZE)
            S = tl.where(q_abs[:, None] >= k_abs[None, :], S, -1.0e6)

        # online softmax update
        m_new = tl.maximum(m, tl.max(S, axis=1))  # [Bq]
        p = tl.exp(S - m_new[:, None])            # [Bq, Bk]

        alpha = tl.exp(m - m_new)                 # [Bq]
        l_new = alpha * l + tl.sum(p, axis=1)     # [Bq]

        # acc = alpha * acc + p @ v
        # p needs to match v dtype before dot
        p = p.to(v.dtype)
        acc = alpha[:, None] * acc
        acc = tl.dot(p, v, acc=acc)

        m = m_new
        l = l_new

        # advance K/V block pointers to the next tile along the sequence dimension
        K_it = K_it.advance((K_TILE_SIZE, 0))
        V_it = V_it.advance((K_TILE_SIZE, 0))

    # write O and L
    # store output in the original input dtype
    o = (acc / l[:, None]).to(q_raw.dtype)

    tl.store(O_bp, o, boundary_check=(0, 1))

    L_out = m + tl.log(l)  # [Bq]
    tl.store(L_bp, L_out, boundary_check=(0,))


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        # Expect (B, Q, D), (B, K, D), (B, K, D)
        if not q.is_cuda:
            raise RuntimeError("Triton implementation requires CUDA tensors")
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError("Expected q/k/v to be 3D: (B, N, D)")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("Batch size mismatch")
        if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
            raise ValueError("k/v shape mismatch")
        if q.shape[2] != k.shape[2]:
            raise ValueError("q/k D mismatch")

        B, Q, D = q.shape
        K = k.shape[1]
        scale = 1.0 / math.sqrt(D)

        # tile sizes
        Bq = 32
        Bk = 32

        # outputs
        o = torch.empty((B, Q, D), device=q.device, dtype=q.dtype)
        L = torch.empty((B, Q), device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(Q, Bq), B)

        flash_fwd_kernel[grid](
            q, k, v,
            o, L,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=Q,
            N_KEYS=K,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            IS_CAUSAL=is_causal,
            num_warps=4
        )

        # save for backward
        ctx.save_for_backward(L, q, k, v, o)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        (L, q, k, v, o) = ctx.saved_tensors
        dq, dk, dv = flash_bwd_recompute_impl(q, k, v, o, do, L, ctx.is_causal)
        return dq, dk, dv, None