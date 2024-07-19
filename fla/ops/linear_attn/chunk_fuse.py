# -*- coding: utf-8 -*-
# Copyright (c) 2023, Yu Zhang, Songlin Yang

from typing import Tuple

import torch
import triton
import triton.language as tl
from packaging import version
from torch.cuda.amp import custom_bwd, custom_fwd

from fla.utils import contiguous

# on-the-fly computation without materializing hidden statets into HBMs


@torch.jit.script
def normalize_output(q, k, o):
    k = k.transpose(-2, -1)
    k = k.cumsum(-1)
    k = k.transpose(-2, -1)
    z = (q * k).sum(-1, keepdim=True)
    return o / (z + 1e-5)


@triton.jit
def fused_chunk_linear_attn_fwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_V]
    v,  # value [B, H, L, D_head_V]
    o,  # output [B, H, L, D_head_V]
    initial_state,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    final_state,  # final state of the chunk [B, H, D_head_K, D_head_V]

    # Stride parameters (s_qk_h, s_qk_t, s_qk_d, etc.) 
    # are used to determine how to step through the tensors in memory, 
    # ensuring that each block accesses the correct data.
    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: L * D_head_V
    s_vo_t,  # stride size: D_head_V
    s_vo_d,  # stride size: 1
    
    B,  # batch size
    H,  # n_heads
    T,  # seq_len
    scale,  # D_head_K ** -0.5

    # tl.constexpr informs the compiler that a parameter's value is constant and known at compile-time. 
    # This allows the compiler to make optimizations that wouldn't be possible if the value were to change at runtime.
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CHECK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # indices
    # # Kernel program IDs, each dimension gets its unique ID to handle different parts of data.
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # defines three variables, i_v, i_k, and i_bh, 
    # each assigned to a specific program identifier returned by tl.program_id(axis). 
    # This function is used to retrieve the unique identifier (ID) of 
    # the currently executing instance of a kernel program on a specified dimension of the execution grid.

    # Array of sequence indices for the current block, used in operations below.
    o_i = tl.arange(0, BT)

    #################################
    ###### VERY IMPORTANT HERE ######
    #################################
    # Generate a mask for the self-attention to ensure causality in sequences.
    # [BT, BT]
    if IS_CAUSAL:
        m_s = o_i[:, None] >= o_i[None, :]
    else:
        # Conditional mask based on is_causal flag
        # `tl.where(condition, x, y): `
        # This function returns elements chosen from x or y based on the condition. 
        # If the condition at a certain index is True, it selects the corresponding element from x at that index; 
        # otherwise, it selects from y.
        m_s = tl.ones((BT, BT), dtype=bool)
    #################################

    # Initialize a block tensor for intermediate storage of results.
    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    # Create block pointers to efficiently handle sections of the input tensors based on program IDs and strides.
    # make block pointers
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh+i_k*B*H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))

    # If using an initial state, load it into the block tensor b_h.
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    # Main loop over chunks of the sequence
    # compute via each chunk, and finally concatenate them together 
    for i in range(0, tl.cdiv(T, BT)):
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_k.dtype)

        # Compute scaled dot products, apply mask, and perform matrix multiplication for attention output
        # [BT, BT]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        # [BT, BV]
        b_o = tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        if CHECK and i == 0:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v, allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v, allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        # Advance pointers for the next chunk
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))

    # Store the final state if required
    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h.to(p_final.dtype.element_ty), boundary_check=(0, 1))


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit
def fused_chunk_linear_attn_bwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    # NV: number of split in the V dimension. NK: number of split in the K dimension
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_V]
    v,  # value [B, H, L, D_head_V]
    do,  # gradient of output [B, H, L, D_head_V]
    dq,  # gradient of query [NV, B, H, L, D_head_K]
    dk,  # gradient of key [NV, B, H, L, D_head_K]
    dv,  # gradient of value [NK, B, H, L, D_head_V]

    initial_state,  # initial state of the chunk [B, H, D_head_K, D_head_V]

    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: L * D_head_V
    s_vo_t,  # stride size: D_head_V
    s_vo_d,  # stride size: 1
    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    scale,  # D_head_K ** -0.5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
    USE_INITIAL_STATE: tl.constexpr,
    CHECK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)

    if IS_CAUSAL:
        m_s = o_i[:, None] >= o_i[None, :]
    else:
        m_s = tl.ones((BT, BT), dtype = bool)
    # [BV, BK]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DV, DK), (1, DV), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v*B*H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i*BT, i_k*BK), (BT, BK), (1, 0))

        # [BT, DK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [DV, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, DV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # [BT, BT]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0)
        # [BT, DK]
        b_dq = tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)
        # [DV, DK]
        if CHECK and i == 0:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    # sync threads
    b_h = None
    tl.debug_barrier()
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    m_s = o_i[:, None] <= o_i[None, :]
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh+i_v*B*H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i*BT, i_k*BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh+i_k*B*H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i*BT, i_v*BV), (BT, BV), (1, 0))
        # [DK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, DK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, DV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # b_dd = (b_do]).to(b_do.dtype)

        # [BT, BT]
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0).to(b_q.dtype)
        # [BT, BT]
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s = tl.where(m_s, b_s, 0).to(b_q.dtype)
        # [BT, DK]
        b_dk = tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        # [BT, DV]
        b_dv = tl.dot(b_s, b_do, allow_tf32=False)
        if CHECK and i == 1:
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype),  allow_tf32=False)
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            b_dh += tl.dot(b_q, b_do, allow_tf32=False)
        else:
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype),  allow_tf32=False)
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            b_dh += tl.dot(b_q, b_do, allow_tf32=False)

        tl.store(p_dk, (b_dk * scale).to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class FusedChunkLinearAttentionFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd
    def forward(
        ctx, 
        q, 
        k, 
        v, 
        scale, 
        initial_state, 
        output_final_state,
        is_causal,
    ):
        ctx.is_causal = is_causal

        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        ctx.scale = scale
        BT = 64 # block size for triton operations, used for kernel launch configuration
        BK, BV = min(triton.next_power_of_2(d_head_qk), 64), min(triton.next_power_of_2(d_head_v), 64)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 4

        # this method creates a new tensor with the same data type and device as `q`
        # create a new tensor `o` with uninitialized data        
        o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
        if output_final_state:
            final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v, dtype=torch.float32, requires_grad=False)
        else:
            final_state = None
        # the bug still exists even for Triton 2.2 on H100 GPUs
        # so we always enable initial checks
        CHECK = True
        if version.parse(triton.__version__) < version.parse('2.2.0'):
            import warnings
            warnings.warn(
                "Triton<2.2.0 detected for running this kernel, "
                "which is known to have some weird compiler issues (refer to https://github.com/openai/triton/issues/2852) "
                "that lead to significant precision loss. "
                "We've add some initial condition checks to resolve this, sadly at the sacrifice of the speed. "
                "For optimal performance, it is recommended to install Triton>=2.2.0 (if possible)."
            )
            CHECK = True

        # the attention mechanism computation is being performed in parallel across batches and heads.
        grid = (NV, NK, batch_size * n_heads)
        # perform parallel attention computation within each block
        # output is stored to `o`
        fused_chunk_linear_attn_fwd_kernel[grid](
            q, k, v, o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            CHECK=CHECK,
            num_warps=num_warps,
            num_stages=num_stages,
            IS_CAUSAL=is_causal,
        )

        o = o.sum(0)
        ctx.save_for_backward(q, k, v, initial_state)
        ctx.CHECK = CHECK
        return o.to(q.dtype), final_state

    @staticmethod
    @custom_bwd
    @contiguous
    def backward(
        ctx, 
        do, 
        d_final_state=None,
    ):
        is_causal = ctx.is_causal
        q, k, v, initial_state = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = ctx.scale

        BT = 64
        BK, BV = min(triton.next_power_of_2(d_head_qk), 64), min(triton.next_power_of_2(d_head_v), 64)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 4

        dq = q.new_empty(NV, batch_size, n_heads,  seq_len, d_head_qk)
        dk = q.new_empty(NV, batch_size, n_heads,  seq_len, d_head_qk)
        dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
        grid = (NV, NK, batch_size * n_heads)

        fused_chunk_linear_attn_bwd_kernel[grid](
            q, k, v, do, dq, dk, dv, initial_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            CHECK=ctx.CHECK,
            num_warps=num_warps,
            num_stages=num_stages,
            IS_CAUSAL=is_causal,
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


def fused_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    normalize: bool = True,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        initial_state = initial_state.detach()
    if scale == -1:
        scale = q.shape[-1] ** -0.5

    # When you call FusedChunkLinearAttentionFunction.apply with the necessary inputs in PyTorch, 
    # you are specifically triggering the execution of the forward method defined within the FusedChunkLinearAttentionFunction class.
    # 
    # Difference between calling FusedChunkLinearAttentionFunction.forward() and FusedChunkLinearAttentionFunction.apply()
    # 
    # Autograd Integration: 
    # When you call apply() on an instance of torch.autograd.Function, you are using a class method that is specifically designed to handle both forward and backward passes correctly with PyTorch's autograd system. This method not only executes the forward() method but also sets up the necessary state and hooks to ensure that the gradients can be properly computed during the backward pass.
    # 
    # Gradient Computation: 
    # By using apply(), PyTorch records all operations performed in forward() on tensors that require gradients. This recording is essential for the autograd engine to later backtrack through these operations to compute gradients when .backward() is invoked on the loss tensor.
    # 
    # Usage: 
    # It is used when you need to ensure that your custom operation integrates seamlessly with PyTorch’s automatic differentiation.
    # 
    o, final_state = FusedChunkLinearAttentionFunction.apply(
        q, k, v, scale, initial_state, output_final_state, is_causal)
    if normalize:
        o = normalize_output(q * scale, k, o)
    return o, final_state
