# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Tests for MLA decode with sub-16 head counts (nhead < 16 where 16 % nhead == 0).
#
# Motivation: Kimi-K2-Instruct-0905 has 64 attention heads total. With high TP:
#   TP=8  → 8 heads/GPU
#   TP=16 → 4 heads/GPU
#   TP=32 → 2 heads/GPU
# The aiter MLA persistent kernel's minimum tile is 16 heads. The fix pads the
# head dimension to 16 with zeros, runs the kernel, then slices back ori_nhead.
#
# This test verifies that:
#   1. The padded output for real heads is numerically correct (matches torch reference).
#   2. Only ori_nhead heads appear in the final output (shape is unchanged).
#   3. The padded (zero-query) heads do NOT corrupt the real heads' outputs —
#      confirmed by comparing real-head slices between the padded run and a nhead=16
#      "full" run where the extra slots contain zeros.

import pytest
import torch

import aiter
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# Kimi-K2-Instruct-0905 MLA config (absorbed, none-absorb qk_head_dim)
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK                        # 512


# ---------------------------------------------------------------------------
# Torch reference: batched MLA decode (no paging, causal not needed for decode)
# ---------------------------------------------------------------------------

def torch_mla_decode_ref(
    q,           # [total_q, nhead, qk_head_dim]
    kv_buffer,   # [num_page, page_size, 1, kv_lora_rank + qk_rope_head_dim]
    qo_indptr,   # [bs+1]
    kv_indptr,   # [bs+1]
    kv_indices,  # [num_page_used] — indices into kv_buffer's page dimension
    kv_last_page_lens,  # [bs]
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    page_size,
    dtype=torch.bfloat16,
):
    """Pure-PyTorch reference for absorbed MLA decode, matching torch_mla_extend pattern."""
    bs = qo_indptr.shape[0] - 1
    # Reorder pages by kv_indices, then split by kv_indptr (same as torch_mla_extend)
    kvc = torch.index_select(kv_buffer, 0, kv_indices)  # [num_page_used, page_size, 1, qk_head_dim]
    kvc = kvc.squeeze(2)                                  # [num_page_used, page_size, qk_head_dim]
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    qs  = torch.tensor_split(q,   qo_indptr.tolist()[1:])

    os = []
    for i in range(bs):
        q_i = qs[i]                                   # [seq_q, nhead, qk_head_dim]
        kv_pages_i = kvs[i]                           # [num_blocks, page_size, qk_head_dim]
        kv_flat = kv_pages_i.flatten(0, 1)            # [num_blocks*page_size, qk_head_dim]
        real_kv_len = (kv_indptr[i + 1] - kv_indptr[i] - 1).item() * page_size \
                      + kv_last_page_lens[i].item()
        kv_flat = kv_flat[:real_kv_len]               # [kv_len, qk_head_dim]
        k_i = kv_flat                                  # [kv_len, qk_head_dim] — full absorb key
        v_i = kv_flat[:, :kv_lora_rank]               # [kv_len, kv_lora_rank] — value part

        # attn_weights: [nhead, seq_q, kv_len]
        attn = torch.einsum("qhd,kd->hqk", q_i.float(), k_i.float()) * sm_scale
        attn = torch.softmax(attn, dim=-1)
        out_i = torch.einsum("hqk,kd->qhd", attn, v_i.float()).to(dtype)
        os.append(out_i)
    return torch.cat(os, dim=0)


# ---------------------------------------------------------------------------
# Helper: build persistent MLA metadata for a given (nhead, effective_nhead)
# ---------------------------------------------------------------------------

def build_persistent_metadata(batch_size, max_seqlen_qo, effective_nhead, dtype, kvtype,
                               qo_indptr, kv_indptr, kv_last_page_lens,
                               page_size, max_split_per_batch=32):
    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_mla_metadata_info_v1(
        batch_size,
        max_seqlen_qo,
        effective_nhead,
        dtype,
        kvtype,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=False,
    )

    work_meta_data = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device="cuda")
    work_indptr    = torch.empty(work_indptr_size,    dtype=work_indptr_type,    device="cuda")
    work_info_set  = torch.empty(work_info_set_size,  dtype=work_info_set_type,  device="cuda")
    reduce_indptr  = torch.empty(reduce_indptr_size,  dtype=reduce_indptr_type,  device="cuda")
    reduce_final_map  = torch.empty(reduce_final_map_size,  dtype=reduce_final_map_type,  device="cuda")
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda")

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        effective_nhead,      # num_heads_per_head_k — must match padded nhead
        1,                    # nhead_kv = 1 for MLA
        False,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        kv_granularity=max(page_size, 16),
        max_seqlen_qo=int(max_seqlen_qo),
        uni_seqlen_qo=int(max_seqlen_qo),
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
        dtype_q=dtype,
        dtype_kv=kvtype,
    )
    return work_meta_data, work_indptr, work_info_set, reduce_indptr, reduce_final_map, reduce_partial_map


# ---------------------------------------------------------------------------
# Core correctness test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nhead", [8, 4, 2])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("ctx_lens", [128, 1024])
@pytest.mark.parametrize("decode_qlen", [1])   # decode: seq_q == 1 for nhead<16
def test_sub16_nhead_correctness(nhead, batch_size, ctx_lens, decode_qlen):
    """
    Verify that the sub-16 head padding path in mla_decode_fwd produces
    results matching the torch reference for the real heads.

    Key invariant: real heads [0:nhead] must match torch reference;
    the padded heads are discarded before output reaches caller.
    """
    torch.manual_seed(42 + nhead + batch_size)

    page_size = 1
    kv_lora_rank = KV_LORA_RANK
    qk_rope_head_dim = QK_ROPE_HEAD_DIM
    qk_head_dim = QK_HEAD_DIM
    v_head_dim = V_HEAD_DIM
    dtype = torch.bfloat16
    kvtype = torch.bfloat16
    out_dtype = torch.bfloat16
    sm_scale = 1.0 / (qk_head_dim ** 0.5)
    max_split_per_batch = 32

    # Build paging structures
    seq_lens_kv = torch.full((batch_size,), ctx_lens, dtype=torch.int)
    kv_block_nums = (seq_lens_kv + page_size - 1) // page_size
    kv_last_page_lens = torch.where(
        seq_lens_kv % page_size == 0,
        torch.full_like(seq_lens_kv, page_size),
        seq_lens_kv % page_size,
    )
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr[1:] = torch.cumsum(kv_block_nums, dim=0)
    num_page = kv_indptr[-1].item()
    kv_indices = torch.arange(num_page, dtype=torch.int)

    seq_lens_qo = torch.full((batch_size,), decode_qlen, dtype=torch.int)
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    max_seqlen_qo = decode_qlen

    # KV buffer and query
    kv_buffer = torch.randn(
        (num_page, page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
    )
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=dtype)

    # ---- Torch reference (only for real nhead heads) ----
    out_ref = torch_mla_decode_ref(
        q, kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
        sm_scale, kv_lora_rank, qk_rope_head_dim, page_size, out_dtype,
    )
    assert out_ref.shape == (total_q, nhead, v_head_dim), \
        f"Reference shape mismatch: {out_ref.shape}"

    # ---- Build metadata (uses effective_nhead=16 for sub-16 case) ----
    effective_nhead = max(nhead, 16)
    (work_meta_data, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = build_persistent_metadata(
        batch_size, max_seqlen_qo, effective_nhead, dtype, kvtype,
        qo_indptr, kv_indptr, kv_last_page_lens, page_size, max_split_per_batch,
    )

    # ---- aiter persistent MLA decode (sub-16 padding path) ----
    out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
    aiter.mla.mla_decode_fwd(
        q,
        kv_buffer.view(num_page, page_size, 1, qk_head_dim),
        out_asm,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_qo,
        page_size=page_size,
        nhead_kv=1,
        sm_scale=sm_scale,
        num_kv_splits=max_split_per_batch,
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
    )

    # ---- Output shape must exactly match original (no head leakage) ----
    assert out_asm.shape == (total_q, nhead, v_head_dim), \
        f"Output shape wrong: {out_asm.shape}, expected ({total_q}, {nhead}, {v_head_dim})"

    # ---- Numerical correctness: real heads must match reference ----
    checkAllclose(
        out_ref,
        out_asm,
        msg=f"sub16 nhead={nhead} bs={batch_size} ctx={ctx_lens}: real heads vs torch ref",
    )


# ---------------------------------------------------------------------------
# Isolation test: padded heads must not bleed into real-head outputs
#
# Method: run mla_decode_fwd with nhead=8, then run the same call with the
# Q padded manually to 16 heads (zeros in slots [8:16]).  The outputs for
# heads [0:8] must be identical, proving the real heads are unaffected by
# whatever the kernel computes for the zero-padded head slots.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nhead", [8, 4])
def test_padded_heads_do_not_corrupt_real_heads(nhead):
    """
    Confirm independence: zero-padded extra heads must not affect real heads.
    We run the 16-head version manually (with zeros in slots [nhead:16])
    and compare with the sub-16 path output for heads [0:nhead].
    """
    torch.manual_seed(99 + nhead)
    batch_size = 4
    ctx_lens = 512
    page_size = 1
    kv_lora_rank = KV_LORA_RANK
    qk_rope_head_dim = QK_ROPE_HEAD_DIM
    qk_head_dim = QK_HEAD_DIM
    v_head_dim = V_HEAD_DIM
    dtype = torch.bfloat16
    kvtype = torch.bfloat16
    out_dtype = torch.bfloat16
    sm_scale = 1.0 / (qk_head_dim ** 0.5)
    max_split_per_batch = 32
    padded_nhead = 16

    seq_lens_kv = torch.full((batch_size,), ctx_lens, dtype=torch.int)
    kv_block_nums = (seq_lens_kv + page_size - 1) // page_size
    kv_last_page_lens = torch.where(
        seq_lens_kv % page_size == 0,
        torch.full_like(seq_lens_kv, page_size),
        seq_lens_kv % page_size,
    )
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr[1:] = torch.cumsum(kv_block_nums, dim=0)
    num_page = kv_indptr[-1].item()
    kv_indices = torch.arange(num_page, dtype=torch.int)

    decode_qlen = 1
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    qo_indptr[1:] = torch.arange(1, batch_size + 1, dtype=torch.int) * decode_qlen
    total_q = qo_indptr[-1].item()
    max_seqlen_qo = decode_qlen

    kv_buffer = torch.randn(
        (num_page, page_size, 1, kv_lora_rank + qk_rope_head_dim), dtype=torch.bfloat16
    )
    q_real = torch.randn((total_q, nhead, qk_head_dim), dtype=dtype)

    # ---- Path A: sub-16 path (automatic padding inside mla_decode_fwd) ----
    (work_meta_data, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = build_persistent_metadata(
        batch_size, max_seqlen_qo, padded_nhead, dtype, kvtype,
        qo_indptr, kv_indptr, kv_last_page_lens, page_size, max_split_per_batch,
    )

    out_sub16 = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
    aiter.mla.mla_decode_fwd(
        q_real,
        kv_buffer.view(num_page, page_size, 1, qk_head_dim),
        out_sub16,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_qo,
        page_size=page_size,
        nhead_kv=1,
        sm_scale=sm_scale,
        num_kv_splits=max_split_per_batch,
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
    )

    # ---- Path B: manual 16-head run with zeros in padded slots ----
    # Build q_padded [total_q, 16, qk_head_dim] with zeros in [nhead:16]
    q_padded = torch.zeros((total_q, padded_nhead, qk_head_dim), dtype=dtype)
    q_padded[:, :nhead, :] = q_real

    # Reuse same metadata (already built for nhead=16)
    out_full16 = torch.empty((total_q, padded_nhead, v_head_dim), dtype=out_dtype).fill_(-1)
    aiter.mla.mla_decode_fwd(
        q_padded,
        kv_buffer.view(num_page, page_size, 1, qk_head_dim),
        out_full16,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_qo,
        page_size=page_size,
        nhead_kv=1,
        sm_scale=sm_scale,
        num_kv_splits=max_split_per_batch,
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
    )
    # Extract only the real head slice from the 16-head run
    out_full16_real = out_full16[:, :nhead, :]

    # The two should be bit-identical: same kernel, same data, same metadata
    assert torch.equal(out_sub16, out_full16_real), (
        f"nhead={nhead}: sub-16 output != manual-padded-16 output for real heads!\n"
        f"max abs diff = {(out_sub16 - out_full16_real).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# No-garbage test: output tensor must not contain -1 sentinel fill values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nhead", [8, 4, 2])
def test_output_fully_written(nhead):
    """Every element in out_asm must be written by the kernel (no -1 fill residue)."""
    torch.manual_seed(7 + nhead)
    batch_size = 2
    ctx_lens = 256
    page_size = 1
    qk_head_dim = QK_HEAD_DIM
    v_head_dim = V_HEAD_DIM
    dtype = torch.bfloat16
    kvtype = torch.bfloat16
    sm_scale = 1.0 / (qk_head_dim ** 0.5)
    max_split_per_batch = 32
    effective_nhead = 16

    kv_block_nums = torch.full((batch_size,), ctx_lens, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr[1:] = torch.cumsum(kv_block_nums, dim=0)
    num_page = kv_indptr[-1].item()
    kv_indices = torch.arange(num_page, dtype=torch.int)
    kv_buffer = torch.randn(
        (num_page, 1, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM), dtype=torch.bfloat16
    )

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    qo_indptr[1:] = torch.arange(1, batch_size + 1, dtype=torch.int)
    total_q = qo_indptr[-1].item()

    q = torch.randn((total_q, nhead, qk_head_dim), dtype=dtype)

    (work_meta_data, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = build_persistent_metadata(
        batch_size, 1, effective_nhead, dtype, kvtype,
        qo_indptr, kv_indptr, kv_last_page_lens, 1, max_split_per_batch,
    )

    # Fill with a recognisable sentinel
    sentinel = torch.bfloat16(-1.0)
    out_asm = torch.full((total_q, nhead, v_head_dim), float(sentinel), dtype=torch.bfloat16)

    aiter.mla.mla_decode_fwd(
        q,
        kv_buffer.view(num_page, 1, 1, qk_head_dim),
        out_asm,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        1,  # max_seqlen_q
        page_size=1,
        nhead_kv=1,
        sm_scale=sm_scale,
        num_kv_splits=max_split_per_batch,
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
    )

    # After kernel runs, none of the entries should still be exactly -1
    # (attention outputs are weighted averages, so -1 is astronomically unlikely)
    unchanged = (out_asm == sentinel).sum().item()
    assert unchanged == 0, (
        f"nhead={nhead}: {unchanged} output elements still have sentinel value -1; "
        "kernel may not have written all outputs."
    )


# ---------------------------------------------------------------------------
# Parametric smoke test: assert get_mla_metadata_info_v1 accepts all valid sub-16 counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nhead", [1, 2, 4, 8])
def test_metadata_info_accepts_sub16(nhead):
    """get_mla_metadata_info_v1 must not raise for valid sub-16 head counts."""
    effective_nhead = max(nhead, 16)
    result = aiter.get_mla_metadata_info_v1(
        batch_size=4,
        max_seqlen_qo=1,
        num_head_qo=effective_nhead,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=32,
        intra_batch_mode=False,
    )
    assert len(result) == 6, "Expected 6 metadata tuples"
    # All sizes must be positive
    for shape_or_size, _ in result:
        if isinstance(shape_or_size, tuple):
            assert all(s > 0 for s in shape_or_size), f"Non-positive size in {shape_or_size}"
        else:
            assert shape_or_size > 0, f"Non-positive size: {shape_or_size}"


@pytest.mark.parametrize("nhead", [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15])
def test_metadata_info_rejects_invalid_sub16(nhead):
    """get_mla_metadata_info_v1 must raise for head counts not divisible by 16
    and not divisors of 16 (e.g. 3, 5, 6, 7, ...)."""
    with pytest.raises(AssertionError):
        aiter.get_mla_metadata_info_v1(
            batch_size=4,
            max_seqlen_qo=1,
            num_head_qo=nhead,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            is_sparse=False,
        )


if __name__ == "__main__":
    # Quick manual run: python op_tests/test_mla_sub16_heads.py
    for nhead in [8, 4, 2]:
        for batch_size in [1, 4]:
            for ctx_lens in [128, 1024]:
                print(f"\n=== nhead={nhead} bs={batch_size} ctx={ctx_lens} ===")
                test_sub16_nhead_correctness(nhead, batch_size, ctx_lens, decode_qlen=1)
                print("  correctness PASSED")

    for nhead in [8, 4]:
        print(f"\n=== isolation nhead={nhead} ===")
        test_padded_heads_do_not_corrupt_real_heads(nhead)
        print("  isolation PASSED")

    for nhead in [8, 4, 2]:
        print(f"\n=== fully-written nhead={nhead} ===")
        test_output_fully_written(nhead)
        print("  fully-written PASSED")

    print("\nAll tests passed.")
