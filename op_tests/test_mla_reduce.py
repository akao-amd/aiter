# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Standalone unit test for aiter.mla_reduce_v1 (kn_mla_reduce_v1 / kn_mla_reduce_v1_ps).
#
# Unlike test_mla_prefill_ps.py this file exercises the reduce kernel in isolation:
# partial outputs and LSE values are constructed synthetically in Python so that
# the reference answer is cheap to compute without running the prefill kernel.
#
# Usage:
#   # quick smoke test (default params)
#   python op_tests/test_mla_reduce.py
#
#   # specific shapes
#   python op_tests/test_mla_reduce.py -b 4 -n 16 -dv 512 -ns 8
#
#   # sweep many configs
#   python op_tests/test_mla_reduce.py --sweep
#
#   # run as pytest
#   pytest op_tests/test_mla_reduce.py -v

import argparse
import itertools
import sys

import torch
import pytest

import aiter
from aiter.test_common import checkAllclose
from aiter.jit.utils.chip_info import get_gfx

# This kernel is gfx950-only
if get_gfx() == "gfx942":
    aiter.logger.info(
        "Skipping test_mla_reduce.py: only supported on gfx950, not gfx942"
    )
    sys.exit(0)

DEVICE = "cuda:0"
torch.set_default_device(DEVICE)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def ref_mla_reduce(
    partial_output: torch.Tensor,  # [num_splits, num_tokens, num_heads, dv]  fp32
    partial_lse: torch.Tensor,     # [num_splits, num_tokens, num_heads]       fp32
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge split-KV partial results using the log-sum-exp rescaling trick.

    output[s,h] = sum_i  exp(lse[i,s,h] - lse_global[s,h]) * partial_output[i,s,h]

    Returns (output [num_tokens, num_heads, dv] bf16,
             lse    [num_tokens, num_heads]      fp32).
    """
    # partial_lse: [num_splits, num_tokens, num_heads]
    lse_global = torch.logsumexp(partial_lse.float(), dim=0)  # [tokens, heads]
    scale = torch.exp(partial_lse.float() - lse_global.unsqueeze(0))  # [splits, tokens, heads]
    # weighted sum over splits
    out = (scale.unsqueeze(-1) * partial_output.float()).sum(dim=0)  # [tokens, heads, dv]
    return out.to(torch.bfloat16), lse_global


# ---------------------------------------------------------------------------
# Metadata builder — no metadata kernel required
# ---------------------------------------------------------------------------

def build_reduce_metadata(
    num_tiles: int,
    splits_per_tile: list[int],
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """
    Build reduce_indptr and reduce_partial_map tensors from a per-tile split list.

    reduce_indptr[i]   = cumulative number of splits before tile i
    reduce_partial_map = flattened list of per-split slot indices (identity mapping here:
                         split j of tile i occupies slot reduce_indptr[i] + j)

    reduce_final_map is None — we use the implicit uniform-qo_len path
    (use_reduce_final_map=False in the kernel).
    """
    assert len(splits_per_tile) == num_tiles

    indptr = [0]
    partial_map = []
    for n in splits_per_tile:
        base = indptr[-1]
        indptr.append(base + n)
        partial_map.extend(range(base, base + n))

    reduce_indptr = torch.tensor(indptr, dtype=torch.int32, device=DEVICE)
    reduce_partial_map = torch.tensor(partial_map, dtype=torch.int32, device=DEVICE)
    return reduce_indptr, reduce_partial_map


# ---------------------------------------------------------------------------
# Core test helper
# ---------------------------------------------------------------------------

def run_one(
    batch_size: int,
    num_heads: int,
    dv: int,
    num_splits: int,           # splits per tile (uniform across tiles for simplicity)
    num_tokens_per_tile: int,  # qo tokens per reduce tile
    output_lse: bool,
    label: str = "",
) -> bool:
    """
    Build synthetic partial_output and partial_lse, call mla_reduce_v1,
    compare against the reference.  Returns True on pass.
    """
    num_tiles = batch_size
    total_slots = num_tiles * num_splits   # total partial-tile slots
    total_tokens = num_tiles * num_tokens_per_tile

    # ---- synthetic partial outputs & LSE ----
    # partial_output layout expected by kernel: [total_slots * num_tokens_per_tile, heads, dv]
    # The kernel indexes: slot_idx * num_heads * dv  (via reduce_partial_map)
    # For each tile the slots are [base .. base+num_splits)
    partial_output_flat = torch.randn(
        total_slots * num_tokens_per_tile, num_heads, dv, dtype=torch.float32
    )
    # partial_lse layout: [total_slots * num_tokens_per_tile, num_heads]
    partial_lse_flat = torch.randn(
        total_slots * num_tokens_per_tile, num_heads, dtype=torch.float32
    )

    # ---- reference: reshape to [tiles, splits, tokens, heads, dv] ----
    # slot ordering: tile0_split0, tile0_split1, ..., tile1_split0, ...
    # Each slot covers num_tokens_per_tile rows.
    po_by_slot = partial_output_flat.view(total_slots, num_tokens_per_tile, num_heads, dv)
    lse_by_slot = partial_lse_flat.view(total_slots, num_tokens_per_tile, num_heads)

    ref_outputs = []
    ref_lses = []
    for t in range(num_tiles):
        base = t * num_splits
        po_tile  = po_by_slot[base : base + num_splits]   # [splits, tokens, heads, dv]
        lse_tile = lse_by_slot[base : base + num_splits]  # [splits, tokens, heads]
        ref_o, ref_l = ref_mla_reduce(po_tile, lse_tile)
        ref_outputs.append(ref_o)
        ref_lses.append(ref_l)

    ref_out = torch.cat(ref_outputs, dim=0)   # [total_tokens, heads, dv]
    ref_lse = torch.cat(ref_lses, dim=0)      # [total_tokens, heads]

    # ---- build metadata ----
    splits_per_tile = [num_splits] * num_tiles
    reduce_indptr, reduce_partial_map = build_reduce_metadata(num_tiles, splits_per_tile)

    # reduce_final_map=None → kernel uses implicit uniform qo_len
    # stride_s_o = num_heads * dv,  stride_h_o = dv  (contiguous [tokens, heads, dv])
    final_output = torch.empty(total_tokens, num_heads, dv, dtype=torch.bfloat16)
    final_lse_tensor = torch.empty(total_tokens, num_heads, dtype=torch.float32) if output_lse else None

    aiter.mla_reduce_v1(
        partial_output_flat,
        partial_lse_flat,
        reduce_indptr,
        None,                 # reduce_final_map — use implicit path
        reduce_partial_map,
        num_tokens_per_tile,  # max_seqlen_q
        final_output,
        final_lse_tensor,
    )
    torch.cuda.synchronize()

    # ---- compare ----
    tag = (
        f"{label}batch={batch_size} heads={num_heads} dv={dv} "
        f"splits={num_splits} tokens_per_tile={num_tokens_per_tile} lse={output_lse}"
    )
    out_err  = checkAllclose(final_output.float(), ref_out.float(),
                             rtol=1e-2, atol=1e-2, msg=f"[output] {tag} ")
    lse_pass = True
    if output_lse:
        lse_err = checkAllclose(final_lse_tensor, ref_lse,
                                rtol=1e-4, atol=1e-4, msg=f"[lse]    {tag} ")
        lse_pass = (lse_err == 0)

    return (out_err == 0) and lse_pass


# ---------------------------------------------------------------------------
# Pytest parametrize
# ---------------------------------------------------------------------------

# Covers:
#  - simple path  (num_splits < 4)
#  - massive path, kUpTo64Splits  (4 <= num_splits <= 64)
#  - massive path, kUpTo256Splits (65 <= num_splits <= 256)
#  - persistent vs non-persistent dispatch (controlled by total work vs ps_grid_size)

_CONFIGS = [
    # (batch, heads, dv, splits, tokens_per_tile, output_lse)
    # --- simple path ---
    (1,  16, 512, 2, 1,   False),
    (1,  16, 512, 3, 1,   True),
    # --- massive kUpTo64Splits ---
    (1,  16, 512, 4, 1,   False),
    (1,  16, 512, 8, 1,   True),
    (4,  16, 512, 16, 1,  False),
    (4,  16, 512, 32, 1,  True),
    (1,  16, 512, 64, 1,  False),
    # --- massive kUpTo256Splits ---
    (1,  16, 512, 65, 1,  False),
    (1,  16, 512, 128, 1, True),
    (1,  16, 512, 256, 1, False),
    # --- non-default dv ---
    (1,  16, 128, 8,  1,  False),
    # --- multi-token tiles ---
    (4,  16, 512, 8,  4,  False),
    (4,  16, 512, 8,  4,  True),
    # --- large batch (exercises persistent dispatch) ---
    (16, 16, 512, 4,  1,  False),
    (32, 16, 512, 4,  1,  True),
]


@pytest.mark.parametrize(
    "batch_size,num_heads,dv,num_splits,tokens_per_tile,output_lse",
    _CONFIGS,
    ids=[
        f"b{b}_h{h}_dv{dv}_s{s}_t{t}_lse{l}"
        for b, h, dv, s, t, l in _CONFIGS
    ],
)
def test_mla_reduce(batch_size, num_heads, dv, num_splits, tokens_per_tile, output_lse):
    ok = run_one(batch_size, num_heads, dv, num_splits, tokens_per_tile, output_lse)
    assert ok, "mla_reduce_v1 output did not match reference"


# ---------------------------------------------------------------------------
# CLI sweep / quick run
# ---------------------------------------------------------------------------

def _sweep():
    batches       = [1, 4, 16, 32]
    heads         = [16]
    dvs           = [512, 128]
    splits        = [2, 3, 4, 8, 16, 32, 64, 128, 256]
    tokens        = [1, 4]
    lse_opts      = [False, True]

    passed = failed = 0
    for b, h, dv, s, t, l in itertools.product(batches, heads, dvs, splits, tokens, lse_opts):
        ok = run_one(b, h, dv, s, t, l, label="")
        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    aiter.logger.info(
        f"\nSweep done: \033[32mpassed {passed}/{total}\033[0m  "
        f"\033[31mfailed {failed}/{total}\033[0m"
    )
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unit test for mla_reduce_v1")
    parser.add_argument("-b", "--batch_size",       type=int, default=4)
    parser.add_argument("-n", "--num_heads",         type=int, default=16)
    parser.add_argument("-dv", "--dv",              type=int, default=512)
    parser.add_argument("-ns", "--num_splits",       type=int, default=8)
    parser.add_argument("-t",  "--tokens_per_tile",  type=int, default=1)
    parser.add_argument("--no_lse", action="store_true")
    parser.add_argument("--sweep",  action="store_true",
                        help="Run a full parameter sweep instead of a single config")
    args = parser.parse_args()

    if args.sweep:
        ok = _sweep()
    else:
        ok = run_one(
            args.batch_size,
            args.num_heads,
            args.dv,
            args.num_splits,
            args.tokens_per_tile,
            output_lse=not args.no_lse,
        )

    sys.exit(0 if ok else 1)
