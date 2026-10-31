# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression test for ROCm/aiter#6518: mha_batch_prefill must not read past the
end of K or V when head_dim is smaller than the kernel's head-dim tile.

The CK batch-prefill kernels only exist for head_dim 128 and 256, so head_dim
64 and 96 run the 128 tile with head-dim padding. With page_size 1 (and other
page sizes below the K tile) the kernel uses flat global loads, which have no
buffer-descriptor range check. Without a bounds check on the padded columns,
the last KV head of the last token reads the bytes right after the tensor:
NaN there corrupts the output, and an unmapped page is a GPU memory fault.

Two checks:
  * test_padded_hdim_k_tail_nan: K is followed by a NaN row in the same
    allocation. The output must stay finite and match a float32 reference.
  * test_padded_hdim_no_read_past_end: K or V is placed flush against the end
    of its own allocation (PyTorch caching allocator off, in a subprocess). A
    read past the end faults, which the parent reports as a failure.
"""

import math
import os
import subprocess
import sys

import pytest
import torch

import aiter

NUM_QO_HEADS, NUM_KV_HEADS = 64, 8
ALLOC_GRANULE = 2 << 20  # place the tensor at the end of a 2 MiB-multiple allocation


def _reference(q, k, v, cu_seqlens, scale):
    out = []
    group = q.shape[1] // k.shape[1]
    for b in range(len(cu_seqlens) - 1):
        s, e = cu_seqlens[b], cu_seqlens[b + 1]
        n = e - s
        qb = q[s:e].float().transpose(0, 1)
        kb = k[s:e].float().repeat_interleave(group, dim=1).transpose(0, 1)
        vb = v[s:e].float().repeat_interleave(group, dim=1).transpose(0, 1)
        scores = qb @ kb.transpose(1, 2) * scale
        causal = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
        p = torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)
        out.append((p @ vb).transpose(0, 1))
    return torch.cat(out)


def _prefill(q, k, v, seqlens):
    """One batch-prefill call: LINEAR layout, page_size 1, SGLang 1-D page table."""
    cu = [0]
    for n in seqlens:
        cu.append(cu[-1] + n)
    total = cu[-1]
    indptr = torch.tensor(cu, dtype=torch.int32, device=q.device)
    page_indices = torch.arange(total, dtype=torch.int32, device=q.device)
    out = aiter.mha_batch_prefill_func(
        q,
        k,
        v,
        indptr,
        indptr,
        page_indices,
        max(seqlens),
        max(seqlens),
        causal=True,
    )
    torch.cuda.synchronize()
    return out, cu


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"batch prefill global-load path is CDNA3+ only (got {arch})")


@pytest.mark.parametrize("head_dim", [64, 96])
@pytest.mark.parametrize("seqlens", [[150], [129], [300, 77]])
def test_padded_hdim_k_tail_nan(head_dim, seqlens):
    _skip_unless_supported()
    torch.manual_seed(0)
    total = sum(seqlens)
    q = torch.randn(total, NUM_QO_HEADS, head_dim, device="cuda", dtype=torch.bfloat16)
    k_buf = torch.randn(
        total + 1, NUM_KV_HEADS, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k_buf[total] = float("nan")  # the row right after K
    v = torch.randn(total, NUM_KV_HEADS, head_dim, device="cuda", dtype=torch.bfloat16)
    k = k_buf[:total]
    out, cu = _prefill(q, k, v, seqlens)
    assert not torch.isnan(out).any(), "the kernel read the row after K into the result"
    ref = _reference(q, k, v, cu, 1.0 / math.sqrt(head_dim))
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def _child(which, head_dim, seqlens):
    """Runs in a fresh process with PYTORCH_NO_CUDA_MEMORY_CACHING=1."""
    torch.manual_seed(0)
    total = sum(seqlens)

    def tensor(shape, at_end):
        if not at_end:
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        numel = math.prod(shape)
        nbytes = numel * 2
        alloc = (nbytes + ALLOC_GRANULE - 1) // ALLOC_GRANULE * ALLOC_GRANULE
        buf = torch.zeros(alloc // 2, device="cuda", dtype=torch.bfloat16)
        t = buf[alloc // 2 - numel :].view(shape)
        t.normal_()
        return t

    q = tensor((total, NUM_QO_HEADS, head_dim), False)
    k = tensor((total, NUM_KV_HEADS, head_dim), which == "k")
    v = tensor((total, NUM_KV_HEADS, head_dim), which == "v")
    out, cu = _prefill(q, k, v, seqlens)
    ref = _reference(q, k, v, cu, 1.0 / math.sqrt(head_dim))
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    print("child ok")


@pytest.mark.parametrize("which", ["k", "v"])
@pytest.mark.parametrize("head_dim", [64, 96])
@pytest.mark.parametrize("seqlens", [[150], [300, 77]])
def test_padded_hdim_no_read_past_end(which, head_dim, seqlens):
    _skip_unless_supported()
    env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1", HSA_ENABLE_COREDUMP="0")
    args = [sys.executable, os.path.abspath(__file__), "--child", which, str(head_dim)]
    args += [str(n) for n in seqlens]
    r = subprocess.run(
        args, env=env, capture_output=True, text=True, timeout=900, check=False
    )
    lines = (r.stdout + r.stderr).splitlines()
    # the cause (e.g. "illegal memory access", "Memory access fault") before the tail
    cause = [
        ln
        for ln in lines
        if not ln.startswith("[aiter]")
        and any(w in ln.lower() for w in ("fault", "illegal", "error"))
    ]
    tail = "\n".join(cause[:5] + ["..."] + lines[-8:])
    assert r.returncode == 0 and "child ok" in r.stdout, (
        f"batch prefill with {which.upper()} at the end of its allocation "
        f"(head_dim {head_dim}, seqlens {seqlens}) failed, rc={r.returncode}:\n{tail}"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child(sys.argv[2], int(sys.argv[3]), [int(n) for n in sys.argv[4:]])
    else:
        sys.exit(pytest.main([__file__, "-v"]))
