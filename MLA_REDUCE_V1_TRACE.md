# MLA Reduce V1 Dimension Logging

## Overview

Simple dimension logging for `mla_reduce_v1` kernel during HIP graph capture phase.

**Important**: This only logs during graph capture, NOT during graph replay. Since kernels are replayed via HIP graphs in production, you'll see one log entry per unique graph captured, showing the dimensions for that graph.

## Usage

```bash
export AITER_LOG_MLA_REDUCE_V1=1
python -m sglang.launch_server --model your_model --backend aiter
```

## What Gets Logged

During graph capture, you'll see:

```
[aiter] [mla_reduce_v1] num_heads=16 head_dim=512 num_reduce_tile=304 max_seqlen_q=1 kNumThreadGroupPerBh=1
```

This tells you:
- **num_heads**: Number of attention heads (16 in example)
- **head_dim**: Head dimension size (512 or 128)
- **num_reduce_tile**: Number of tiles being reduced
- **max_seqlen_q**: Maximum sequence length for this operation
- **kNumThreadGroupPerBh**: Selected parallelism level (1, 2, 4, 8, 16, 64, or 256)

## Matching to Kernel Symbol

For kernel symbol like:
```
_Z19kn_mla_reduce_v1_psI23MlaReduceKernelV1TraitsILi512ELi16ELi1EEfDF16bEv23MlaReduceKernelV1Params
```

This corresponds to template parameters `<512, 16, 1>`:
- `512` = head_dim (kSizeDV)
- `16` = num_heads (kNumHeadQ)
- `1` = kNumThreadGroupPerBh

So when you see the log:
```
[aiter] [mla_reduce_v1] num_heads=16 head_dim=512 ... kNumThreadGroupPerBh=1
```

You know it's instantiating the `<512, 16, 1>` variant.

## Why Only Graph Capture?

SGLang and other frameworks use HIP graph replay for performance. The kernel is:
1. **Captured once** during graph construction (you see the log here)
2. **Replayed many times** during inference (no logs, for performance)

This approach gives you the dimensions without flooding your logs with repeated entries during inference.

## Kernel Variant Selection

- **kNumThreadGroupPerBh=1**: No QO parallelism (many tiles, GPU well-utilized)
- **kNumThreadGroupPerBh>1**: QO parallelism enabled (few tiles, extra parallelism needed)

The promotion factor (changed from 1.3→2.0 in recent commit) affects when higher parallelism is selected.

## Collecting Dimensions

To collect dimensions for different workloads:

```bash
# Clear previous logs
rm -f /tmp/mla_dimensions.txt

# Run with logging (redirecting aiter logs)
AITER_LOG_MLA_REDUCE_V1=1 AITER_LOG_LEVEL=INFO \
    python -m sglang.launch_server <args> 2>&1 | \
    grep "mla_reduce_v1" | tee /tmp/mla_dimensions.txt

# Analyze unique configurations
cat /tmp/mla_dimensions.txt | sort | uniq
```

## Example Output

During server startup/graph capture:
```
[aiter] [mla_reduce_v1] num_heads=16 head_dim=512 num_reduce_tile=304 max_seqlen_q=1 kNumThreadGroupPerBh=1
[aiter] [mla_reduce_v1] num_heads=16 head_dim=512 num_reduce_tile=150 max_seqlen_q=2 kNumThreadGroupPerBh=2
```

During inference: (no additional logs - graphs are being replayed)
