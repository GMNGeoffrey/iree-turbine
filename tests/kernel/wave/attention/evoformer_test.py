# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from torch.nn import functional as F
import math
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    get_default_run_config,
    device_randn,
    device_zeros,
    device_randint,
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.evoformer import get_evoformer_kernel
from iree.turbine.kernel.lang import DataType
import os
from ..common.utils import (
    require_e2e,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
from ..common.shapes import get_test_shapes


default_tile_sizes = [(1, 1, 32, 1, None, 64, 32)]


# From: https://github.com/microsoft/DeepSpeed/blob/master/tests/unit/ops/deepspeed4science/test_DS4Sci_EvoformerAttention.py
def attention_reference(
    q_input: torch.Tensor,      # B x BN x  M  x H x K1
    k_input: torch.Tensor,      # B x BN x K2  x H x K1
    v_input: torch.Tensor,      # B x BN x K2  x H x  N
    biases: list[torch.Tensor], # B x BN x  1  x 1 x K2
                                # B x  1 x  H  x M x K2
    sm_scale: float,
) -> torch.Tensor:
    q = q_input.transpose(-2, -3)        # B x BN x H x  M x K1
    k = k_input.transpose(-2, -3)        # B x BN x H x K2 x K1
    v = v_input.transpose(-2, -3)        # B x BN x H x K2 x  N
    k_t = k.transpose(-1, -2)            # B x BN x H x K1 x K2
    a = torch.matmul(q, k_t) * sm_scale  # B x BN x H x  M x K2

    for b in biases:
        a += b                           # B x BN x H x  M x K2

    a = F.softmax(a, dim=-1)             # B x BN x H x  M x K2
    a_v = torch.matmul(a, v)             # B x BN x H x  M x  N
    o = a_v.transpose(-2, -3)            # B x BN x M x  H x  N

    return o


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("evoformer"))
@pytest.mark.parametrize("tile_sizes", default_tile_sizes)
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_32x32x8_F16,
    ],
)
@pytest.mark.parametrize("dtype", [tkl.f16, tkl.bf16])
def testEvoformerAttentionForward(
    shape: tuple[int],
    tile_sizes: tuple[int],
    enable_scheduling: bool,
    mfma_variant: MMAType,
    dtype: DataType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    shapes_and_tile_sizes = [(x, y) for x, y in zip(shape, tile_sizes)]
    evoformer_fwd, evoformer_bwd, symbols = get_evoformer_kernel(
        *shapes_and_tile_sizes, mfma_variant, dtype
    )

    symbols.update(get_default_scheduling_params())

    config = get_default_run_config()
    if run_bench:
        config["benchmark_batch_size"] = 1000
        config["benchmark_repetitions"] = 3
    if dump_perf is not None:
        perf_filename = request.node.name + ".json"
        config["benchmark_results_file"] = os.path.join(
            dump_perf, "tk_" + perf_filename
        )
    compile_config = {"waves_per_eu": 2, "denorm_fp_math_f32": "preserve-sign"}

    with tk.gen.TestLaunchContext(
        symbols,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        compile_config=compile_config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        torch.manual_seed(0)
        if dtype == tkl.bf16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        # # Order of shapes: (B, BN, K2, H, K1, M, N)
        # default_test_shapes = [(1, 256, 256, 4, 32, 256, 32), (1, 512, 256, 8, 8, 256, 8)]
        # B = batch
        # BN = n
        # K2 = kv_seq_len
        # H = heads
        # K1 = head_dim
        # M = q_seq_len
        # N = v_dim

        # Simplification:
        #  - kv_seq_len = q_seq_len (K2 = M)
        #  - head_dim = v_dim (K1 = N)

        batch, n, kv_seq_len, heads, head_dim, q_seq_len, v_dim = shape
        q = device_randn(batch, n, q_seq_len, heads, head_dim, dtype=torch_dtype)
        k = device_randn(batch, n, kv_seq_len, heads, head_dim, dtype=torch_dtype)
        v = device_randn(batch, n, kv_seq_len, heads, v_dim, dtype=torch_dtype)
        mask = device_randint(0, 2, (batch, n, kv_seq_len), dtype=torch_dtype)
        mask_bias = 1e9 * (mask - 1)
        bias = device_randn(batch, heads, q_seq_len, kv_seq_len, dtype=torch_dtype)
        output = device_zeros(batch, n, q_seq_len, heads, v_dim, dtype=torch_dtype)
        lse = device_zeros(batch, n, heads, q_seq_len, dtype=torch_dtype)
        # The kernel uses log base 2 instead of base e, so we need to scale by
        # this constant factor.
        log2e = 1.44269504089
        dk_sqrt = math.sqrt(1.0 / head_dim)
        # TODO: Add scaling of QK as part of kernel.
        # TODO: Add v-permute as part of kernel.
        # mb = evoformer_fwd(
        #     q * dk_sqrt * log2e,
        #     k,
        #     v.permute([0, 1, 4, 3, 2]),
        #     mask_bias,
        #     bias * log2e,
        #     output,
        #     lse,
        # )


        o = output.transpose(-2, -3)
        # pretend gradient from loss function
        do = device_randn(batch, n, heads, q_seq_len, v_dim, dtype=torch_dtype)

        # output tensors
        dq = device_zeros(batch, n, heads, q_seq_len, head_dim, dtype=torch_dtype)
        dk = device_zeros(batch, n, heads, kv_seq_len, head_dim, dtype=torch_dtype)
        dv = device_zeros(batch, n, heads, kv_seq_len, v_dim, dtype=torch_dtype)
        # dbias = device_zeros(batch, heads, q_seq_len, kv_seq_len, dtype=torch_dtype)

        # TODO: maybe compute this within the kernel? The description of the
        # backward pass in the Flash Attention 2 paper has it computed
        # external to both loops though.
        D = torch.sum(do * o, -1)

        q_perm = q.transpose(-2, -3)
        k_perm = k.transpose(-2, -3)
        v_perm = v.transpose(-2, -3)
        # print(f"{shape=}")
        # print(f"{do.shape=}\n{o.shape=}\n{D.shape=}\n{q_perm.shape=}\n{k_perm.shape=}\n{v_perm.shape=}\n{lse.shape}\n{dq.shape=}\n{dk.shape=}\n{dv.shape=}")


        # (B, BN, K2, H, K1, M, N)
        # (1, 512, 128, 4, 16, 256, 8)
        # 1=B
        # 512=BN
        # 128=K2
        # 4=H
        # 16=K1
        # 256=M
        # 8=N

        mb_bwd = evoformer_bwd(
            do,
            D,
            q_perm * dk_sqrt * log2e,
            k_perm,
            v_perm,
            # mask_bias,
            # bias,
            # o,
            lse,
            dq,
            dk,
            dv,
            # dbias,
        )

        mask_bias = mask_bias.view([batch, n, 1, 1, kv_seq_len])
        bias = bias.view([batch, 1, heads, q_seq_len, kv_seq_len])
        torch_ref = attention_reference(q, k, v, [mask_bias, bias], dk_sqrt)

        if dump_generated_mlir:
            filename = f"wave_evoformer_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        eps = 1e-2 if output.dtype == torch.float16 else 5e-2
        assert (
            torch.max(torch.abs(torch_ref - output)).item() < eps
        ), f"out eps: {torch.max(torch.abs(torch_ref - output))}"
