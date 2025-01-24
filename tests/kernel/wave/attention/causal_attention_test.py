# Copyright 2025 The IREE Authors
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
    get_default_scheduling_params,
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
    device_randn,
    device_zeros,
)
from iree.turbine.kernel.wave.constraints import MMAType
import os
from torch.testing import assert_close
from ..common.utils import (
    require_e2e,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
from ..common.shapes import get_test_shapes

# Pytest seems to change its mind about what order it does parametrizations in.
# Before it was top to bottom.
shapes_16x16x16 = reversed(
    [
        (8, 128, 128, 64, 256),
        (40, 1024, 64, 64, 1024),
        (2, 64, 128, 32, 256),
        (1, 16, 16, 16, 32),
        (1, 16, 16, 32, 16),
        (1, 16, 32, 16, 16),
        (1, 32, 16, 16, 16),
        (2, 16, 16, 16, 16),
        (1, 16, 16, 16, 16),
    ]
)


def get_param_id(val):
    if isinstance(val, tuple) and all(isinstance(el, int) for el in val):
        return "x".join(str(el) for el in val)
    elif isinstance(val, MMAType):
        return f"MMA_{val.name}"


def get_causal_attention_kernel(
    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len, mfma_variant
):
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    MMA_INPUT_ELS_PER_THREAD = tkl.sym.MMA_INPUT_ELS_PER_THREAD
    MMA_OUTPUT_ELS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 1)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]
    constraints += [tkw.WaveConstraint(B, BLOCK_B / 1)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: vec_size, N: vec_size},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_n_m_write_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )
    flip_k2_n_read_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, K2: k, N: j}, outputs={B: i, N: j, K2: k}
    )

    @tkw.wave(constraints)
    def attention_causal(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        o: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        init_m = tkl.Register[B, M, tkl.f32](-1e6)
        init_l = tkl.Register[B, M, tkl.f32](0.0)
        init_o = tkl.Register[B, N, M, tkl.f32](0.0)

        @tkw.reduction(K2, init_args=[init_m, init_l, init_o])
        def repeat(
            m_prev: tkl.Register[B, M, tkl.f32],
            l_prev: tkl.Register[B, M, tkl.f32],
            o_ij: tkl.Register[B, N, M, tkl.f32],
        ):
            s_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            log2e = tkl.Register[B, M, tkl.f32](1.44269504089)
            q_i = tkw.read(q, elements_per_thread=MMA_INPUT_ELS_PER_THREAD)
            k_j = tkw.read(k, elements_per_thread=MMA_INPUT_ELS_PER_THREAD)
            s_ij = tkw.mma(k_j, q_i, s_acc)
            s_ij = tkw.permute(s_ij, target_shape=[B, M, K2])
            m_ij = tkw.max(s_ij, m_prev, dim=K2)
            e_delta_max = tkw.exp2(log2e * (m_prev - m_ij))
            p_ij = tkw.exp2(log2e * (s_ij - m_ij))
            l_init = l_prev * e_delta_max
            l_ij = tkw.sum(p_ij, l_init, dim=K2)
            v_j = tkw.read(v, mapping=flip_k2_n_read_mapping, elements_per_thread=MMA_INPUT_ELS_PER_THREAD)
            o_acc = e_delta_max * o_ij
            o_ij = tkw.mma(v_j, tkw.cast(p_ij, tkl.f16), o_acc)
            return m_ij, l_ij, o_ij

        m_i, l_i, o_i = repeat
        o_i = tkw.reciprocal(l_i) * o_i
        tkw.write(
            tkw.cast(o_i, tkl.f16),
            o,
            mapping=flip_n_m_write_mapping,
            elements_per_thread=MMA_OUTPUT_ELS_PER_THREAD,
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        MMA_INPUT_ELS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        MMA_OUTPUT_ELS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: min(128, q_seq_len),
        BLOCK_N: min(64, v_head_dim),
        BLOCK_K2: min(64, kv_seq_len),
        B: batch,
        M: q_seq_len,
        N: v_head_dim,
        K1: qk_head_dim,
        K2: kv_seq_len,
    }

    return attention_causal, hyperparams


@require_e2e
@pytest.mark.parametrize("shape", shapes_16x16x16, ids=get_param_id)
@pytest.mark.parametrize("mfma_variant", [MMAType.F32_16x16x16_F16], ids=get_param_id)
def testCausalAttention(shape, mfma_variant, request):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")

    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape

    # Our float tolerances here are really bad on mi210
    tols = dict(atol=3e-2, rtol=5e-3)

    torch.manual_seed(0)
    q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16)
    # q = torch.full((batch, q_seq_len, qk_head_dim), 0.1, device=get_default_device(), dtype=torch.float16)
    # q = torch.arange(
    #     (batch * q_seq_len * qk_head_dim), dtype=torch.float16, device=get_default_device()
    # ).reshape((batch, q_seq_len, qk_head_dim)) / (batch * q_seq_len * qk_head_dim)
    # q[0, :, (qk_head_dim//2):] = 0

    k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16)
    # k = torch.full((batch, kv_seq_len, qk_head_dim), 0.2, device=get_default_device(), dtype=torch.float16)
    # k = torch.arange(
    #     (batch * kv_seq_len * qk_head_dim), dtype=torch.float16
    # ).reshape((batch, kv_seq_len, qk_head_dim)) / (batch * kv_seq_len * qk_head_dim)
    # k[0, :, (qk_head_dim//2):] = 0

    v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16)
    # v = torch.full((batch, kv_seq_len, v_head_dim), 0.3, device=get_default_device(), dtype=torch.float16)
    # v = torch.arange(
    #     (batch * kv_seq_len * v_head_dim), dtype=torch.float16
    # ).reshape((batch, kv_seq_len, v_head_dim)) / 64

    o_ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=False, scale=1
    )

    causal_attention, hyperparams = get_causal_attention_kernel(
        batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len, mfma_variant
    )

    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    if run_bench:
        config["benchmark_batch_size"] = 10
        config["benchmark_repetitions"] = 3
    if dump_perf is not None:
        perf_filename = request.node.name + ".json"
        config["benchmark_results_file"] = os.path.join(
            dump_perf, "tk_" + perf_filename
        )
    compile_config = {"waves_per_eu": 2, "denorm_fp_math_f32": "preserve-sign"}

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        compile_config=compile_config,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        output = device_zeros(batch, q_seq_len, v_head_dim, dtype=torch.float16)
        mb = causal_attention(q, k, v, output)

        if dump_generated_mlir:
            filename = f"out/wave_attention_causal_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        assert_close(output, o_ref, **tols)
