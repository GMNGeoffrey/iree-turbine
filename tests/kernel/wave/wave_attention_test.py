# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
from torch.nn import functional as F
import math
import unittest
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.iree_utils import generate_iree_ref
from iree.turbine.kernel.wave.utils import (
    get_default_run_config,
    get_default_arch,
    get_default_scheduling_params,
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
    device_randn,
    device_zeros,
    to_default_device,
)
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.decode_attention import (
    get_decode_attention_kernels,
)
import os
import json
from torch.testing import assert_close
from enum import Enum


require_e2e = pytest.mark.require_e2e
require_cdna3 = pytest.mark.skipif(
    "gfx94" not in get_default_arch(), reason="Default device is not CDNA3"
)
# Whether to dump the generated MLIR module.
test_dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))
# Whether to use scheduling group barriers (needs LLVM fix).
enable_scheduling_barriers = int(os.environ.get("WAVE_USE_SCHED_BARRIERS", 0))

# Add test shapes for validation and performance testing.
perf_test = lambda *a: pytest.param(*a, marks=pytest.mark.perf_only)
scheduling_test = lambda *a: pytest.param(*a, marks=pytest.mark.scheduling)
dynamic_dims_test = lambda *a: pytest.param(*a, marks=pytest.mark.dynamic_dims)

param_scheduling = pytest.mark.parametrize(
    "enable_scheduling", [False, scheduling_test(True)], ids=["no_sched", "sched"]
)
param_no_scheduling = pytest.mark.parametrize(
    "enable_scheduling", [False], ids=["no_sched"]
)
param_dynamic_dims = pytest.mark.parametrize(
    "dynamic_dims", [False, dynamic_dims_test(True)], ids=["no_dyn", "dyn"]
)
param_no_dynamic_dims = pytest.mark.parametrize("dynamic_dims", [False], ids=["no_dyn"])

default_test_shapes = {}
# Order of shapes: (B, M, N, K1, K2)
default_test_shapes["test_attention"] = [
    (1, 64, 128, 16, 32),
    (1, 16, 16, 16, 32),
    (1, 16, 16, 32, 16),
    (1, 16, 32, 16, 16),
    (1, 32, 16, 16, 16),
    # (40, 1024, 64, 64, 1024),
    (1, 16, 16, 16, 16),
]
# default_test_shapes["test_attention"] += [
#     perf_test(x) for x in default_test_shapes["test_attention"]
# ]


user_specified_test_shapes = ""

test_params_path = os.environ.get("TEST_PARAMS_PATH", None)

if test_params_path:
    with open(test_params_path, "r") as file:
        user_specified_test_shapes = json.load(file)


def get_test_shapes(test_name: str) -> list[tuple[int]]:
    if test_name in user_specified_test_shapes:
        test_shapes = user_specified_test_shapes[test_name]
    test_shapes = default_test_shapes[test_name]
    return [pytest.param(s, id="x".join(map(str, s))) for s in test_shapes]


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@param_no_scheduling
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testChainedGemm(
    shape: tuple[int],
    enable_scheduling: bool,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
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
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, M: j, N: k}, outputs={B: i, N: k, M: j}
    )

    @tkw.wave(constraints)
    def chained_gemm(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, N, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K2, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32]
        ) -> tkl.Register[B, M, N, tkl.f32]:
            inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
            qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
            qk_cast_reg = tkw.cast(qk_reg, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(qk_cast_reg, v_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(
            repeat, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    b, m, n, k1, k2 = shape
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        B: b,
        M: m,
        N: n,
        K1: k1,
        K2: k2,
    }
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

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        torch.manual_seed(0)
        q = device_randn(b, m, k1, dtype=torch.float16)
        k = device_randn(b, k2, k1, dtype=torch.float16)
        v = device_randn(b, n, k2, dtype=torch.float16)
        output = device_zeros(b, n, m, dtype=torch.float32)
        mb = chained_gemm(q, k, v, output)

        if test_dump_generated_mlir:
            filename = f"wave_cgemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        iree_ref = torch.zeros(b, n, m, dtype=torch.float32)
        generate_iree_ref(
            "chain_mmt", [q, k, v], [iree_ref], config, run_bench=run_bench
        )
        assert_close(output, iree_ref, check_device=False)

        # print(f"{q.shape=}\n{k.shape=}\n{v.shape=}\n{output.shape=}")
        torch_qk = torch.matmul(q, k.transpose(-1, -2))  # m x k1 @ k1 x k2 -> m x k2
        # print(f"{torch_qk.shape=}")
        torch_ref = torch.matmul(
            torch_qk, v.transpose(-1, -2)
        )  # m x k2 @ k2 x n -> m x n
        output_for_cmp = output.transpose(-1, -2).to(torch.float16)
        assert_close(output_for_cmp, torch_ref, atol=5e-2, rtol=1e-2)


# This test requires some more analysis on the index sequences between
# the two chained GEMMs.
@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@param_no_scheduling
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8),
        (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8),
    ],
)
def testChainedGemmF8(
    shape: tuple[int], enable_scheduling: bool, mfma_variant: tuple[MMAType], request
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
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
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant[0],
            vector_shapes={B: 0},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, M: j, N: k}, outputs={B: i, N: k, M: j}
    )

    @tkw.wave(constraints)
    def chained_gemm_f8(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, N, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K2, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32]
        ) -> tkl.Register[B, M, N, tkl.f32]:
            inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            q_reg = tkw.cast(q_reg, tkl.f8e4m3fnuz)
            k_reg = tkw.cast(k_reg, tkl.f8e4m3fnuz)
            kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
            qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
            qk_cast_reg = tkw.cast(qk_reg, tkl.f8e4m3fnuz)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            v_reg = tkw.cast(v_reg, tkl.f8e4m3fnuz)
            acc = tkw.mma(qk_cast_reg, v_reg, acc, mfma_variant[1])
            return acc

        # repeat represents the results of the loop
        tkw.write(
            repeat, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant[0]),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
    }
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

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        torch.manual_seed(0)
        q = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = device_randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = device_randn(shape[0], shape[2], shape[4], dtype=torch.float16)
        output = device_zeros(shape[0], shape[2], shape[1], dtype=torch.float32)
        mb = chained_gemm_f8(q, k, v, output)

        if test_dump_generated_mlir:
            filename = f"wave_cgemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        iree_ref = torch.zeros(shape[0], shape[2], shape[1], dtype=torch.float32)
        generate_iree_ref(
            "chain_mmt_f8", [q, k, v], [iree_ref], config, run_bench=run_bench
        )
        assert_close(output, iree_ref, atol=7e-5, rtol=2e-3, check_device=False)


def testFlashAttention2SimpleFwdBwd(
    # shape: tuple[int],
    # enable_scheduling: bool,
    # dynamic_dims: bool,
    # mfma_variant: MMAType,
    # request,
):
    mfma_variant = MMAType.F32_16x16x16_F16
    dynamic_dims = False
    batch = 8
    seq_len = 128
    head_dim = 64
    # run_bench = request.config.getoption("--runperf")
    # dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    B = tkl.sym.B  # batch
    # Even if the sequence lengths are the same size, we still need different
    # symbols for their dimensions because they are tiled separately.
    Nq = tkl.sym.Nq  # query sequence length
    Nkv = tkl.sym.Nkv  # key/value sequence length
    Dv = tkl.sym.Dv  # head dimension
    Dqk = tkl.sym.Dqk

    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_Nq = tkl.sym.BLOCK_Nq
    BLOCK_Dv = tkl.sym.BLOCK_Dv
    BLOCK_Nkv = tkl.sym.BLOCK_Nkv
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(Nq, BLOCK_Nq, 0)]
    constraints += [tkw.WorkgroupConstraint(Dv, BLOCK_Dv, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(Nkv, BLOCK_Nkv)]
    constraints += [tkw.WaveConstraint(Nq, BLOCK_Nq / 4)]
    constraints += [tkw.WaveConstraint(Dv, BLOCK_Dv / 1)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Nq_vec = 16
        Dv_vec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Nq_vec = 32
        Dv_vec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, Nq: Nq_vec, Dv: Dv_vec},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, Dv: j, Nkv: k}, outputs={B: i, Nkv: k, Dv: j}
    )

    # Name : Flash2 : Wave
    # Query sequence length : M : M
    # KV sequence length : N : K2
    # Head dimension : D : K1/N

    @tkw.wave(constraints)
    def base_attention(
        Q: tkl.Memory[B, Nq, Dqk, GLOBAL_ADDRESS_SPACE, tkl.f16],
        KT: tkl.Memory[B, Dqk, Nkv, ADDRESS_SPACE, tkl.f16],
        V: tkl.Memory[B, Nkv, Dv, ADDRESS_SPACE, tkl.f16],
        O: tkl.Memory[B, Nq, Dv, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        init_O = tkl.Register[B, Nq, Dv, tkl.f32](0.0)
        init_l = tkl.Register[B, Nq, tkl.f32](0.0)
        init_m = tkl.Register[B, Nq, tkl.f32](-1e6)

        @tkw.reduction(Nkv, init_args=[init_m, init_l, init_O])
        def repeat(
            m_jm1: tkl.Register[B, Nq, tkl.f32],
            l_jm1: tkl.Register[B, Nq, tkl.f32],
            O_jm1: tkl.Register[B, Nq, Dv, tkl.f32],
        ) -> (
            tkl.Register[B, Nq, tkl.f32],
            tkl.Register[B, Nq, tkl.f32],
            tkl.Register[B, Nq, Dv, tkl.f32],
        ):
            S_acc = tkl.Register[B, Nq, Nkv, tkl.f32](0.0)
            Q_reg = tkw.read(
                Q, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )  # BxNqxDqk f16
            KT_reg = tkw.read(
                KT, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )  # BxDqkxNkv f16
            S_j = tkw.mma(Q_reg, KT_reg, S_acc)  # BxNqxNkv f32
            m_j = tkw.max(S_j, m_jm1, dim=Nkv)  # BxNq f32
            # TODO: figure out why this has to be explicit
            m_j_broadcast = tkw.broadcast(
                m_j, target_shape=[B, Nq, Nkv]
            )  # BxNqxNkv f32
            e_delta_max = tkw.exp2(m_jm1 - m_j)  # BxNq f32
            P_j = tkw.exp2(S_j - m_j_broadcast)  # BxNqxNkv f32
            l_init = l_jm1 * e_delta_max  # BxNq f32
            l_j = tkw.sum(P_j, l_init, dim=Nkv)  # BxNq f32
            P_j_f16 = tkw.cast(P_j, tkl.f16)  # BxNqxNkv f16
            V_reg = tkw.read(
                V, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )  # BxNkvxDv f16
            O_init = O_jm1 * e_delta_max  # BxNqxDv f32
            O_j = tkw.mma(P_j_f16, V_reg, O_init)  # BxNqxDv f32
            return m_j, l_j, O_j

        # repeat represents the results of the loop
        m_i, l_i, O_i = repeat
        reciprocal_l_i = tkw.reciprocal(l_i)
        res = O_i * reciprocal_l_i
        res_O_casted = tkw.cast(res, tkl.f16)
        tkw.write(res_O_casted, O, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_Nq: 128,
        BLOCK_Nkv: 64,
        BLOCK_Dv: 64,
        B: batch,
        Nq: seq_len,
        Nkv: seq_len,
        Dv: head_dim,
    }
    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    # if run_bench:
    #     config["benchmark_batch_size"] = 10
    #     config["benchmark_repetitions"] = 3
    # if dump_perf is not None:
    #     perf_filename = request.node.name + ".json"
    #     config["benchmark_results_file"] = os.path.join(
    #         dump_perf, "tk_" + perf_filename
    #     )
    compile_config = {"waves_per_eu": 2, "denorm_fp_math_f32": "preserve-sign"}

    dynamic_symbols = []
    dynamic_symbols_map = {}
    # if dynamic_dims:
    #     dynamic_symbols_map[M] = hyperparams[M]
    #     dynamic_symbols_map[D] = hyperparams[D]
    #     dynamic_symbols_map[B] = hyperparams[B]
    #     dynamic_symbols_map[M] = hyperparams[M]
    #     dynamic_symbols.append(M)
    #     dynamic_symbols.append(D)
    #     dynamic_symbols.append(B)
    #     dynamic_symbols.append(M)
    #     del hyperparams[M]
    #     del hyperparams[D]
    #     del hyperparams[B]
    #     del hyperparams[M]

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        compile_config=compile_config,
        schedule=False,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        torch.manual_seed(0)
        q = device_randn(batch, seq_len, head_dim, dtype=torch.float16)
        k = device_randn(batch, seq_len, head_dim, dtype=torch.float16)
        v = device_randn(batch, seq_len, head_dim, dtype=torch.float16)
        output = device_zeros(batch, seq_len, head_dim, dtype=torch.float32)
        log2e = 1.44269504089
        dk_sqrt = math.sqrt(1.0 / head_dim)
        mb = base_attention(q * dk_sqrt * log2e, k.transpose(-2, -1), v, output)
        torch_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None
        )

        # if test_dump_generated_mlir:
        #     filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
        #     with open(filename, "w") as f:
        #         f.write(mb.module_op.get_asm())

        assert_close(output, torch_ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@param_scheduling
@param_dynamic_dims
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testAttentionPaperSymbols(
    shape: tuple[int],
    enable_scheduling: bool,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")

    # Input sizes
    B = tkl.sym.B  # batch
    Nq = tkl.sym.Nq  # query sequence length
    Dv = tkl.sym.Dv  # value head dimension
    Dqk = tkl.sym.Dqk  # query/key head dimension
    Nkv = tkl.sym.Nkv  # key/value sequence length
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_Nq = tkl.sym.BLOCK_Nq
    BLOCK_Dv = tkl.sym.BLOCK_Dv
    BLOCK_Nkv = tkl.sym.BLOCK_Nkv
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(Nq, BLOCK_Nq, 0)]
    constraints += [tkw.WorkgroupConstraint(Dv, BLOCK_Dv, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(Nkv, BLOCK_Nkv)]
    constraints += [tkw.WaveConstraint(Nq, BLOCK_Nq / 4)]
    constraints += [tkw.WaveConstraint(Dv, BLOCK_Dv / 1)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Nq_vec = 16
        Dv_vec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Nq_vec = 32
        Dv_vec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, Nq: Nq_vec, Dv: Dv_vec},
        )
    ]

    if dynamic_dims:
        constraints += [tkw.Assumption(Nkv > BLOCK_Nkv * 4)]

    # i = tkw.IndexMapping.iterator(0)
    # j = tkw.IndexMapping.iterator(1)
    # k = tkw.IndexMapping.iterator(2)
    # mapping = tkw.IndexMapping(
    #     num_iterators=3, inputs={B: i, Dv: j, Nq: k}, outputs={B: i, Nq: k, Dv: j}
    # )

    @tkw.wave(constraints)
    def base_attention(
        Q: tkl.Memory[B, Nq, Dqk, GLOBAL_ADDRESS_SPACE, tkl.f16],
        K: tkl.Memory[B, Dqk, Nkv, ADDRESS_SPACE, tkl.f16],
        V: tkl.Memory[B, Dv, Nkv, ADDRESS_SPACE, tkl.f16],
        O: tkl.Memory[B, Nq, Dv, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        init_m = tkl.Register[B, Nq, tkl.f32](-1e6)
        init_l = tkl.Register[B, Nq, tkl.f32](0.0)
        init_O = tkl.Register[B, Nq, Dv, tkl.f32](0.0)

        @tkw.reduction(
            Nkv,
            init_args=[
                init_m,
                init_l,
                init_O,
            ],
        )
        def repeat(
            m_jm1: tkl.Register[B, Nq, tkl.f32],
            l_jm1: tkl.Register[B, Nq, tkl.f32],
            O_jm1: tkl.Register[B, Nq, Dv, tkl.f32],
        ) -> (
            tkl.Register[B, Nq, tkl.f32],
            tkl.Register[B, Nq, tkl.f32],
            tkl.Register[B, Nq, Dv, tkl.f32],
        ):
            S_acc = tkl.Register[B, Nq, Nkv, tkl.f32](0.0)
            Q_reg = tkw.read(Q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            K_reg = tkw.read(K, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            S_j = tkw.mma(Q_reg, K_reg, S_acc)
            # x_j = tkw.permute(S_j, target_shape=[B, Nq, Nkv])
            # m_j = tkw.max(S_j, m_jm1, dim=Nkv)
            # e_delta_max = tkw.exp2(m_jm1 - m_j)
            # P_j = tkw.exp2(S_j - m_j)
            # l_init = e_delta_max * l_jm1
            # l_j = tkw.sum(P_j, l_init, dim=Nkv)
            # P_j_f16 = tkw.cast(P_j, tkl.f16)
            V_reg = tkw.read(V, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # O_init = e_delta_max * O_jm1
            # O_j = tkw.mma(P_j_f16, V_reg, O_init)
            O_j = tkw.mma(tkw.cast(S_j, tkl.f16), V_reg, O_jm1)
            return (
                # m_j,
                # l_j,
                O_j,
            )

        (
            m_i,
            l_i,
            O_i,
        ) = repeat
        reciprocal_l_i = tkw.reciprocal(l_i)
        res = reciprocal_l_i * O_i
        tkw.write(O_i, O, elements_per_thread=STORE_ELEMS_PER_THREAD)

    # (8, 128, 128, 64, 256)
    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_Nq: 128,
        BLOCK_Dv: 64,
        BLOCK_Nkv: 64,
        B: batch,
        Nq: q_seq_len,
        Dv: v_head_dim,
        Dqk: qk_head_dim,
        Nkv: kv_seq_len,
    }
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

    dynamic_symbols = []
    dynamic_symbols_map = {}
    if dynamic_dims:
        dynamic_symbols_map[Nq] = hyperparams[Nq]
        dynamic_symbols_map[Dv] = hyperparams[Dv]
        dynamic_symbols_map[B] = hyperparams[B]
        dynamic_symbols_map[Nkv] = hyperparams[Nkv]
        dynamic_symbols.append(Nq)
        dynamic_symbols.append(Dv)
        dynamic_symbols.append(B)
        dynamic_symbols.append(Nkv)
        del hyperparams[Nq]
        del hyperparams[Dv]
        del hyperparams[B]
        del hyperparams[Nkv]

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        compile_config=compile_config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        torch.manual_seed(0)
        q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16)
        k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16)
        v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16)
        output = device_zeros(batch, q_seq_len, v_head_dim, dtype=torch.float32)
        log2e = 1.44269504089
        dk_sqrt = math.sqrt(1.0 / qk_head_dim)
        # TODO: Add scaling of QK as part of kernel.
        # TODO: Add variant of non-transposed V attention kernel.
        mb = base_attention(
            q * dk_sqrt * log2e, k.transpose(-1, -2), v.transpose(-1, -2), output
        )
        torch_ref = torch.matmul(
            torch.matmul(q * dk_sqrt * log2e, k.transpose(-1, -2)), v
        )

        # torch_ref = torch.nn.functional.scaled_dot_product_attention(
        #     q, k, v, attn_mask=None
        # )

        if test_dump_generated_mlir:
            filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        assert_close(output, torch_ref)


def print_small_tensor(t, name=""):
    print_limit = 512
    if t.shape.numel() > print_limit:
        print(f"eliding tensor larger than {print_limit} elements")
        return
    shape = "x".join([str(i) for i in t.shape])

    abs = torch.abs(t)
    # max = torch.max(abs)
    # min = torch.min(abs)
    median = torch.median(abs)
    print_mode = "reg" if median > 0.1 or median == 0 else "sci"
    width = max(
        len(f"{d: .3f}" if print_mode == "reg" else f"{d: .2e}")
        for d in t.flatten().tolist()
    )

    print(f"{name}[{shape}]")

    if len(t.shape) > 2:
        t = t.squeeze()
    if len(t.shape) < 2:
        t = t.unsqueeze(0)

    for row in t:
        print(
            " ".join(
                f"{d: {width}.3f}" if print_mode == "reg" else f"{d: {width}.2e}"
                for d in row.tolist()
            )
        )


def attention_fwd_loops(q, k, v, o, lse):
    B, M_qs, K1_qkd = q.shape
    _, K2_kvs, _ = k.shape
    assert k.shape == (B, K2_kvs, K1_qkd)
    _, _, N_vd = v.shape
    assert v.shape == (B, K2_kvs, N_vd)
    assert o.shape == (B, M_qs, N_vd)
    assert lse.shape == (B, M_qs)

    # ensure we have different shapes so we get errors if there are mismatches.
    if M_qs == K2_kvs:
        BLOCK_M_Br = M_qs // 4
        BLOCK_K2_Bc = K2_kvs // 2
    else:
        BLOCK_M_Br = M_qs // 2
        BLOCK_K2_Bc = K2_kvs // 2

    assert M_qs % BLOCK_M_Br == 0
    assert K2_kvs % BLOCK_K2_Bc == 0

    for batch in range(B):
        for start_m in range(0, M_qs, BLOCK_M_Br):
            q_i = q[batch, start_m : start_m + BLOCK_M_Br, :]
            assert q_i.shape == (BLOCK_M_Br, K1_qkd)
            o_i = device_zeros(BLOCK_M_Br, N_vd)
            l_i = device_zeros(BLOCK_M_Br)
            m_i = to_default_device(torch.full((BLOCK_M_Br,), -torch.inf))
            for start_k2 in range(0, K2_kvs, BLOCK_K2_Bc):
                k_j = k[batch, start_k2 : start_k2 + BLOCK_K2_Bc, :]
                assert k_j.shape == (BLOCK_K2_Bc, K1_qkd)
                v_j = v[batch, start_k2 : start_k2 + BLOCK_K2_Bc, :]
                assert v_j.shape == (BLOCK_K2_Bc, N_vd)
                s_ij = torch.matmul(q_i, k_j.transpose(-1, -2))
                assert s_ij.shape == (BLOCK_M_Br, BLOCK_K2_Bc)
                m_ij = torch.maximum(m_i, torch.max(s_ij, dim=-1)[0])
                assert m_ij.shape == (BLOCK_M_Br,)
                p_ij = torch.exp(s_ij - torch.reshape(m_ij, (BLOCK_M_Br, 1)))
                assert p_ij.shape == (BLOCK_M_Br, BLOCK_K2_Bc)
                e_delta_max = torch.exp(m_i - m_ij)
                l_ij = e_delta_max * l_i + torch.sum(p_ij, dim=-1)
                assert l_ij.shape == (BLOCK_M_Br,)
                o_ij = e_delta_max.reshape(BLOCK_M_Br, 1) * o_i + torch.matmul(
                    p_ij, v_j
                )
                assert o_ij.shape == (BLOCK_M_Br, N_vd)

                o_i = o_ij
                m_i = m_ij
                l_i = l_ij

            o_i = o_i / l_i.reshape(BLOCK_M_Br, 1)
            L_i = m_i + torch.log(l_i)
            o[batch, start_m : start_m + BLOCK_M_Br, :] = o_i
            lse[batch, start_m : start_m + BLOCK_M_Br] = L_i


def attention_bwd_loops(
    q,
    k,
    v,
    do,
    lse,
    D,
    dq,
    dk,
    dv,
):
    B, M_qs, K1_qkd = q.shape
    _, K2_kvs, _ = k.shape
    assert k.shape == (B, K2_kvs, K1_qkd)
    _, _, N_vd = v.shape
    assert v.shape == (B, K2_kvs, N_vd)
    assert do.shape == (B, M_qs, N_vd)
    assert lse.shape == (B, M_qs)
    assert D.shape == (B, M_qs)
    assert dq.shape == (B, M_qs, K1_qkd)
    assert dk.shape == (B, K2_kvs, K1_qkd)
    assert dv.shape == (B, K2_kvs, N_vd)

    # ensure we have different shapes so we get errors if there are mismatches.
    # if M_qs == K2_kvs:
    #     BLOCK_M_Br = M_qs // 4
    #     BLOCK_K2_Bc = K2_kvs // 2
    # else:
    #     BLOCK_M_Br = M_qs // 2
    #     BLOCK_K2_Bc = K2_kvs // 2
    BLOCK_M_Br = 8
    BLOCK_K2_Bc = 4

    assert M_qs % BLOCK_M_Br == 0
    assert K2_kvs % BLOCK_K2_Bc == 0

    for batch in range(B):
        for start_k2 in range(0, K2_kvs, BLOCK_K2_Bc):
            k_j = k[batch, start_k2 : start_k2 + BLOCK_K2_Bc, :]
            assert k_j.shape == (BLOCK_K2_Bc, K1_qkd)
            v_j = v[batch, start_k2 : start_k2 + BLOCK_K2_Bc, :]
            assert v_j.shape == (BLOCK_K2_Bc, N_vd)

            dv_j = torch.zeros(BLOCK_K2_Bc, N_vd, device=q.device)
            dk_j = torch.zeros(BLOCK_K2_Bc, K1_qkd, device=q.device)

            for start_m in range(0, M_qs, BLOCK_M_Br):
                q_i = q[batch, start_m : start_m + BLOCK_M_Br, :]
                assert q_i.shape == (BLOCK_M_Br, K1_qkd)
                do_i = do[batch, start_m : start_m + BLOCK_M_Br, :]
                assert do_i.shape == (BLOCK_M_Br, N_vd)
                dq_i = dq[batch, start_m : start_m + BLOCK_M_Br, :]
                assert dq_i.shape == (BLOCK_M_Br, K1_qkd)
                lse_i = lse[batch, start_m : start_m + BLOCK_M_Br]
                assert lse_i.shape == (BLOCK_M_Br,)
                D_i = D[batch, start_m : start_m + BLOCK_M_Br]
                assert D_i.shape == (BLOCK_M_Br,)
                s_ij = torch.matmul(q_i, k_j.transpose(-1, -2))
                assert s_ij.shape == (BLOCK_M_Br, BLOCK_K2_Bc)
                p_ij = torch.exp(s_ij - lse_i.reshape(BLOCK_M_Br, 1))
                assert p_ij.shape == (BLOCK_M_Br, BLOCK_K2_Bc)
                dv_j += torch.matmul(p_ij.transpose(-1, -2), do_i)
                dp_ij = torch.matmul(do_i, v_j.transpose(-1, -2))
                assert dp_ij.shape == (BLOCK_M_Br, BLOCK_K2_Bc)
                ds_ij = p_ij * (dp_ij - D_i.reshape(BLOCK_M_Br, 1))
                assert ds_ij.shape == (BLOCK_M_Br, BLOCK_K2_Bc)
                dq_i += torch.matmul(ds_ij, k_j)
                dk_j += torch.matmul(ds_ij.transpose(-1, -2), q_i)

                dv[batch, start_k2 : start_k2 + BLOCK_K2_Bc, :] = dv_j
                # if batch == 1 and start_k2 == K2_kvs - BLOCK_K2_Bc:
                #     print_small_tensor(dk_j, f"dk[1][-1]")

            dk[batch, start_k2 : start_k2 + BLOCK_K2_Bc, :] = dk_j


def get_attention_fwd_kernel(
    batch: int,
    kv_seq_len: int,
    qk_head_dim: int,
    q_seq_len: int,
    v_head_dim: int,
    mfma_variant: MMAType,
):
    # Input sizes
    B = tkl.sym.B  # batch
    M_qs = tkl.sym.M  # query sequence length
    N_vd = tkl.sym.N  # value head dimension
    K1_qkd = tkl.sym.K1  # query/key head dimension
    K2_kvs = tkl.sym.K2  # key/value sequence length

    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M_r_qs = tkl.sym.BLOCK_M
    BLOCK_N_vd = tkl.sym.BLOCK_N
    BLOCK_K2_c_kvs = tkl.sym.BLOCK_K2
    # TODO: why can't this be a symbol?
    WAVES_PER_BLOCK_M = 1
    WAVES_PER_BLOCK_N = 1
    WAVES_PER_BLOCK_B = 1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M_qs, BLOCK_M_r_qs, 0)]
    constraints += [tkw.WorkgroupConstraint(N_vd, BLOCK_N_vd, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2_kvs, BLOCK_K2_c_kvs)]
    constraints += [tkw.WaveConstraint(M_qs, BLOCK_M_r_qs / WAVES_PER_BLOCK_M)]
    constraints += [tkw.WaveConstraint(N_vd, BLOCK_N_vd / WAVES_PER_BLOCK_N)]
    constraints += [tkw.WaveConstraint(B, BLOCK_B / WAVES_PER_BLOCK_B)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(WAVES_PER_BLOCK_M, WAVES_PER_BLOCK_N, WAVES_PER_BLOCK_B),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M_qs: Mvec, N_vd: Nvec},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    o_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N_vd: j, M_qs: k}, outputs={B: i, M_qs: k, N_vd: j}
    )

    s_mapping = tkw.IndexMapping(num_iterators=3, inputs={B: i, K2_kvs: j, M_qs: k}, outputs={B: i, M_qs: k, K2_kvs: j})
    

    # TODO: tune address space
    @tkw.wave(constraints)
    def attention_fwd(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N_vd, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        o: tkl.Memory[B, M_qs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        lse: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        init_o = tkl.Register[B, N_vd, M_qs, tkl.f32](0.0)
        init_l = tkl.Register[B, M_qs, tkl.f32](0.0)
        init_m = tkl.Register[B, M_qs, tkl.f32](-1e6)

        @tkw.reduction(K2_kvs, init_args=[init_m, init_l, init_o])
        def repeat(
            m_prev: tkl.Register[B, M_qs, tkl.f32],
            l_prev: tkl.Register[B, M_qs, tkl.f32],
            o_prev: tkl.Register[B, N_vd, M_qs, tkl.f32],
            # Having this get passed through every time appears to be the only
            # way to make it accessible inside the loop but have the read be
            # outside, which I think is clearer since it isn't dependent on the
            # induction variable. Unsure what it does to the codegen.
            # Unfortunately, it creates a compilation failure except when
            # there's only one block.
            # q_i: tkl.Register[B, M, K1, tkl.f16],
        ) -> tuple[
            tkl.Register[B, M_qs, tkl.f32],
            tkl.Register[B, M_qs, tkl.f32],
            tkl.Register[B, N_vd, M_qs, tkl.f32],
            # tkl.Register[B, M, K1, tkl.f16],
        ]:
            s_acc = tkl.Register[B, K2_kvs, M_qs, tkl.f32](0.0)
            log2e = tkl.Register[B, M_qs, tkl.f32](1.44269504089)
            q_i = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_j = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # I don't really understand why we have to compute the transpose and
            # then transpose it, but apparently we do, otherwise the compiler
            # complains in set_node_indices about being unable to resolve thread
            # shape discrepancies when neither of the shapes is 1.
            s_ij = tkw.mma(k_j, q_i, s_acc)
            tkw.write(s_ij, s, mapping=s_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)
            s_ij = tkw.permute(s_ij, target_shape=[B, M_qs, K2_kvs])
            m_ij = tkw.max(s_ij, m_prev, dim=K2_kvs)
            e_delta_max = tkw.exp2(log2e * (m_prev - m_ij))
            p_ij = tkw.exp2(log2e * (s_ij - m_ij))
            l_init = e_delta_max * l_prev
            l_ij = tkw.sum(p_ij, l_init, dim=K2_kvs)
            v_j = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            o_init = e_delta_max * o_prev
            o_ij = tkw.mma(v_j, tkw.cast(p_ij, tkl.f16), o_init)
            return m_ij, l_ij, o_ij  # , q_i

        log2e = tkl.Register[B, M_qs, tkl.f32](1.44269504089)
        m_i, l_i, o_i = repeat
        o_i = tkw.reciprocal(l_i) * o_i
        tkw.write(
            tkw.cast(o_i, tkl.f16),
            o,
            mapping=o_mapping,
            elements_per_thread=STORE_ELEMS_PER_THREAD,
        )
        lse_i = m_i + (tkw.log2(l_i) / log2e)
        tkw.write(tkw.cast(lse_i, tkl.f16), lse, elements_per_thread=1)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M_r_qs: 16,
        BLOCK_N_vd: 16,
        BLOCK_K2_c_kvs: 16,
        B: batch,
        M_qs: q_seq_len,
        N_vd: v_head_dim,
        K1_qkd: qk_head_dim,
        K2_kvs: kv_seq_len,
    }

    return attention_fwd, hyperparams


def get_attention_bwd_kernel(
    batch: int,
    kv_seq_len: int,
    qk_head_dim: int,
    q_seq_len: int,
    v_head_dim: int,
    mfma_variant: MMAType,
):
    # Input sizes
    B = tkl.sym.B  # batch
    M_qs = tkl.sym.M  # query sequence length
    N_vd = tkl.sym.N  # value head dimension
    K1_qkd = tkl.sym.K1  # query/key head dimension
    K2_kvs = tkl.sym.K2  # key/value sequence length

    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_K1 = tkl.sym.BLOCK_K1
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # TODO: why can't this be a symbol?
    WAVES_PER_BLOCK_M = 1
    WAVES_PER_BLOCK_K1 = 1
    WAVES_PER_BLOCK_B = 1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(B, BLOCK_B, 0)]
    constraints += [tkw.WorkgroupConstraint(K2_kvs, BLOCK_K2, 1)]
    # constraints += [tkw.WorkgroupConstraint(K1, BLOCK_K1, 1)]
    constraints += [tkw.TilingConstraint(M_qs, BLOCK_M)]
    # constraints += [tkw.WaveConstraint(M, BLOCK_M / WAVES_PER_BLOCK_M)]
    # constraints += [tkw.WaveConstraint(K1, BLOCK_K1 / WAVES_PER_BLOCK_K1)]
    # constraints += [tkw.WaveConstraint(B, BLOCK_B / WAVES_PER_BLOCK_B)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
        K1vec = 16
        K2vec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32
        K1vec = 32
        K2vec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(WAVES_PER_BLOCK_M, WAVES_PER_BLOCK_K1, WAVES_PER_BLOCK_B),
            mma_type=mfma_variant,
            vector_shapes={B: 0}, #, M_qs: Mvec, N_vd: Nvec, K1_qkd: K1vec, K2_kvs: K2vec},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_k2_m_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, K2_kvs: j, M_qs: k}, outputs={B: i, M_qs: k, K2_kvs: j}
    )
    flip_n_m_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N_vd: k, M_qs: j}, outputs={B: i, M_qs: j, N_vd: k}
    )

    # p_reload_mapping = tkw.IndexMapping(num_iterators=3, inputs={B: i, K2_kvs: j, M_qs

    @tkw.wave(constraints)
    def attention_bwd(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        do: tkl.Memory[B, N_vd, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # This has to be loaded in this way or set_node_indices fails with resolving thread shapes
        lse: tkl.Memory[B, K2_kvs, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        D: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # dq: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # dk: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dv: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        p: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        ds: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dp: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        dp_sub: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # dp_sub_ref: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # p_ref: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        dv_init = tkl.Register[B, K2_kvs, N_vd, tkl.f32](0.0)
        # dk_init = tkl.Register[B, K2_kvs, K1_qkd, tkl.f32](0.0)

        @tkw.reduction(M_qs, init_args=[
            dv_init,
            # dk_init,
        ])
        def loop_q_seq_len(
            dv_prev: tkl.Register[B, K2_kvs, N_vd, tkl.f32],
            # dk_prev: tkl.Register[B, K2_kvs, K1_qkd, tkl.f32],
        ) -> tkl.Register[B, K2_kvs, N_vd, tkl.f32]:
        #     # tkl.Register[B, K2_kvs, K1_qkd, tkl.f32],
        # :
            k_j = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)  # B x K2 x K1
            q_i = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)  # B x M  x K1

            s_acc = tkl.Register[B, M_qs, K2_kvs, tkl.f32](0.0)  # B x K2 x M
            log2e = tkl.Register[B, M_qs, K2_kvs, tkl.f16](1.44269504089)  # B x M  x K2
            s_ij = tkw.mma(q_i, k_j, s_acc)  # B x K2 x M
            # TODO: Only half the elements get written if K2 is 32 here
            tkw.write(s_ij, s, elements_per_thread=STORE_ELEMS_PER_THREAD)
            # s_ij = tkw.permute(s_ij, [B, M_qs, K2_kvs])
            s_ij = tkw.permute(s_ij, [B, K2_kvs, M_qs])
            lse_i = tkw.read(lse, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            p_ij = tkw.exp2(log2e * (tkw.cast(s_ij, tkl.f16) - lse_i))
            tkw.write(p_ij, p, mapping=flip_k2_m_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

            do_i_for_dv = tkw.read(do, elements_per_thread=LOAD_ELEMS_PER_THREAD)  # B x M  x N
            dv_j = tkw.mma(p_ij, do_i_for_dv, dv_prev)  # B x M x K2 @ B x M x N -> B x
            
            v_j = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)  # B x K2 x N
            # This has to be loaded separately so N is the fastest-varying dimension
            do_i_for_dp = tkw.read(do, mapping=flip_n_m_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            dp_acc = tkl.Register[B, K2_kvs, M_qs, tkl.f32](0.0)  # B x M  x K2
            dp_ij = tkw.mma(v_j, do_i_for_dp, dp_acc)  # B x M x K2
            dp_ij = tkw.permute(dp_ij, [B, M_qs, K2_kvs])
            tkw.write(dp_ij, dp, elements_per_thread=STORE_ELEMS_PER_THREAD)

            D_i = tkw.read(D, elements_per_thread=1)  # B x M
            dp_ij_sub = tkw.cast(dp_ij, tkl.f16) - tkw.broadcast(D_i, [B, M_qs, K2_kvs])
            tkw.write(dp_ij_sub, dp_sub, elements_per_thread=STORE_ELEMS_PER_THREAD)
            
            # Just multiplying p_ij * dp_ij_sub breaks the previously calculated
            # dp. Trying to write dp_sub and then read it back again just
            # results in ds being mostly zero (race condition?). Trying to pass
            # in the reference dp_sub doesn't fix it though. It actually looks
            # like it's just really bad numerics maybe?
            # But loading back p works, so probably this is another shape/layout thing I don't understand.
            # dp_ij_sub_for_ds = dp_ij_sub # tkw.read(dp_sub_ref, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            
            # We have to read this back again :-(
            p_ij_for_ds = tkw.read(p, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            ds_ij = p_ij_for_ds * dp_ij_sub  # B x M  x K2
            tkw.write(ds_ij, ds, elements_per_thread=STORE_ELEMS_PER_THREAD)
            # dk_j = tkw.mma(
            #     tkw.permute(ds_ij, [B, K2_kvs, M_qs]), tkw.permute(q_i, [B, K1_qkd, M_qs]), dk_prev
            # )
            # dq_prev = tkw.read(
            #     dq, elements_per_thread=LOAD_ELEMS_PER_THREAD
            # )  # B x M  x K1
            # dq_i = tkw.mma(
            #     ds_ij, tkw.permute(k_j, [B, K1_qkd, K2_kvs]), tkw.cast(dq_prev, tkl.f32)
            # )  # B x M  x K1
            # tkw.write(dq_i, dq, elements_per_thread=STORE_ELEMS_PER_THREAD)
            return dv_j #, dk_j

        (
            dv_j
            # dk_j,
        ) = loop_q_seq_len
        tkw.write(
            tkw.cast(dv_j, tkl.f16), dv, elements_per_thread=STORE_ELEMS_PER_THREAD
        )
        # tkw.write(
        #     tkw.cast(dk_j, tkl.f16), dk, elements_per_thread=STORE_ELEMS_PER_THREAD
        # )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 16,
        # TODO: Not actually what we want probably, but I couldn't get nested
        # loops to work (the only option is a reduction, it insists on having
        # args to reduce, and then it can't figure out the vector shapes for
        # them if I give it dummy args) and I don't think our read/write scheme
        # for dq is thread safe. So force the tiling of K2 to be degenerate.
        BLOCK_K2: kv_seq_len,
        B: batch,
        M_qs: q_seq_len,
        N_vd: v_head_dim,
        K1_qkd: qk_head_dim,
        K2_kvs: kv_seq_len,
    }

    return attention_bwd, hyperparams



def get_attention_bwd_single_block_kernel(
    batch: int,
    kv_seq_len: int,
    qk_head_dim: int,
    q_seq_len: int,
    v_head_dim: int,
    mfma_variant: MMAType,
):
    # Input sizes
    B = tkl.sym.B  # batch
    M_qs = tkl.sym.M  # query sequence length
    N_vd = tkl.sym.N  # value head dimension
    K1_qkd = tkl.sym.K1  # query/key head dimension
    K2_kvs = tkl.sym.K2  # key/value sequence length

    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # TODO: why can't this be a symbol?
    WAVES_PER_BLOCK_M = 1
    WAVES_PER_BLOCK_K1 = 1
    WAVES_PER_BLOCK_B = 1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(B, BLOCK_B, 0)]
    # constraints += [tkw.WorkgroupConstraint(K2_kvs, BLOCK_K2, 1)]
    # constraints += [tkw.WorkgroupConstraint(K1, BLOCK_K1, 1)]
    # constraints += [tkw.TilingConstraint(M_qs, BLOCK_M)]
    # constraints += [tkw.WaveConstraint(M, BLOCK_M / WAVES_PER_BLOCK_M)]
    # constraints += [tkw.WaveConstraint(K1, BLOCK_K1 / WAVES_PER_BLOCK_K1)]
    # constraints += [tkw.WaveConstraint(B, BLOCK_B / WAVES_PER_BLOCK_B)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
        K1vec = 16
        K2vec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32
        K1vec = 32
        K2vec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0}, #, M_qs: Mvec, N_vd: Nvec, K1_qkd: K1vec, K2_kvs: K2vec},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_seqs_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, K2_kvs: j, M_qs: k}, outputs={B: i, M_qs: k, K2_kvs: j}
    )

    # p_reload_mapping = tkw.IndexMapping(num_iterators=3, inputs={B: i, K2_kvs: j, M_qs

    @tkw.wave(constraints)
    def attention_bwd(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # v: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # do: tkl.Memory[B, N_vd, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        lse: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # D: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # dq: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # dk: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # dv: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        s_minus_l: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        # p: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # ds: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # dp: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        # dp_sub: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        # dv_init = tkl.Register[B, K2_kvs, N_vd, tkl.f32](0.0)
        # dk_init = tkl.Register[B, K2_kvs, K1_qkd, tkl.f32](0.0)

        q_i = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)  # B x M  x K1
        k_j = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)  # B x K2 x K1
        # v_j = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)  # B x K2 x N
        lse_i = tkw.read(lse, elements_per_thread=4)

        # s_acc = tkl.Register[B, M_qs, K2_kvs, tkl.f32](0.0)  # B x K2 x M
        # log2e = tkl.Register[B, M_qs, K2_kvs, tkl.f16](1.44269504089)  # B x M  x K2
        sml = tkw.mma(q_i, k_j, -tkw.cast(lse_i, tkl.f32))  # B x K2 x M
        # tkw.write(s_ij, s, elements_per_thread=STORE_ELEMS_PER_THREAD)
        # s_ij = tkw.permute(s_ij, [B, M_qs, K2_kvs]) # This should be entirely unnecessary
        # s_ij = tkw.permute(s_ij, [B, K2_kvs, M_qs])
        # sml = tkw.cast(s_ij, tkl.f16) - lse_i
        tkw.write(sml, s_minus_l, elements_per_thread=STORE_ELEMS_PER_THREAD)
        # p_ij = tkw.exp2(log2e * tkw.cast(sml, tkl.f16))
        # tkw.write(p_ij, p, elements_per_thread=STORE_ELEMS_PER_THREAD)

        # do_i = tkw.read(do, elements_per_thread=LOAD_ELEMS_PER_THREAD)  # B x M  x N
        # dp_acc = tkl.Register[B, K2_kvs, M_qs, tkl.f32](0.0)  # B x M  x K2
        # dp_ij = tkw.mma(v_j, do_i, dp_acc)  # B x M x K2
        # dp_ij = tkw.permute(dp_ij, [B, M_qs, K2_kvs])
        # tkw.write(dp_ij, dp, elements_per_thread=STORE_ELEMS_PER_THREAD)

        # Doing this before do_i is used for dp_ij results in the latter
        # being incorrect. Is permute operating inplace somehow?
        # dv_j = tkw.mma(p_ij, do_i, dv_prev)  # B x M x K2 @ B x M x N -> B x
        # D_i = tkw.read(D, elements_per_thread=1)  # B x M
        # dp_ij_sub = tkw.cast(dp_ij, tkl.f16) - tkw.broadcast(D_i, [B, M_qs, K2_kvs])
        # tkw.write(dp_ij_sub, dp_sub, elements_per_thread=STORE_ELEMS_PER_THREAD)
        # ds_ij = p_ij * dp_ij_sub  # B x M  x K2
        # tkw.write(ds_ij, ds, elements_per_thread=STORE_ELEMS_PER_THREAD)
        # dq_prev = tkw.read(
        #     dq, elements_per_thread=LOAD_ELEMS_PER_THREAD
        # )  # B x M  x K1
        # dq_i = tkw.mma(
        #     ds_ij, tkw.permute(k_j, [B, K1_qkd, K2_kvs]), tkw.cast(dq_prev, tkl.f32)
        # )  # B x M  x K1
        # tkw.write(dq_i, dq, elements_per_thread=STORE_ELEMS_PER_THREAD)
        # dk_j = tkw.mma(
        #     tkw.permute(ds_ij, [B, K2_kvs, M_qs]), tkw.permute(q_i, [B, K1_qkd, M_qs]), dk_prev
        # )
        # return dv_j #, dk_j

        # (
        #     dv_j
        #     # dk_j,
        # ) = loop_q_seq_len
        # tkw.write(
        #     tkw.cast(dv_j, tkl.f16), dv, elements_per_thread=STORE_ELEMS_PER_THREAD
        # )
        # tkw.write(
        #     tkw.cast(dk_j, tkl.f16), dk, elements_per_thread=STORE_ELEMS_PER_THREAD
        # )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        # BLOCK_M: 16,
        # TODO: Not actually what we want probably, but I couldn't get nested
        # loops to work (the only option is a reduction, it insists on having
        # args to reduce, and then it can't figure out the vector shapes for
        # them if I give it dummy args) and I don't think our read/write scheme
        # for dq is thread safe. So force the tiling of K2 to be degenerate.
        # BLOCK_K2: kv_seq_len,
        B: batch,
        M_qs: q_seq_len,
        N_vd: v_head_dim,
        K1_qkd: qk_head_dim,
        K2_kvs: kv_seq_len,
    }

    return attention_bwd, hyperparams



@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        # MMAType.F32_32x32x8_F16,
    ],
)
def testAttentionMine(shape: tuple[int], mfma_variant: MMAType, request):
    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape

    torch.manual_seed(0)
    # doing all this manual stuff in float32 or we lose too much precision
    # q = to_default_device(
    #     torch.arange((batch * q_seq_len * qk_head_dim), dtype=torch.float32).reshape(
    #         (batch, q_seq_len, qk_head_dim)
    #     )
    #     / 256
    # ).requires_grad_(True)
    # k = to_default_device(
    #     torch.arange((batch * kv_seq_len * qk_head_dim), dtype=torch.float32).reshape(
    #         (batch, kv_seq_len, qk_head_dim)
    #     )
    #     / 128
    # ).requires_grad_(True)
    # v = to_default_device(
    #     torch.arange((batch * kv_seq_len * v_head_dim), dtype=torch.float32).reshape(
    #         (batch, kv_seq_len, v_head_dim)
    #     )
    #     / 64
    # ).requires_grad_(True)

    q = device_randn(batch, q_seq_len, qk_head_dim, requires_grad=True)
    k = device_randn(batch, kv_seq_len, qk_head_dim, requires_grad=True)
    v = device_randn(batch, kv_seq_len, v_head_dim, requires_grad=True)

    # Save these to check for shenanigans later
    q_orig = q.detach().clone()
    k_orig = k.detach().clone()
    v_orig = v.detach().clone()

    o_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, scale=1
    )

    # temporarily not scaling so I don't have to figure out how to get that into the backward kernel
    dk_sqrt = math.sqrt(1.0 / qk_head_dim)

    # Fake derivative for whatever the loss function might be
    # do = to_default_device(
    #     torch.full((batch, q_seq_len, v_head_dim), 10, dtype=torch.float32)
    # )
    # do = to_default_device(
    #     torch.arange(batch*q_seq_len, dtype=torch.float32).reshape((batch, q_seq_len, 1)).expand(batch, q_seq_len, v_head_dim)
    # )
    do = device_randn(batch, q_seq_len, v_head_dim)
    do_orig = do.detach().clone()

    # IDK why we have to do retain_grad here, but the deepspeed test doesn't
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    o_ref.backward(do)

    dq_ref = q.grad.detach()
    dk_ref = k.grad.detach()
    dv_ref = v.grad.detach()

    q.grad = None
    k.grad = None
    v.grad = None

    # Manual normal attention
    s = torch.matmul(q, k.transpose(-1, -2))
    # print_small_tensor(s, "s")
    p = torch.softmax(s, dim=-1)
    # print_small_tensor(p, "p")
    o = torch.matmul(p, v)

    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    s.retain_grad()
    p.retain_grad()

    o.backward(do)
    dq = q.grad.detach()
    dk = k.grad.detach()
    dv = v.grad.detach()
    ds = s.grad.detach()
    dp = p.grad.detach()

    assert_close(o, o_ref, atol=5e-2, rtol=3e-3)
    assert_close(dq, dq_ref, atol=5e-2, rtol=3e-3)
    assert_close(dk, dk_ref, atol=5e-2, rtol=3e-3)
    assert_close(dv, dv_ref, atol=5e-2, rtol=3e-3)

    # These are presumably correct at this point and now we can use them as a
    # reference for other intermediates.
    ds_ref = ds.detach().clone()
    dp_ref = dp.detach().clone()
    # And use the manual ones of these so they're consistent with the
    # intermediate gradients.
    dq_ref = dq.detach().clone()
    dk_ref = dk.detach().clone()
    dv_ref = dv.detach().clone()
    s_ref = s.detach().clone()
    p_ref = p.detach().clone()
    o_ref = o.detach().clone()

    del ds, dp, dq, dk, dv, s, p, o

    q = q.detach().clone()
    k = k.detach().clone()
    v = v.detach().clone()

    # Manual backward pass for normal attention
    s = torch.matmul(q, k.transpose(-1, -2))
    p = torch.softmax(s, dim=-1)

    dv = torch.matmul(p.transpose(-1, -2), do)

    assert_close(dv, dv_ref, atol=5e-2, rtol=1e-2)

    dp = torch.matmul(do, v.transpose(-1, -2))

    assert_close(dp, dp_ref, atol=5e-2, rtol=1e-2)

    ds = device_zeros(batch, q_seq_len, kv_seq_len)
    for b in range(batch):
        for row in range(q_seq_len):
            p_row = p[b, row, :]
            jacobian = p_row.unsqueeze(0) * (
                to_default_device(torch.eye(kv_seq_len, dtype=torch.float16))
                - p_row.unsqueeze(-1)
            )

            ds[b, row, :] = torch.matmul(dp[b, row, :], jacobian.transpose(-1, -2))

    assert_close(ds, ds_ref, atol=5e-2, rtol=1e-2)

    dq = torch.matmul(ds, k)
    assert_close(dq, dq_ref, atol=5e-2, rtol=1e-2)

    dk = torch.matmul(ds.transpose(-1, -2), q)
    assert_close(dk, dk_ref, atol=5e-2, rtol=1e-2)

    del ds, dp, dq, dk, dv, s, p

    # Flash attention implemented as loops
    o_loops = device_zeros(batch, q_seq_len, v_head_dim)
    lse_loops = device_zeros(batch, q_seq_len)
    attention_fwd_loops(q, k, v, o_loops, lse_loops)

    # print_small_tensor(o_loops, "o_loops")
    # print_small_tensor(ref_out, "ref_out")

    assert_close(o_loops, o_ref)

    # Make sure nothing's gone wonky here.
    assert_close(q, q_orig)
    assert_close(k, k_orig)
    assert_close(v, v_orig)
    assert_close(do, do_orig)

    D_loops = torch.sum(do * o_loops, -1)
    dq_loops = torch.zeros_like(q).to("cpu")
    dk_loops = torch.zeros_like(k).to("cpu")
    dv_loops = torch.zeros_like(v).to("cpu")

    attention_bwd_loops(q.to("cpu"), k.to("cpu"), v.to("cpu"), do.to("cpu"), lse_loops.to("cpu"), D_loops.to("cpu"), dq_loops, dk_loops, dv_loops)

    # Make sure nothing's gone wonky here.
    assert_close(q, q_orig)
    assert_close(k, k_orig)
    assert_close(v, v_orig)
    assert_close(do, do_orig)

    assert_close(dq_loops, dq_ref.to("cpu"), atol=5e-2, rtol=1e-2)

    # assert_close(dk_loops[0], dk_ref[0], atol=5e-2, rtol=1e-2)
    # print_small_tensor(dk_ref[1], "dk_ref[1]")
    # print_small_tensor(dk_loops[1], "dk_loops[1]")
    # assert_close(dk_loops[1], dk_ref[1], atol=5e-2, rtol=1e-2)
    # assert_close(dk_loops[2], dk_ref[2], atol=5e-2, rtol=1e-2)

    assert_close(dk_loops, dk_ref.to("cpu"), atol=5e-2, rtol=1e-2)
    assert_close(dv_loops, dv_ref.to("cpu"), atol=5e-2, rtol=1e-2)

    # Alright, back to float16, which wave requires

    o_loops = o_loops.to(torch.float16)
    lse_loops = lse_loops.to(torch.float16)

    # s_ref and dp_ref are matrix accumulators, so still f32
    p_ref = p_ref.to(torch.float16)
    o_ref = o_ref.to(torch.float16)
    ds_ref = ds_ref.to(torch.float16)
    dq_ref = dq_ref.to(torch.float16)
    dk_ref = dk_ref.to(torch.float16)
    dv_ref = dv_ref.to(torch.float16)

    q = q.to(torch.float16)
    k = k.to(torch.float16)
    v = v.to(torch.float16)
    do = do.to(torch.float16)

    attention_fwd, hyperparams = get_attention_fwd_kernel(
        batch=batch,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        q_seq_len=q_seq_len,
        v_head_dim=v_head_dim,
        mfma_variant=mfma_variant,
    )
    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    # config["print_ir_after_all"] = True
    compile_config = {
        "waves_per_eu": 2,
        "denorm_fp_math_f32": "preserve-sign",
        # "print_ir_after": ["set_node_indices", "first", "last"],
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        compile_config=compile_config,
        schedule=False,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):

        output = device_zeros(batch, q_seq_len, v_head_dim, dtype=torch.float16)
        lse = device_zeros(batch, q_seq_len, dtype=torch.float16)
        s = device_zeros(batch, q_seq_len, kv_seq_len)
        # log2e = 1.44269504089

        mb_fwd = attention_fwd(q, k, v.transpose(-1, -2), s, output, lse)

        if test_dump_generated_mlir:
            filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_fwd.module_op.get_asm())
            print(f"IR dumped to {filename}")

        assert_close(s, s_ref, atol=5e-2, rtol=1e-2)
        # Can't check P, since we don't actually compute the "real" thing and rescale as we go.

        assert_close(output, o_ref, atol=5e-2, rtol=1e-2)
        # Use the manual loops implementation to verify the LSE part of the
        # kernel, which we can't do with the base torch implementaiton.
        assert_close(lse, lse_loops, atol=5e-2, rtol=1e-2)

    attention_bwd, hyperparams = get_attention_bwd_kernel(
        batch=batch,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        q_seq_len=q_seq_len,
        v_head_dim=v_head_dim,
        mfma_variant=mfma_variant,
    )
    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    # config["print_ir_after_all"] = True
    compile_config = {
        "waves_per_eu": 2,
        "denorm_fp_math_f32": "preserve-sign",
        # "print_ir_after": ["last", "set_node_indices"],
        # "print_ir_before": ["first", "set_node_indices"],
        # "print_signature": True,
    }

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        compile_config=compile_config,
        schedule=False,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        # TODO: maybe compute this within the kernel? The description of the
        # backward pass in the Flash Attention 2 paper has it computed external
        # to both loops though. It also seems there's a typo because it says it
        # has dimensionality of the head dimension, which doesn't make much
        # sense give that it is subtracted from dP which only has sequence
        # length dimensions. I'm assuming that's a typo (there are actually
        # quite a few in the paper, unfortunately).
        D = torch.sum(do * output, -1)

        # q:   tkl.Memory[B, M,  K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # k:   tkl.Memory[B, K2, K1, ADDRESS_SPACE,        tkl.f16],
        # v:   tkl.Memory[B, K2, N,  ADDRESS_SPACE,        tkl.f16],
        # do:  tkl.Memory[B, M,  N,  GLOBAL_ADDRESS_SPACE, tkl.f16],
        # dq:  tkl.Memory[B, M,  K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # lse: tkl.Memory[B, M,      GLOBAL_ADDRESS_SPACE, tkl.f16],
        # D:   tkl.Memory[B, M,      GLOBAL_ADDRESS_SPACE, tkl.f16],

        # print(f"{q.shape=}")
        # print(f"{k.shape=}")
        # print(f"{v.shape=}")
        # print(f"{do.shape=}")
        # print(f"{dq.shape=}")
        # print(f"{lse.shape=}")
        # print(f"{D.shape=}")

        # print(f"q:   {batch} x {q_seq_len} x  {qk_head_dim}")
        # print(f"k:   {batch} x {kv_seq_len} x {qk_head_dim}")
        # print(f"v:   {batch} x {kv_seq_len} x {v_head_dim}")
        # print(f"do:  {batch} x {q_seq_len} x  {v_head_dim}")
        # print(f"dq:  {batch} x {q_seq_len} x  {qk_head_dim}")
        # print(f"lse: {batch} x {q_seq_len}")
        # print(f"D:   {batch} x {q_seq_len}")

        if False and batch == 1 and q_seq_len * qk_head_dim <= 256:
            print_small_tensor(q, "q")
            print_small_tensor(k, "k")
            print_small_tensor(v, "v")
            print_small_tensor(do, "do")
            print_small_tensor(D, "D")
            print_small_tensor(lse, "lse")

        assert_close(q, q_orig.to(torch.float16))
        assert_close(k, k_orig.to(torch.float16))
        assert_close(v, v_orig.to(torch.float16))
        assert_close(do, do_orig.to(torch.float16))

        dp_sub_ref = (dp_ref - D.reshape((batch, q_seq_len, 1))).to(torch.float16)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        s = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
        p = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
        s_minus_l = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
        ds = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
        dp = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
        dp_sub = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)

        expanded_lse = lse.reshape((batch, q_seq_len, 1)).expand(batch, q_seq_len, kv_seq_len)

        mb_bwd = attention_bwd(
            q,
            k,
            v,
            do.transpose(-1, -2),
            expanded_lse.transpose(-1, -2),
            D,
            # dq,
            # dk,
            dv,
            s,
            # s_minus_l,
            p,
            ds,
            dp,
            dp_sub,
            # dp_sub_ref,
            # p_ref,
        )
            
        if test_dump_generated_mlir:
            filename = f"wave_attention_bwd_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_bwd.module_op.get_asm())
            print(f"IR dumped to {filename}")

        # Make sure nothing's gone wonky here.
        assert_close(q, q_orig.to(torch.float16))
        assert_close(k, k_orig.to(torch.float16))
        assert_close(v, v_orig.to(torch.float16))
        assert_close(do, do_orig.to(torch.float16))

        assert_close(s, s_ref, atol=5e-2, rtol=1e-2)
        # s_minus_l_ref = s_ref - expanded_lse
        # print_small_tensor(s_ref, "s_ref")
        # print_small_tensor(lse)
        # print_small_tensor(s_minus_l_ref, "s_minus_l_ref")
        # print_small_tensor(s_minus_l, "s_minus_l")
        # implied_lse = s_ref - s_minus_l
        # print_small_tensor(implied_lse, "implied_lse")
        # for row in implied_lse.squeeze():
        #     lse_used = []
        #     for implied_lse_entry in row.tolist():
        #         found_close = None
        #         for i, lse_entry in enumerate(lse.squeeze()):
        #             if abs(implied_lse_entry - lse_entry) <= 1e-2 + 1e-4*lse_entry:
        #                 if found_close is not None:
        #                     print(f"Found multiple matches for {implied_lse_entry}")
        #                 found_close = i
        #         if found_close is None:
        #             print(f"Didn't find a match for {implied_lse_entry}")
        #         lse_used.append(found_close)
        #     print(" ".join([f"{d:2}" for d in lse_used]))
        # assert_close(s_minus_l, s_minus_l_ref, atol=5e-2, rtol=1e-2)
        # for row in p.isclose(p_ref, atol=5e-2, rtol=1e-2).squeeze():
        #     print(" ".join(["T" if i else "F" for i in row.tolist()]))
        assert_close(p, p_ref, atol=5e-2, rtol=1e-2)
        assert_close(dv, dv_ref, atol=5e-2, rtol=1e-2)

        assert_close(dp, dp_ref, atol=5e-2, rtol=1e-2)
        assert_close(dp_sub, dp_sub_ref, atol=5e-2, rtol=1e-2)

        assert_close(ds_ref, p_ref*dp_sub_ref, atol=5e-2, rtol=1e-2) # I don't actually understand why this works
        
        # print_small_tensor(ds_ref, "ds_ref")
        # print_small_tensor(ds, "ds")

        assert_close(ds, ds_ref, atol=5e-2, rtol=1e-2)
        assert_close(dk, dk_ref, atol=5e-2, rtol=1e-2)

        # print_small_tensor(s, "s")
        # print_small_tensor(p, "p")
        # print_small_tensor(o_ref, "o_ref")
        # print_small_tensor(dp_ref, "dp_ref")
        # print_small_tensor(ds_ref, "ds_ref")
        # # print_small_tensor(dq_ref, "dq_ref")
        # print_small_tensor(dk_ref, "dk_ref")
        # # print_small_tensor(dv_ref, "dv_ref")
        # # print_small_tensor(dq, "dq")
        # print_small_tensor(dk, "dk")
        # # print_small_tensor(dv, "dv")

        assert_close(dq, dq_ref, atol=5e-2, rtol=1e-2)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testDoubleMatmul(shape: tuple[int], mfma_variant: MMAType, request):
    # Input sizes
    B = tkl.sym.B  # batch
    Nq = tkl.sym.Nq  # query sequence length
    Dv = tkl.sym.Dv  # value head dimension
    Dqk = tkl.sym.Dqk  # query/key head dimension
    Nkv = tkl.sym.Nkv  # key/value sequence length

    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_Nq = tkl.sym.BLOCK_Nq
    BLOCK_Dv = tkl.sym.BLOCK_Dv
    BLOCK_Nkv = tkl.sym.BLOCK_Nkv
    # TODO: why can't this be a symbol?
    WAVES_PER_BLOCK_Nq = 4
    WAVES_PER_BLOCK_Dv = 1
    WAVES_PER_BLOCK_B = 1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(Nq, BLOCK_Nq, 0)]
    constraints += [tkw.WorkgroupConstraint(Dv, BLOCK_Dv, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(Nkv, BLOCK_Nkv)]
    constraints += [tkw.WaveConstraint(Nq, BLOCK_Nq / WAVES_PER_BLOCK_Nq)]
    constraints += [tkw.WaveConstraint(Dv, BLOCK_Dv / WAVES_PER_BLOCK_Dv)]
    constraints += [tkw.WaveConstraint(B, BLOCK_B / WAVES_PER_BLOCK_B)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Nq_vec = 16
        Dv_vec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Nq_vec = 32
        Dv_vec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(WAVES_PER_BLOCK_Nq, WAVES_PER_BLOCK_Dv, WAVES_PER_BLOCK_B),
            mma_type=mfma_variant,
            vector_shapes={B: 0, Nq: Nq_vec, Dv: Dv_vec},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, Nq: j, Dv: k}, outputs={B: i, Dv: k, Nq: j}
    )

    @tkw.wave(constraints)
    def double_matmul(
        Q: tkl.Memory[B, Nq, Dqk, GLOBAL_ADDRESS_SPACE, tkl.f16],
        K: tkl.Memory[B, Nkv, Dqk, ADDRESS_SPACE, tkl.f16],
        V: tkl.Memory[B, Dv, Nkv, ADDRESS_SPACE, tkl.f16],
        O: tkl.Memory[B, Dv, Nq, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        init_O = tkl.Register[B, Dv, Nq, tkl.f32](0.0)

        @tkw.reduction(Nkv, init_args=[init_O])
        def repeat(
            O_jm1: tkl.Register[B, Dv, Nq, tkl.f32]
        ) -> tkl.Register[B, Dv, Nq, tkl.f32]:
            S_acc = tkl.Register[B, Nq, Nkv, tkl.f32](0.0)
            Q_reg = tkw.read(Q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            K_reg = tkw.read(K, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            S_j = tkw.mma(K_reg, Q_reg, S_acc)
            S_j = tkw.cast(S_j, tkl.f16)
            V_reg = tkw.read(V, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            O_j = tkw.mma(V_reg, S_j, O_jm1)
            return O_j

        tkw.write(repeat, O, elements_per_thread=STORE_ELEMS_PER_THREAD)

    # (8, 128, 128, 64, 256)
    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_Nq: 128,
        BLOCK_Dv: 64,
        BLOCK_Nkv: 64,
        B: batch,
        Nq: q_seq_len,
        Dv: v_head_dim,
        Dqk: qk_head_dim,
        Nkv: kv_seq_len,
    }
    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    compile_config = {"waves_per_eu": 2, "denorm_fp_math_f32": "preserve-sign"}

    dynamic_symbols = []
    dynamic_symbols_map = {}

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        compile_config=compile_config,
        schedule=False,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        torch.manual_seed(0)
        q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16)
        k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16)
        v = device_randn(batch, v_head_dim, kv_seq_len, dtype=torch.float16)
        output = device_zeros(batch, v_head_dim, q_seq_len, dtype=torch.float32)
        mb = double_matmul(q, k, v, output)
        torch_ref = torch.matmul(
            torch.matmul(q, k.transpose(-1, -2)), v.transpose(-1, -2)
        ).transpose(-1, -2)

        if test_dump_generated_mlir:
            filename = f"wave_double_matmul_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        torch.testing.assert_close(
            output.to(torch.float16), torch_ref, atol=5e-2, rtol=1e-2
        )


@require_e2e
# @pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
# @pytest.mark.parametrize(
#     "mfma_variant",
#     [
#         MMAType.F32_16x16x16_F16,
#         # MMAType.F32_32x32x8_F16,
#     ],
# )
def testSingleMatmul(request):
    # Input sizes
    B = tkl.sym.B  # batch
    M = tkl.sym.M  # query sequence length
    N = tkl.sym.N  # key/value sequence length
    K = tkl.sym.K  # query/key head dimension
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    # constraints += [tkw.WaveConstraint(M, BLOCK_M / 1)]
    # constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    # if mfma_variant == MMAType.F32_16x16x16_F16:
    #     Nq_vec = 16
    #     Dv_vec = 16
    # if mfma_variant == MMAType.F32_32x32x8_F16:
    #     Nq_vec = 32
    #     Dv_vec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={B: 0, M: 8, N: 16, K: 16},
        )
    ]

    @tkw.wave(constraints)
    def matmul(
        a: tkl.Memory[B, M, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        # V: tkl.Memory[B, Dv, Nkv, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        init_c = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[init_c])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32]
        ) -> tkl.Register[B, M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    # (8, 128, 128, 64, 256)
    # (8, 256, 128, 192)
    shape = 1, 8, 16, 16
    b, m, n, k = shape
    # batch, q_seq_len, kv_seq_len, qk_head_dim = 8, 256, 128, 192

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 4,
        BLOCK_B: 1,
        BLOCK_M: m,
        BLOCK_N: n,
        BLOCK_K: k,
        B: b,
        M: m,
        K: k,
        N: n,
    }
    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    # config["print_ir_after_all"] = True

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        schedule=False,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        torch.manual_seed(0)
        q = device_randn(b, m, k, dtype=torch.float16)
        k = device_randn(b, n, k, dtype=torch.float16)
        # v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16)
        output = device_zeros(b, m, n, dtype=torch.float32)
        mb = matmul(q, k, output)
        torch_ref = torch.matmul(q.to(torch.float), k.transpose(-1, -2).to(torch.float))

        if test_dump_generated_mlir:
            filename = f"wave_single_matmul_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        # torch.set_printoptions(threshold=1024)

        torch.testing.assert_close(output, torch_ref, atol=2e-3, rtol=5e-3)  #
        assert False


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@param_scheduling
@param_dynamic_dims
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testAttention(
    shape: tuple[int],
    enable_scheduling: bool,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    B = tkl.sym.B  # batch
    M = tkl.sym.M  # query sequence length
    N = tkl.sym.N  # value head dimension
    K1 = tkl.sym.K1  # query/key head dimension
    K2 = tkl.sym.K2  # key/value sequence length
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: Mvec, N: Nvec},
        )
    ]

    if dynamic_dims:
        constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ) -> (
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[B, N, K, tkl.f16]
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[B, N, M, tkl.f32]
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
    }
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
    compile_config = {
        "waves_per_eu": 2,
        "denorm_fp_math_f32": "preserve-sign",
        # "print_ir_after": ["set_node_indices", "expand_graph"],
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}
    if dynamic_dims:
        dynamic_symbols_map[M] = hyperparams[M]
        dynamic_symbols_map[N] = hyperparams[N]
        dynamic_symbols_map[B] = hyperparams[B]
        dynamic_symbols_map[K2] = hyperparams[K2]
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(B)
        dynamic_symbols.append(K2)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[B]
        del hyperparams[K2]

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        compile_config=compile_config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        torch.manual_seed(0)
        q = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = device_randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = device_randn(shape[0], shape[4], shape[2], dtype=torch.float16)
        output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        log2e = 1.44269504089
        dk_sqrt = math.sqrt(1.0 / shape[3])
        # TODO: Add scaling of QK as part of kernel.
        # TODO: Add variant of non-transposed V attention kernel.
        mb = base_attention(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
        torch_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None
        )

        if test_dump_generated_mlir:
            filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        assert_close(output.to(torch.float16), torch_ref, atol=5e-2, rtol=1e-2)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@param_no_scheduling
@param_dynamic_dims
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testAttentionBias(
    shape: tuple[int],
    enable_scheduling: bool,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
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
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: Mvec, N: Nvec},
        )
    ]

    if dynamic_dims:
        constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    @tkw.wave(constraints)
    def base_attention_bias(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        bias: tkl.Memory[B, M, K2, GLOBAL_ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ) -> (
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[B, N, K, tkl.f16]
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[B, N, M, tkl.f32]
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            bias_reg = tkw.read(bias, elements_per_thread=STORE_ELEMS_PER_THREAD)
            x_j = x_j + bias_reg
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
    }
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

    dynamic_symbols = []
    dynamic_symbols_map = {}
    if dynamic_dims:
        dynamic_symbols_map[M] = hyperparams[M]
        dynamic_symbols_map[N] = hyperparams[N]
        dynamic_symbols_map[B] = hyperparams[B]
        dynamic_symbols_map[K2] = hyperparams[K2]
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(B)
        dynamic_symbols.append(K2)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[B]
        del hyperparams[K2]

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        torch.manual_seed(0)
        q = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = device_randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = device_randn(shape[0], shape[4], shape[2], dtype=torch.float16)
        bias = device_randn(shape[0], shape[1], shape[4], dtype=torch.float32)
        output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        log2e = 1.44269504089
        dk_sqrt = math.sqrt(1.0 / shape[3])
        # TODO: Add scaling of QK as part of kernel.
        # TODO: Add variant of non-transposed V attention kernel.
        mb = base_attention_bias(
            q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), bias * log2e, output
        )
        k_t = k.transpose(-1, -2)
        a = torch.matmul(q, k_t) * dk_sqrt
        a += bias
        a = F.softmax(a, dim=-1)
        torch_ref = torch.matmul(a, v)

        if test_dump_generated_mlir:
            filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        if "gfx94" in config["target"]:
            assert_close(output, torch_ref, atol=2e-3, rtol=5e-3)
        else:
            # TODO: Determine why the error is higher on gfx90.
            assert_close(output, torch_ref, atol=3e-3, rtol=8e-1)


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@param_scheduling
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8),
        (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8),
    ],
)
def testAttentionF8(
    shape: tuple[int], enable_scheduling: bool, mfma_variant: tuple[MMAType], request
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
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
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]
    if mfma_variant[0] == MMAType.F32_16x16x32_F8:
        Mvec = 16
        Nvec = 16
    if mfma_variant[0] == MMAType.F32_32x32x16_F8:
        Mvec = 32
        Nvec = 32
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant[0],
            vector_shapes={B: 0, M: Mvec, N: Nvec},
        )
    ]
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ) -> (
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            q_reg = tkw.cast(q_reg, tkl.f8e4m3fnuz)
            k_reg = tkw.cast(k_reg, tkl.f8e4m3fnuz)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f8 = tkw.cast(e_delta, tkl.f8e4m3fnuz)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            v_reg = tkw.cast(v_reg, tkl.f8e4m3fnuz)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f8, new_acc, mfma_variant[1])
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant[0]),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
        BLOCK_B: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
    }
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
    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        torch.manual_seed(0)
        q = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = device_randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = device_randn(shape[0], shape[4], shape[2], dtype=torch.float16)
        output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        log2e = 1.44269504089
        dk_sqrt = math.sqrt(1.0 / shape[3])
        # TODO: Add scaling of QK as part of kernel.
        # TODO: Add variant of non-transposed V attention kernel.
        mb = base_attention(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
        torch_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None
        )
        if test_dump_generated_mlir:
            filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())
        rmse = torch.sqrt(torch.mean(torch.square(output - torch_ref)))
        assert rmse <= 0.006


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@param_no_scheduling
@param_no_dynamic_dims
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
    ],
)
def testFlashDecoding(
    shape: tuple[int],
    enable_scheduling: bool,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    phase_0, phase_1, hyperparams_0, hyperparams_1 = get_decode_attention_kernels(
        shape, mfma_variant
    )
    hyperparams_0.update(get_default_scheduling_params())
    hyperparams_1.update(get_default_scheduling_params())
    config = get_default_run_config()
    if run_bench:
        config["benchmark_batch_size"] = 10
        config["benchmark_repetitions"] = 3
    if dump_perf is not None:
        perf_filename = request.node.name + ".json"
        config["benchmark_results_file"] = os.path.join(
            dump_perf, "tk_" + perf_filename
        )

    torch.manual_seed(0)
    B, M, N, K1, K2 = shape
    U = hyperparams_0[index_symbol("U")]
    q = device_randn(B, M, K1, dtype=torch.float16)
    k = device_randn(B, K2, K1, dtype=torch.float16)
    v = device_randn(B, K2, N, dtype=torch.float16)
    phase_0_output = device_zeros(U, B, N, M, dtype=torch.float32)
    phase_0_output_max = device_zeros(U, B, M, dtype=torch.float32)
    output = device_zeros(B, M, N, dtype=torch.float32)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / K1)

    with tk.gen.TestLaunchContext(
        hyperparams_0,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        # TODO: Add scaling of QK as part of kernel.
        mb_qk = phase_0(
            q * dk_sqrt * log2e,
            k,
            v.permute([0, 2, 1]),
            phase_0_output,
            phase_0_output_max,
        )

    with tk.gen.TestLaunchContext(
        hyperparams_1,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        # TODO: Add variant of non-transposed V attention kernel.
        mb_sv = phase_1(phase_0_output, phase_0_output_max, output)

    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None
    )

    if test_dump_generated_mlir:
        filename = f"wave_phase_0_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(mb_qk.module_op.get_asm())
        filename = f"wave_phase_1_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(mb_sv.module_op.get_asm())

    assert_close(output, torch_ref)
