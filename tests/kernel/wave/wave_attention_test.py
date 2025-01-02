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
    (8, 32, 128, 64, 256),
    (40, 1024, 64, 64, 1024),
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

        print(f"{q.shape=}\n{k.shape=}\n{v.shape=}\n{output.shape=}")
        torch_qk = torch.matmul(q, k.transpose(-1, -2))  # m x k1 @ k1 x k2 -> m x k2
        print(f"{torch_qk.shape=}")
        torch_ref = torch.matmul(
            torch_qk, v.transpose(-1, -2)
        )  # m x k2 @ k2 x n -> m x n
        output_for_cmp = output.transpose(-1, -2).to(torch.float16)
        assert_close(output_for_cmp, torch_ref, atol=5e-2, rtol=1e-3)


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
    M = tkl.sym.M  # query sequence length
    N = tkl.sym.N  # value head dimension
    K1 = tkl.sym.K1  # query/key head dimension
    K2 = tkl.sym.K2  # key/value sequence length

    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
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
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / WAVES_PER_BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / WAVES_PER_BLOCK_N)]
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
    def attention_fwd(
        q:   tkl.Memory[B, M,  K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k:   tkl.Memory[B, K2, K1, ADDRESS_SPACE,        tkl.f16],
        v:   tkl.Memory[B, N,  K2, ADDRESS_SPACE,        tkl.f16],
        o:   tkl.Memory[B, M,  N,  GLOBAL_ADDRESS_SPACE, tkl.f16],
        lse: tkl.Memory[B, M,      GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        init_o = tkl.Register[B, N, M, tkl.f32](0.0)
        init_l = tkl.Register[B, M, tkl.f32](0.0)
        init_m = tkl.Register[B, M, tkl.f32](-1e6)

        @tkw.reduction(K2, init_args=[init_m, init_l, init_o])
        def repeat(
            m_prev: tkl.Register[B, M, tkl.f32],
            l_prev: tkl.Register[B, M, tkl.f32],
            o_prev: tkl.Register[B, N, M, tkl.f32],
        ) -> tuple[
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ]:
            s_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            s_j = tkw.mma(k_reg, q_reg, s_acc)
            s_j = tkw.permute(s_j, target_shape=[B, M, K2])
            m_j = tkw.max(s_j, m_prev, dim=K2)
            e_delta_max = tkw.exp2(m_prev - m_j)
            p_j = tkw.exp2(s_j - m_j)
            l_init = e_delta_max * l_prev
            l_j = tkw.sum(p_j, l_init, dim=K2)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            o_init = e_delta_max * o_prev
            o_j = tkw.mma(v_reg, tkw.cast(p_j, tkl.f16), o_init)
            return m_j, l_j, o_j

        m_i, l_i, o_i = repeat
        o_i = tkw.reciprocal(l_i) * o_i
        tkw.write(tkw.cast(o_i, tkl.f16), o, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)
        l_i = m_i + tkw.log2(l_i)
        tkw.write(tkw.cast(l_i, tkl.f16), lse, elements_per_thread=1)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 16,
        BLOCK_N: 16,
        BLOCK_K2: 16,
        B: batch,
        M: q_seq_len,
        N: v_head_dim,
        K1: qk_head_dim,
        K2: kv_seq_len,
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
    M = tkl.sym.M  # query sequence length
    N = tkl.sym.N  # value head dimension
    K1 = tkl.sym.K1  # query/key head dimension
    K2 = tkl.sym.K2  # key/value sequence length

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
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(K1, BLOCK_K1, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / WAVES_PER_BLOCK_M)]
    constraints += [tkw.WaveConstraint(K1, BLOCK_K1 / WAVES_PER_BLOCK_K1)]
    constraints += [tkw.WaveConstraint(B, BLOCK_B / WAVES_PER_BLOCK_B)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
        K1vec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32
        K1vec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(WAVES_PER_BLOCK_M, WAVES_PER_BLOCK_K1, WAVES_PER_BLOCK_B),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: Mvec, N: Nvec, K1: K1vec},
        )
    ]

    # i = tkw.IndexMapping.iterator(0)
    # j = tkw.IndexMapping.iterator(1)
    # k = tkw.IndexMapping.iterator(2)
    # mapping = tkw.IndexMapping(
    #     num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    # )

    @tkw.wave(constraints)
    def attention_bwd_dq(
        q:   tkl.Memory[B, M,  K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k:   tkl.Memory[B, K2, K1, ADDRESS_SPACE,        tkl.f16],
        v:   tkl.Memory[B, K2, N,  ADDRESS_SPACE,        tkl.f16],
        do:  tkl.Memory[B, M,  N,  GLOBAL_ADDRESS_SPACE, tkl.f16],
        lse: tkl.Memory[B, M,      GLOBAL_ADDRESS_SPACE, tkl.f16],
        D:   tkl.Memory[B, M,      GLOBAL_ADDRESS_SPACE, tkl.f16],
        dq:  tkl.Memory[B, M,  K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        init_dq = tkl.Register[B, M, K1, tkl.f32](0.0)

        @tkw.reduction(K2, init_args=[init_dq])
        def repeat(dq_prev: tkl.Register[B, M, K1, tkl.f32]) -> tkl.Register[B, M, K1, tkl.f32]:
            s_acc = tkl.Register[B, K2, M, tkl.f32](0.0)                     # B x K2 x M
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)   # B x M  x K1
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)   # B x K2 x K1
            s_j = tkw.mma(k_reg, q_reg, s_acc)                               # B x K2 x M
            s_j = tkw.permute(s_j, target_shape=[B, M, K2])                  # B x M  x K2
            lse_reg = tkw.read(lse, elements_per_thread=1)                   # B x M
            p_j = tkw.exp2(tkw.cast(s_j, tkl.f16) - lse_reg)                 # B x M  x K2
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)   # B x K2 x N
            do_reg = tkw.read(do, elements_per_thread=LOAD_ELEMS_PER_THREAD) # B x M  x N
            dp_acc = tkl.Register[B, M, K2, tkl.f32](0.0)                    # B x M  x K2
            dp_j = tkw.mma(do_reg, v_reg, dp_acc)                            # B x M  x K2
            D_reg = tkw.read(D, elements_per_thread=1)                       # B x M
            ds_j = p_j * (tkw.cast(dp_j, tkl.f16) - D_reg)                   # B x M  x K2
            dq_j = tkw.mma(ds_j, tkw.permute(k_reg, [B, K1, K2]), dq_prev)   # B x M  x K1
            return dq_j

        tkw.write(tkw.cast(repeat, tkl.f16), dq, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 16,
        BLOCK_K1: 16,
        BLOCK_K2: 16,
        B: batch,
        M: q_seq_len,
        N: v_head_dim,
        K1: qk_head_dim,
        K2: kv_seq_len,
    }

    return attention_bwd_dq, hyperparams

@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testAttentionMine(shape: tuple[int], mfma_variant: MMAType, request):
    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
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
    compile_config = {
        "waves_per_eu": 2,
        "denorm_fp_math_f32": "preserve-sign",
        # "print_ir_after": ["set_node_indices", "expand_graph"],
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
        torch.manual_seed(0)
        q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16, requires_grad=True)
        k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16, requires_grad=True)
        v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16, requires_grad=True)

        ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, scale=1)

        # temporarily not scaling so I don't have to figure out how to get that into the backward kernel
        dk_sqrt = math.sqrt(1.0 / qk_head_dim)
        
        # Fake derivative for whatever the loss function might be
        do = device_randn(batch, q_seq_len, v_head_dim, dtype=torch.float16)
        # do = device_zeros(batch, q_seq_len, v_head_dim, dtype=torch.float16)

        # IDK why we have to do retain_grad here, but the deepspeed test doesn't
        q.retain_grad()
        k.retain_grad()
        v.retain_grad()
        ref_out.backward(do)

        ref_dq = q.grad.detach()
        ref_dk = k.grad.detach()
        ref_dv = v.grad.detach()

        q.grad = None
        k.grad = None
        v.grad = None

        s = torch.matmul(q, k.transpose(-1, -2))
        p = torch.softmax(s, dim=-1)
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
        ds2 = s.grad.detach()
        dp = p.grad.detach()


        assert_close(ref_out, o, atol=5e-2, rtol=1e-3)
        assert_close(q.grad, ref_dq, atol=5e-2, rtol=1e-3)

        assert_close(dq, ref_dq, atol=5e-2, rtol=1e-3)
        assert_close(dk, ref_dk, atol=5e-2, rtol=1e-3)
        assert_close(dv, ref_dv, atol=5e-2, rtol=1e-3)

        ref_dq = dq
        ref_dk = dk
        ref_dv = dv
        ref_ds = ds2
        ref_dp = dp

        q = q.detach()
        k = k.detach()
        v = v.detach()

        s = torch.matmul(q, k.transpose(-1, -2))
        p = torch.softmax(s, dim=-1)

        dv = torch.matmul(p.transpose(-1, -2), do)

        assert_close(dv, ref_dv, atol=5e-2, rtol=1e-3)

        # D = torch.sum(fake_grad * torch_o, -1)
        dp = torch.matmul(do, v.transpose(-1, -2))

        assert_close(dp, ref_dp, atol=5e-2, rtol=1e-3)

        # Torch function for the 
        # ds =  p * (dp - D)
        I = to_default_device(torch.eye(q_seq_len, kv_seq_len, dtype=torch.float16))
        ds1 = dp * p * (I - p)


        ds2 = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)

        for b in range(batch):
            for i in range(q_seq_len):
                p_diag = torch.diag(p[b, i, :])
                dp_row = dp[b, i, :]
                p_row = p[b, i, :].unsqueeze(0)
                p_sq = torch.matmul(p_row, p_row.transpose(-1, -2))
                ds_row = torch.matmul((p_diag - p_sq), dp_row)
                ds2[b, i, :] = ds_row

        ds3 = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
        for b in range(batch):
            for i in range(q_seq_len):
                # alternate with explicit loop
                for j in range(kv_seq_len):
                    ind = 1 if i == j else 0
                    ds3[b, i, j] = dp[b, i, j] * p[b, i, j]*(ind - p[b, i, j])


        ds4 = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
        for b in range(batch):
            for row in range(q_seq_len):
                p_row = p[b, row, :]
                jacobian = device_zeros(kv_seq_len, kv_seq_len, dtype=torch.float16)

                for i in range(kv_seq_len):
                    for j in range(kv_seq_len):
                        ind = 1 if i == j else 0
                        jacobian[i, j] = p[b, row, j] * (ind - p[b, row, i])
                

                ds4[b, row, :] = torch.matmul(dp[b, row, :], jacobian)

        ds5 = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
        for b in range(batch):
            for row in range(q_seq_len):
                p_row = p[b, row, :]
                jacobian = p_row.unsqueeze(0) * (to_default_device(torch.eye(kv_seq_len, dtype=torch.float16)) - p_row.unsqueeze(-1))

                ds5[b, row, :] = torch.matmul(dp[b, row, :], jacobian.transpose(-1, -2))


        ds_attempts = [ref_ds, ds1, ds2, ds3, ds4, ds5]

        for i, ds_lhs in enumerate(ds_attempts):
            for j, ds_rhs in enumerate(ds_attempts):
                if j <= i:
                    continue
                if torch.isclose(ds_lhs, ds_rhs, atol=5e-2, rtol=1e-3).all():
                    print(f"ds{i} and ds{j} match")
        
        assert_close(ds1, ds3, atol=5e-2, rtol=1e-3)
        assert_close(ds2, ds3, atol=5e-2, rtol=1e-3)
        assert_close(ds1, ds2, atol=5e-2, rtol=1e-3)
        assert_close(ds1, ref_ds, atol=5e-2, rtol=1e-3)

        # print(ds.shape)
        manual_dq = torch.matmul(ds2, k)
        torch.set_printoptions(sci_mode=False, linewidth=160, precision=3)
        print(manual_dq.detach().cpu().contiguous().to(torch.float32)) # Just converting them to make it print less, honestly
        print(ref_dq.detach().cpu().contiguous().to(torch.float32)) # Just converting them to make it print less, honestly
        assert_close(manual_dq, ref_dq,  atol=5e-2, rtol=1e-3)

        output = device_zeros(batch, q_seq_len, v_head_dim, dtype=torch.float16)
        lse = device_zeros(batch, q_seq_len, dtype=torch.float16)
        log2e = 1.44269504089

        mb_fwd = attention_fwd(q * log2e, k, v.transpose(-1, -2), output, lse)

        if test_dump_generated_mlir:
            filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_fwd.module_op.get_asm())

        assert_close(output, ref_out, atol=5e-2, rtol=1e-3)


    attention_bwd_dq, hyperparams = get_attention_bwd_kernel(
        batch=batch,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        q_seq_len=q_seq_len,
        v_head_dim=v_head_dim,
        mfma_variant=mfma_variant,
    )
    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    compile_config = {
        "waves_per_eu": 2,
        "denorm_fp_math_f32": "preserve-sign",
        # "print_ir_after": ["expand_graph"],
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
        torch.manual_seed(0)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # TODO: maybe compute this within the kernel? The description of the
        # backward pass in the Flash Attention 2 paper has it computed
        # external to both loops though.
        D = torch.sum(do * output, -1)


        # q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16, requires_grad=True)
        # k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16, requires_grad=True)
        # v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16, requires_grad=True)
        # fake_grad = device_randn(batch, q_seq_len, v_head_dim, dtype=torch.float16)

        # q:   tkl.Memory[B, M,  K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # k:   tkl.Memory[B, K2, K1, ADDRESS_SPACE,        tkl.f16],
        # v:   tkl.Memory[B, K2, N,  ADDRESS_SPACE,        tkl.f16],
        # do:  tkl.Memory[B, M,  N,  GLOBAL_ADDRESS_SPACE, tkl.f16],
        # dq:  tkl.Memory[B, M,  K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # lse: tkl.Memory[B, M,      GLOBAL_ADDRESS_SPACE, tkl.f16],
        # D:   tkl.Memory[B, M,      GLOBAL_ADDRESS_SPACE, tkl.f16],

        print(f"{q.shape=}")
        print(f"{k.shape=}")
        print(f"{v.shape=}")
        print(f"{do.shape=}")
        print(f"{dq.shape=}")
        print(f"{lse.shape=}")
        print(f"{D.shape=}")

        print(f"q:   {batch} x {q_seq_len} x  {qk_head_dim}")
        print(f"k:   {batch} x {kv_seq_len} x {qk_head_dim}")
        print(f"v:   {batch} x {kv_seq_len} x {v_head_dim}")
        print(f"do:  {batch} x {q_seq_len} x  {v_head_dim}")
        print(f"dq:  {batch} x {q_seq_len} x  {qk_head_dim}")
        print(f"lse: {batch} x {q_seq_len}")
        print(f"D:   {batch} x {q_seq_len}")

        mb_bwd = attention_bwd_dq(q * log2e, k, v, do, lse, D, dq)

        if test_dump_generated_mlir:
            filename = f"wave_attention_bwd_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_bwd.module_op.get_asm())

        print(f"{dq=}")
        print(f"{ref_dq=}")
        assert_close(dq, ref_dq, atol=5e-2, rtol=1e-3)
        



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
        init_O = tkl.Register[B, Nq, Dv, tkl.f32](0.0)

        @tkw.reduction(Nkv, init_args=[init_O])
        def repeat(
            O_jm1: tkl.Register[B, Nq, Dv, tkl.f32]
        ) -> tkl.Register[B, Nq, Dv, tkl.f32]:
            S_acc = tkl.Register[B, Nq, Nkv, tkl.f32](0.0)
            Q_reg = tkw.read(Q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            K_reg = tkw.read(K, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            S_j = tkw.mma(K_reg, Q_reg, S_acc)
            S_j = tkw.cast(S_j, tkl.f16)
            V_reg = tkw.read(V, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            O_j = tkw.mma(S_j, V_reg, O_jm1)
            return O_j

        tkw.write(
            repeat, O, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

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
        v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16)
        output = device_zeros(batch, v_head_dim, q_seq_len, dtype=torch.float32)
        mb = double_matmul(q, k, v.transpose(-1, -2), output)
        torch_ref = torch.matmul(torch.matmul(q, k.transpose(-1, -2)), v).transpose(
            -1, -2
        )

        if test_dump_generated_mlir:
            filename = f"wave_double_matmul_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        torch.testing.assert_close(
            output.to(torch.float16), torch_ref, atol=5e-2, rtol=1e-3
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
        "print_ir_after": ["set_node_indices", "expand_graph"],
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

        assert_close(output.to(torch.float16), torch_ref, atol=5e-2, rtol=1e-3)


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
