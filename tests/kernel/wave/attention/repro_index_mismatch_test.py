# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from torch.nn import functional as F
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
from ..common.utils import (
    require_e2e,
    dump_generated_mlir,
    param_bool,
)
from torch.testing import assert_close

# m, n, k
shapes_16x16x16 = [
    (16, 16, 16),
    (16, 16, 32),
    (16, 32, 16),
    (32, 16, 16),
    (64, 32, 128),
]

shapes_32x32x32 = [tuple(2 * dim for dim in shape) for shape in shapes_16x16x16]


def get_param_id(val):
    if isinstance(val, tuple) and all(isinstance(el, int) for el in val):
        return "x".join(str(el) for el in val)
    elif isinstance(val, MMAType):
        return f"MMA_{val.name}"


param_mfma_shape = pytest.mark.parametrize(
    "mfma_variant,shape",
    [(MMAType.F32_16x16x16_F16, shape) for shape in shapes_16x16x16]
    + [(MMAType.F32_32x32x8_F16, shape) for shape in shapes_32x32x32],
    ids=get_param_id,
)


def get_repro_410_kernel(
    dim_b: int,
    dim_m: int,
    dim_n: int,
    dim_k: int,
    mfma_variant: MMAType,
    noop_permute: bool = False,
):
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K

    BLOCK_K = tkl.sym.BLOCK_K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_B = tkl.sym.BLOCK_B
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.MFMA_INPUT_ELS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.MFMA_OUTPUT_ELS_PER_THREAD

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    elif mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(B, BLOCK_B, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        ),
    ]

    @tkw.wave(constraints)
    def repro_410(
        a: tkl.Memory[B, K, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, K, M, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        dummy_reg = tkl.Register[B, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[dummy_reg])
        def loop_k(dummy_prev: tkl.Register[B, tkl.f32]):
            a_i_for_c = tkw.read(a, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            b_j = tkw.read(b, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)

            c_acc = tkl.Register[B, K, M, tkl.f32](0.0)
            c_ij = tkw.mma(a_i_for_c, b_j, c_acc)

            # TODO(#410): we have to do a no-op permute first or the cast gets
            # confused, resulting in either a compiler failure (16x16x16 mfma)
            # or a miscompile and incorrect output (32x32x8 mfma).
            if noop_permute:
                c_ij = tkw.permute(c_ij, [B, K, M])

            unity_reg = tkl.Register[B, K, M, tkl.f16](1.0)

            c_ij = tkw.cast(c_ij, tkl.f16) * unity_reg

            tkw.write(c_ij, c, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            return dummy_prev

    hyperparams = {
        MFMA_INPUT_ELS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        MFMA_OUTPUT_ELS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        B: dim_b,
        BLOCK_K: vec_size,
        BLOCK_M: vec_size,
        M: dim_m,
        N: dim_n,
        K: dim_k,
    }

    return repro_410, hyperparams


@require_e2e
@param_bool("noop_permute", "perm")
@param_mfma_shape
def testRepro410(mfma_variant: MMAType, shape: tuple[int, ...], noop_permute: bool):
    torch.manual_seed(0)
    dim_m, dim_n, dim_k = shape
    dim_b = 1
    cmp_params = dict(atol=3e-3, rtol=3e-3, check_dtype=False)

    a = device_randn(dim_b, dim_k, dim_n, dtype=torch.float16) / 10
    b = device_randn(dim_b, dim_m, dim_n, dtype=torch.float16) / 10

    c_ref = torch.matmul(a, b.transpose(-1, -2))

    repro_410, hyperparams = get_repro_410_kernel(
        dim_b=dim_b,
        dim_m=dim_m,
        dim_n=dim_n,
        dim_k=dim_k,
        mfma_variant=mfma_variant,
        noop_permute=noop_permute,
    )
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    repro_410 = wave_compile(options, repro_410)

    c = device_zeros(dim_b, dim_k, dim_m, dtype=torch.float16)

    asm = repro_410(a, b, c)

    if dump_generated_mlir:
        filename = f"out/wave_repro_410_{mfma_variant}_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)
        print(f"IR dumped to {filename}")

    assert_close(c, c_ref, **cmp_params)
