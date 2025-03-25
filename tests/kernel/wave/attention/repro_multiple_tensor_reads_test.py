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


def get_repro_603_kernel(
    dim_m: int,
    dim_n: int,
    dim_k: int,
    mfma_variant: MMAType,
    read_twice: bool = False,
):
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K

    BLOCK_K = tkl.sym.BLOCK_K
    BLOCK_M = tkl.sym.BLOCK_M
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.MFMA_INPUT_ELS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.MFMA_OUTPUT_ELS_PER_THREAD

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    elif mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    flip_n_k_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={K: j, N: i},
        outputs={N: i, K: j},
    )

    @tkw.wave(constraints)
    def repro_603(
        a: tkl.Memory[K, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        e: tkl.Memory[M, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[K, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
        d: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        d_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[d_reg])
        def loop_m(d_acc: tkl.Register[M, N, tkl.f32]):
            a_reg_for_c = tkw.read(a, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)

            c_acc = tkl.Register[K, M, tkl.f32](0.0)
            c_acc = tkw.mma(a_reg_for_c, b_reg, c_acc)
            tkw.write(c_acc, c, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            e_reg = tkw.read(e, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)

            # TODO(#603): Wave has implicit layout requirements for MMAs, so we
            # have to read q again.
            if read_twice:
                a_reg_for_d = tkw.read(
                    a,
                    mapping=flip_n_k_read_mapping,
                    elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
                )
            else:
                a_reg_for_d = tkw.permute(a_reg_for_c, [N, K])
            d_acc = tkw.mma(e_reg, a_reg_for_d, d_acc)

            return d_acc

        tkw.write(
            tkw.cast(loop_m, tkl.f16),
            d,
            elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD,
        )

    hyperparams = {
        MFMA_INPUT_ELS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        MFMA_OUTPUT_ELS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_K: vec_size,
        BLOCK_M: vec_size,
        M: dim_m,
        N: dim_n,
        K: dim_k,
    }

    return repro_603, hyperparams


@require_e2e
@param_bool("read_twice")
@param_mfma_shape
def testRepro603(mfma_variant: MMAType, shape: tuple[int, ...], read_twice: bool):
    """This tests a kernel only for the gradient of k."""
    torch.manual_seed(0)
    dim_m, dim_n, dim_k = shape
    cmp_params = dict(atol=3e-3, rtol=3e-3, check_dtype=False)

    a = device_randn(dim_k, dim_n, dtype=torch.float16) / 10
    b = device_randn(dim_m, dim_n, dtype=torch.float16) / 10
    e = device_randn(dim_m, dim_k, dtype=torch.float16) / 10

    c_ref = torch.matmul(a, b.transpose(-1, -2))
    d_ref = torch.matmul(e, a)

    repro_603, hyperparams = get_repro_603_kernel(
        dim_m=dim_m,
        dim_n=dim_n,
        dim_k=dim_k,
        mfma_variant=mfma_variant,
        read_twice=read_twice,
    )
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    repro_603 = wave_compile(options, repro_603)

    c = device_zeros(dim_k, dim_m, dtype=torch.float32)
    d = torch.zeros_like(b)

    asm = repro_603(a, b, e, c, d)

    if dump_generated_mlir:
        filename = f"out/wave_repro_603_read_{'twice' if read_twice else 'once'}_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)
        print(f"IR dumped to {filename}")

    assert_close(c, c_ref, **cmp_params)
    assert_close(d, d_ref, **cmp_params)
