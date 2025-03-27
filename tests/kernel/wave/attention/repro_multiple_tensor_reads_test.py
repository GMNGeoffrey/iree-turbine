# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import pathlib
import re

import pytest
import torch
from torch.testing import assert_close, make_tensor

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
    get_default_device,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
from ..common.utils import (
    require_e2e,
    dump_generated_mlir,
    param_bool,
)

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
    BLOCK_N = tkl.sym.BLOCK_N
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.MFMA_INPUT_ELS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.MFMA_OUTPUT_ELS_PER_THREAD

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    elif mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
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
        a_transpose: tkl.Memory[N, K, GLOBAL_ADDRESS_SPACE, tkl.f16],
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
            tkw.write(
                a_reg_for_d, a_transpose, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD
            )
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
        BLOCK_N: vec_size,
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

    # a = device_randn(dim_k, dim_n, dtype=torch.float16) / 10
    # a = make_tensor(dim_k, dim_n, dtype=torch.float16, device=get_default_device(), low=0.001, high=0.1)
    a = (
        torch.arange(
            0, 256, 1, device=get_default_device(), dtype=torch.float16
        ).reshape(16, 16)
        / 100
    )
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
        canonicalize=True,
    )
    options = set_default_run_config(options)
    repro_603 = wave_compile(options, repro_603)

    asm = prettify_mlir(repro_603.asm, options)

    if dump_generated_mlir:
        filename = f"out/wave_repro_603_read_{'twice' if read_twice else 'once'}_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)
        print(f"IR dumped to {filename}")

    c = device_zeros(dim_k, dim_m, dtype=torch.float32)
    d = torch.zeros_like(b)
    a_transpose = device_zeros(dim_n, dim_k, dtype=torch.float16)
    repro_603(a, b, e, a_transpose, c, d)

    assert_close(c, c_ref, **cmp_params)
    print(small_tensor_string(a, "a"))
    assert_close(a_transpose, a.transpose(-1, -2), atol=0, rtol=0)
    assert_close(d, d_ref, **cmp_params)


def small_tensor_string(
    t,
    name="",
    row_limit=50,
    col_limit=150,
    min_important_value=1e-5,
    sci_mode=None,
    precision=None,
):
    # Unfortunately, pytest usually captures the output and so we can't access the real width here :-(
    # col_limit = col_limit or shutil.get_terminal_size().columns
    shape = "x".join([str(i) for i in t.shape])
    if len(t.shape) > 2:
        t = t.squeeze()
    if len(t.shape) < 2:
        t = t.unsqueeze(0)

    abs = torch.abs(t)
    # Things large enough that we can't round them off to zero and small enough
    # that we need scientific notation to print them or if anything's big enough
    # that we need scientific notation.
    sci_mode = sci_mode or (
        torch.any(torch.logical_and(abs > min_important_value, abs < 1e-3))
        or torch.max(abs) > 1e3
    )
    precision = precision or 2 if sci_mode else 3

    def fallback():
        with torch._tensor_str.printoptions(
            precision=2 if sci_mode else 3,
            linewidth=col_limit,
            sci_mode=sci_mode,
            threshold=0,
        ):
            return f"{name}[{shape}], {t.dtype}:\n{t}"

    if len(t.shape) > 2 or t.shape[0] > row_limit:
        return fallback()

    def f_entry(d, width=0):
        return (
            f"{d: {width}.{precision}e}" if sci_mode else f"{d: {width}.{precision}f}"
        )

    width = max(len(f_entry(d)) for d in t.flatten().tolist())

    row_width = len(" ".join(f_entry(d, width) for d in t[0].tolist()))

    if row_width > col_limit:
        return fallback()

    rows = []
    for row in t:
        rows.append(" ".join(f_entry(d, width) for d in row.tolist()))

    nl = "\n"
    return f"{name}[{shape}], {t.dtype}:\n{nl.join(rows)}"


@require_e2e
@param_mfma_shape
def testOverrideAsm(mfma_variant: MMAType, shape: tuple[int, ...]):
    torch.manual_seed(0)
    dim_m, dim_n, dim_k = shape
    cmp_params = dict(atol=3e-3, rtol=3e-3, check_dtype=False)

    # a = device_randn(dim_k, dim_n, dtype=torch.float16) / 10
    a = device_zeros(dim_k, dim_n, dtype=torch.float16)
    a[:4, :4] = torch.arange(16, dtype=torch.float16).reshape(4, 4)

    b = device_randn(dim_m, dim_n, dtype=torch.float16) / 10
    # e = device_randn(dim_m, dim_k, dtype=torch.float16) / 10
    e = device_zeros(dim_m, dim_k, dtype=torch.float16)
    e[:4, :4] = torch.arange(16, 32, dtype=torch.float16).reshape(4, 4)

    c_ref = torch.matmul(a, b.transpose(-1, -2))
    d_ref = torch.matmul(e, a)

    bad_d_ref = torch.matmul(e, a.transpose(-1, -2))

    repro_603, hyperparams = get_repro_603_kernel(
        dim_m=dim_m,
        dim_n=dim_n,
        dim_k=dim_k,
        mfma_variant=mfma_variant,
        read_twice=False,
    )
    hyperparams.update(get_default_scheduling_params())

    asm_path = pathlib.Path(f"wave_repro_603_override_{'x'.join(map(str, shape))}.mlir")
    # asm_path = "out" / pathlib.Path(f"wave_repro_603_read_once_{'x'.join(map(str, shape))}.mlir")
    asm = asm_path.read_text()

    options = WaveCompileOptions(
        subs=hyperparams,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        override_mlir=asm,
    )
    options = set_default_run_config(options)
    repro_603 = wave_compile(options, repro_603)

    c = device_zeros(dim_k, dim_m, dtype=torch.float32)
    d = torch.zeros_like(b)

    asm = repro_603(a, b, e, c, d)

    if dump_generated_mlir:
        filepath = "out" / asm_path
        filepath.write_text(asm)
        print(f"IR dumped to {filepath}")

    assert_close(c, c_ref, **cmp_params)
    print(small_tensor_string(e, "e"))
    print(small_tensor_string(a, "a"))
    # assert not torch.allclose(d, bad_d_ref, atol=3e-3, rtol=3e-3)
    assert_close(d, d_ref, **cmp_params)


def get_transpose_kernel(dim_size: int):
    M = tkl.sym.M
    N = tkl.sym.N

    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_M = tkl.sym.BLOCK_M

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: dim_size, N: dim_size},
            max_bits_per_load=512,
        ),
    ]

    @tkw.wave(constraints)
    def transpose(
        a: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        a_transpose: tkl.Memory[N, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(a, elements_per_thread=dim_size)
        a_transpose_reg = tkw.permute(a_reg, [N, M])
        tkw.write(a_transpose_reg, a_transpose, elements_per_thread=dim_size)

    hyperparams = {
        BLOCK_M: dim_size,
        BLOCK_N: dim_size,
        M: dim_size,
        N: dim_size,
    }

    return transpose, hyperparams


def prettify_mlir(asm: str, options: WaveCompileOptions):
    # Sub arguments
    for i, b in enumerate(options.kernel_sig.kernel_buffer_bindings):
        if b.name:
            asm = re.sub(rf"%arg{i}\b", f"%arg_{b.name}", asm)
            arg_access = re.findall(
                rf"%(\d+) = stream.binding.subspan %arg_{b.name}\[%c0\]", asm
            )
            if len(arg_access) > 1:
                raise RuntimeError(f"Found more than one access of binding {b.name}")
            if arg_access:
                asm = re.sub(rf"%{arg_access[0]}\b", f"%{b.name}", asm)

            loads = re.findall(rf"%(\d+) = vector.load %{b.name}\b", asm)

            for i, load_ssa in enumerate(loads):
                find = rf"%{load_ssa}\b"
                replace = f"%{b.name}_reg_{i}"
                asm = re.sub(find, replace, asm)

    # Sub floats
    log2e_matches = re.findall(
        r"%(\w+) = arith\.constant dense<1\.44269502(?:e\+00)?> : vector<(\d+)x(f\d+)>",
        asm,
    )
    for m in log2e_matches:
        ssa, v_size, dtype = m
        asm = re.sub(rf"%{ssa}\b", f"%log2e_{v_size}v{dtype}", asm)

    zero_matches = re.findall(
        r"%(\w+) = arith\.constant dense<0\.0*(?:e\+00)?> : vector<(\d+)x(f\d+)>",
        asm,
    )
    for m in zero_matches:
        ssa, v_size, dtype = m
        asm = re.sub(rf"%{ssa}\b", f"%c0_{v_size}v{dtype}", asm)

    neg_inf_matches = re.findall(
        r"%(\w+) = arith\.constant dense<-1\.0*e\+06> : vector<(\d+)x(f\d+)>",
        asm,
    )
    for m in neg_inf_matches:
        ssa, v_size, dtype = m
        asm = re.sub(rf"%{ssa}\b", f"%c_minf_{v_size}v{dtype}", asm)

    return asm


def testTranspose():
    torch.manual_seed(0)
    a = torch.arange(256, device=get_default_device(), dtype=torch.float32).reshape(
        16, 16
    )

    transpose, hyperparams = get_transpose_kernel(dim_size=16)
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        canonicalize=True,
    )
    options = set_default_run_config(options)
    transpose = wave_compile(options, transpose)

    asm = prettify_mlir(transpose.asm, options)

    asm_path = pathlib.Path(f"wave_transpose_16x16.mlir")

    if dump_generated_mlir:
        filepath = "out" / asm_path
        filepath.write_text(asm)
        print(f"IR dumped to {filepath}")

    a_transpose_ref = a.transpose(-1, -2)

    a_transpose = torch.zeros_like(a_transpose_ref)
    transpose(a, a_transpose)
    assert_close(a_transpose, a_transpose_ref)
