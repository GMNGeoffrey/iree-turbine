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
    get_default_device
)
from iree.turbine.kernel.wave.constraints import MMAType
from ..common.utils import (
    require_e2e,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
import os
import json
from torch.testing import assert_close



default_test_shapes = {}
# Order of shapes: (B, M, N, K1, K2)
default_test_shapes["test_attention"] = [
    # (2, 64, 128, 32, 256),
    (1, 16, 16, 16, 32),
    (1, 16, 16, 32, 16),
    (1, 16, 32, 16, 16),
    (1, 32, 16, 16, 16),
    # (40, 1024, 64, 64, 1024),
    (1, 16, 16, 16, 16),
    # (1, 16, 16, 8, 16),
    (1, 16, 16, 64, 16),
    (1, 16, 16, 32, 32),
]

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


def small_tensor_string(t, name="", print_limit=512, min_important_value=5e-2):
    print_limit = 512
    if t.shape.numel() > print_limit:
        return f"{t}"
    shape = "x".join([str(i) for i in t.shape])

    abs = torch.abs(t)
    print_mode = "reg"
    
    # Things large enough that we can't round them off to zero and small enough
    # that we need scientific notation to print them or if anything's big enough
    # that we need scientific notation.
    if torch.any(torch.logical_and(abs > min_important_value, abs < 1e-3)) or torch.max(abs) > 1e3:
        print_mode = "sci"


    width = max(
        len(f"{d: .3f}" if print_mode == "reg" else f"{d: .2e}")
        for d in t.flatten().tolist()
    )

    if len(t.shape) > 2:
        t = t.squeeze()
    if len(t.shape) < 2:
        t = t.unsqueeze(0)
    if len(t.shape) > 2:
        return f"{t}" # Can't print this nicely
    
    rows = []
    for row in t:
        rows.append(
            " ".join(
                f"{d: {width}.3f}" if print_mode == "reg" else f"{d: {width}.2e}"
                for d in row.tolist()
            )
        )

    nl = "\n"
    return f"\n{name}[{shape}], {t.dtype}:\n{nl.join(rows)}\n"


def print_small_tensor(t, *args, **kwargs):
    print(small_tensor_string(t, *args, **kwargs))


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
    if M_qs == K2_kvs:
        BLOCK_M_Br = M_qs // 4
        BLOCK_K2_Bc = K2_kvs // 2
    else:
        BLOCK_M_Br = M_qs // 2
        BLOCK_K2_Bc = K2_kvs // 2
    # BLOCK_M_Br = 8
    # BLOCK_K2_Bc = 4

    assert M_qs % BLOCK_M_Br == 0
    assert K2_kvs % BLOCK_K2_Bc == 0

    for batch in range(B):
        for start_k2 in range(0, K2_kvs, BLOCK_K2_Bc):
            k_j = k[batch, start_k2 : start_k2 + BLOCK_K2_Bc, :]
            assert k_j.shape == (BLOCK_K2_Bc, K1_qkd)
            v_j = v[batch, start_k2 : start_k2 + BLOCK_K2_Bc, :]
            assert v_j.shape == (BLOCK_K2_Bc, N_vd)

            dv_j = device_zeros(BLOCK_K2_Bc, N_vd)
            dk_j = device_zeros(BLOCK_K2_Bc, K1_qkd)

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
            s_ij = tkw.permute(s_ij, target_shape=[B, M_qs, K2_kvs])
            tkw.write(s_ij, s, elements_per_thread=STORE_ELEMS_PER_THREAD)
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
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(K2_kvs, BLOCK_K2, 0),
        # Degenerate distribution seems to fix some bugs
        tkw.WorkgroupConstraint(K1_qkd, K1_qkd, 1),
        # This one causes some other weird compilation error though...
        tkw.WorkgroupConstraint(N_vd, N_vd, 2),
        # Can only have 3 dimensions distributed in actual blocks or the
        # compiler tries to index to far into waves_per_block (and if that is
        # made longer there's just a fatal crash), so batch dimension needs to
        # be last.
        tkw.WorkgroupConstraint(B, BLOCK_B, 3),
        tkw.TilingConstraint(M_qs, BLOCK_M),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_k2_m_write_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, K2_kvs: j, M_qs: k}, outputs={B: i, M_qs: k, K2_kvs: j}
    )

    flip_m_n_read_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, M_qs: k, N_vd: j}, outputs={B: i, N_vd: j, M_qs: k}
    )
    flip_m_k1_read_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, M_qs: k, K1_qkd: j}, outputs={B: i, K1_qkd: j, M_qs: k},
    )
    flip_m_k2_read_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, M_qs: k, K2_kvs: j}, outputs={B: i, K2_kvs: j, M_qs: k},
    )
    flip_k2_k1_read_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, K2_kvs: k, K1_qkd: j}, outputs={B: i, K1_qkd: j, K2_kvs: k},
    )


    @tkw.wave(constraints)
    def attention_bwd(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        do: tkl.Memory[B, M_qs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # This has to be loaded in this way or set_node_indices fails with resolving thread shapes
        lse: tkl.Memory[B, K2_kvs, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        D: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dq: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dk: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dv: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        p: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        ds: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dp: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        dp_sub: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        dv_init = tkl.Register[B, K2_kvs, N_vd, tkl.f32](0.0)
        dk_init = tkl.Register[B, K2_kvs, K1_qkd, tkl.f32](0.0)

        @tkw.reduction(M_qs, init_args=[
            dv_init,
            dk_init,
        ])
        def loop_q_seq_len(
            dv_prev: tkl.Register[B, K2_kvs, N_vd, tkl.f32],
            dk_prev: tkl.Register[B, K2_kvs, K1_qkd, tkl.f32],
        ):
            k_j = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            q_i = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)

            s_acc = tkl.Register[B, M_qs, K2_kvs, tkl.f32](0.0)
            log2e = tkl.Register[B, M_qs, K2_kvs, tkl.f16](1.44269504089)
            s_ij = tkw.mma(q_i, k_j, s_acc)
            tkw.write(s_ij, s, elements_per_thread=STORE_ELEMS_PER_THREAD)
            s_ij = tkw.permute(s_ij, [B, K2_kvs, M_qs])
            lse_i = tkw.read(lse, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            p_ij = tkw.exp2(log2e * (tkw.cast(s_ij, tkl.f16) - lse_i))
            tkw.write(p_ij, p, mapping=flip_k2_m_write_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

            do_i_for_dv = tkw.read(do, mapping=flip_m_n_read_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            dv_j = tkw.mma(p_ij, do_i_for_dv, dv_prev)
            
            v_j = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # This has to be loaded separately so we have N last
            do_i_for_dp = tkw.read(do, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            dp_acc = tkl.Register[B, K2_kvs, M_qs, tkl.f32](0.0)
            dp_ij = tkw.mma(v_j, do_i_for_dp, dp_acc)
            dp_ij = tkw.permute(dp_ij, [B, M_qs, K2_kvs])
            tkw.write(dp_ij, dp, elements_per_thread=STORE_ELEMS_PER_THREAD)

            D_i = tkw.read(D, elements_per_thread=1)
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
            ds_ij = p_ij_for_ds * dp_ij_sub
            tkw.write(ds_ij, ds, elements_per_thread=STORE_ELEMS_PER_THREAD)

            ds_ij_for_dk = tkw.read(ds, mapping=flip_m_k2_read_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            q_i_for_dk = tkw.read(q, mapping=flip_m_k1_read_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            dk_j = tkw.mma(ds_ij_for_dk, q_i_for_dk, dk_prev)

            k_j_for_dq = tkw.read(k, mapping=flip_k2_k1_read_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            dq_prev = tkw.read(dq, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            dq_i = tkw.mma(ds_ij, k_j_for_dq, tkw.cast(dq_prev, tkl.f32))
            tkw.write(dq_i, dq, elements_per_thread=STORE_ELEMS_PER_THREAD)
            return (
                dv_j,
                dk_j,
            )

        (
            dv_j,
            dk_j,
        ) = loop_q_seq_len
        tkw.write(
            tkw.cast(dv_j, tkl.f16), dv, elements_per_thread=STORE_ELEMS_PER_THREAD
        )
        tkw.write(
            tkw.cast(dk_j, tkl.f16), dk, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

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
        # for dq is thread safe. So force the distribution of K2 to be degenerate.
        BLOCK_K2: kv_seq_len,
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

    # q = device_randn(batch, q_seq_len, qk_head_dim)
    q = torch.full((batch, q_seq_len, qk_head_dim), 0.1, device=get_default_device())
    # q = torch.arange(
    #     (batch * q_seq_len * qk_head_dim), dtype=torch.float32, device=get_default_device()
    # ).reshape((batch, q_seq_len, qk_head_dim)) / (batch * q_seq_len * qk_head_dim)
    # q[0, :, (qk_head_dim//2):] = 0

    # k = device_randn(batch, kv_seq_len, qk_head_dim)
    k = torch.full((batch, kv_seq_len, qk_head_dim), 0.2, device=get_default_device())
    # k = torch.arange(
    #     (batch * kv_seq_len * qk_head_dim), dtype=torch.float32
    # ).reshape((batch, kv_seq_len, qk_head_dim)) / (batch * kv_seq_len * qk_head_dim)
    # k[0, :, (qk_head_dim//2):] = 0

    v = device_randn(batch, kv_seq_len, v_head_dim)
    # v = torch.arange(
    #     (batch * kv_seq_len * v_head_dim), dtype=torch.float32
    # ).reshape((batch, kv_seq_len, v_head_dim)) / 64

    # Save these to check for shenanigans later
    q_orig = q.clone()
    k_orig = k.clone()
    v_orig = v.clone()
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

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

    assert_close(o, o_ref)
    assert_close(dq, dq_ref)
    assert_close(dk, dk_ref)
    assert_close(dv, dv_ref)

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

    assert_close(dv, dv_ref)

    dp = torch.matmul(do, v.transpose(-1, -2))

    assert_close(dp, dp_ref)

    ds = device_zeros(batch, q_seq_len, kv_seq_len)
    for b in range(batch):
        for row in range(q_seq_len):
            p_row = p[b, row, :]
            jacobian = p_row.unsqueeze(0) * (
                to_default_device(torch.eye(kv_seq_len, dtype=torch.float16))
                - p_row.unsqueeze(-1)
            )

            ds[b, row, :] = torch.matmul(dp[b, row, :], jacobian.transpose(-1, -2))

    assert_close(ds, ds_ref)

    dq = torch.matmul(ds, k)
    assert_close(dq, dq_ref)

    dk = torch.matmul(ds.transpose(-1, -2), q)
    assert_close(dk, dk_ref)

    del ds, dp, dq, dk, dv, s, p

    # Flash attention implemented as loops
    o_loops = device_zeros(batch, q_seq_len, v_head_dim)
    lse_loops = device_zeros(batch, q_seq_len)
    attention_fwd_loops(q, k, v, o_loops, lse_loops)

    assert_close(o_loops, o_ref)

    # Make sure nothing's gone wonky here.
    assert_close(q, q_orig)
    assert_close(k, k_orig)
    assert_close(v, v_orig)
    assert_close(do, do_orig)

    D_loops = torch.sum(do * o_loops, -1)
    dq_loops = torch.zeros_like(q)
    dk_loops = torch.zeros_like(k)
    dv_loops = torch.zeros_like(v)

    attention_bwd_loops(q, k, v, do, lse_loops, D_loops, dq_loops, dk_loops, dv_loops)

    # Make sure nothing's gone wonky here.
    assert_close(q, q_orig)
    assert_close(k, k_orig)
    assert_close(v, v_orig)
    assert_close(do, do_orig)

    assert_close(dq_loops, dq_ref, atol=1e-4, rtol=1e-5)
    assert_close(dk_loops, dk_ref, atol=1e-4, rtol=1e-5)
    assert_close(dv_loops, dv_ref, atol=1e-4, rtol=1e-5)

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
        # "print_grid": True,
        # "print_ir_after": ["set_node_indices", "expand_graph", "first", "last"],
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

        mb_fwd = attention_fwd(
            q,
            k,
            v.transpose(-1, -2),
            s,
            output,
            lse
        )

        if dump_generated_mlir:
            filename = f"wave_attention_fwd_{'x'.join(map(str, shape))}.mlir"
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
        "print_ir_after": ["first", "last"],
        # "print_ir_before": ["first", "set_node_indices"],
        # "print_signature": True,
        # "print_pretty_mlir": True,
        "print_indices": True,
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
        ds = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
        dp = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
        dp_sub = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)

        expanded_lse = lse.reshape((batch, q_seq_len, 1)).expand(batch, q_seq_len, kv_seq_len)

        mb_bwd = attention_bwd(
            q,
            k,
            v,
            do,
            expanded_lse.transpose(-1, -2),
            D,
            dq,
            dk,
            dv,
            s,
            p,
            ds,
            dp,
            dp_sub,
        )
            
        if dump_generated_mlir:
            filename = f"wave_attention_bwd_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_bwd.module_op.get_asm())
            print(f"IR dumped to {filename}")

        # Make sure nothing's gone wonky here.
        assert_close(q, q_orig.to(torch.float16))
        assert_close(k, k_orig.to(torch.float16))
        assert_close(v, v_orig.to(torch.float16))
        assert_close(do, do_orig.to(torch.float16))

        assert_close(dv, dv_ref, atol=5e-2, rtol=1e-2)

        assert_close(dp, dp_ref, atol=5e-2, rtol=1e-2)
        assert_close(dp_sub, dp_sub_ref, atol=5e-2, rtol=1e-2)

        # I don't actually understand why this way of calculating ds works at all.
        assert_close(ds_ref, p_ref*dp_sub_ref, atol=5e-2, rtol=1e-2)
        
        # print_small_tensor(ds_ref, "ds_ref")
        # print_small_tensor(ds, "ds")

        assert_close(ds, ds_ref, atol=5e-2, rtol=1e-2)
        assert_close(dk, dk_ref, atol=5e-2, rtol=1e-2)

        assert_close(dq, dq_ref, atol=5e-2, rtol=1e-2)



def testReproWriteInLoopBug():
    mfma_variant = MMAType.F32_16x16x16_F16
    shape = (16, 32, 16)
    q_seq_len, qk_head_dim, kv_seq_len = shape
    M = tkl.sym.M
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2

    BLOCK_K2 = tkl.sym.BLOCK_K2
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, M, 0)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
        )
    ]

    @tkw.wave(constraints)
    def attention_fwd(
        q: tkl.Memory[M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[K2, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        s: tkl.Memory[M, K2, GLOBAL_ADDRESS_SPACE, tkl.f32],
        max: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        init_m = tkl.Register[M, tkl.f32](-1e6)

        @tkw.reduction(K2, init_args=[init_m])
        def repeat(m_prev: tkl.Register[M, tkl.f32]) -> tkl.Register[M, tkl.f32]:
            s_acc = tkl.Register[K2, M, tkl.f32](0.0)
            q_i = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_j = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            s_ij = tkw.mma(k_j, q_i, s_acc)
            s_ij = tkw.permute(s_ij, target_shape=[M, K2])
            tkw.write(s_ij, s, elements_per_thread=STORE_ELEMS_PER_THREAD)
            m_ij = tkw.max(s_ij, m_prev, dim=K2)
            return m_ij
        tkw.write(repeat, max, elements_per_thread=1)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 4,
        BLOCK_K2: 16,
        M: q_seq_len,
        K1: qk_head_dim,
        K2: kv_seq_len,
    }

    # q = device_randn(batch, q_seq_len, qk_head_dim)
    q = torch.full((q_seq_len, qk_head_dim), 0.1, device=get_default_device())
    # q = torch.arange(
    #     (batch * q_seq_len * qk_head_dim), dtype=torch.float32, device=get_default_device()
    # ).reshape((batch, q_seq_len, qk_head_dim)) / (batch * q_seq_len * qk_head_dim)
    # q[0, :, (qk_head_dim//2):] = 0

    # k = device_randn(batch, kv_seq_len, qk_head_dim)
    k = torch.full((kv_seq_len, qk_head_dim), 0.2, device=get_default_device())
    # k = torch.arange(
    #     (batch * kv_seq_len * qk_head_dim), dtype=torch.float32
    # ).reshape((batch, kv_seq_len, qk_head_dim)) / (batch * kv_seq_len * qk_head_dim)
    # k[0, :, (qk_head_dim//2):] = 0

    s_ref = torch.matmul(q, k.transpose(-1, -2))
    m_ref = torch.max(s_ref, dim=-1)[0]

    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    # config["print_ir_after_all"] = True
    compile_config = {
        "waves_per_eu": 2,
        "denorm_fp_math_f32": "preserve-sign",
        "print_grid": True,
        "print_ir_after": ["set_node_indices", "expand_graph", "first", "last"],
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
    ):
        m = device_zeros(q_seq_len)
        s = device_zeros(q_seq_len, kv_seq_len)

        mb_fwd = attention_fwd(
            q.to(torch.float16),
            k.to(torch.float16),
            s,
            m
        )
        if dump_generated_mlir:
            filename = f"wave_matmul_in_reduction_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_fwd.module_op.get_asm())
            print(f"IR dumped to {filename}")
        
        assert_close(m, m_ref, atol=5e-2, rtol=1e-2)
        assert_close(s, s_ref, atol=5e-2, rtol=1e-2)


def testReproWriteAlongUnconstrainedDimension():
    shape = (16, 32, 16, 16)
    q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
    mfma_variant = MMAType.F32_16x16x16_F16
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2

    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(K2, K2, 0),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
        )
    ]

    @tkw.wave(constraints)
    def attention_bwd(
        q: tkl.Memory[M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[K2, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        do: tkl.Memory[N, M, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dv: tkl.Memory[K2, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        dv_init = tkl.Register[K2, N, tkl.f32](0.0)

        k_j = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        q_i = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)

        s_acc = tkl.Register[M, K2, tkl.f32](0.0)
        s_ij = tkw.mma(q_i, k_j, s_acc)
        s_ij = tkw.permute(s_ij, [K2, M])

        do_i = tkw.read(do, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        dv_j = tkw.mma(tkw.cast(s_ij, tkl.f16), do_i, dv_init)
            
        tkw.write(dv_j, dv, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        M: q_seq_len,
        N: v_head_dim,
        K1: qk_head_dim,
        K2: kv_seq_len,
    }

    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    # config["print_ir_after_all"] = True
    compile_config = {
        "waves_per_eu": 2,
        "denorm_fp_math_f32": "preserve-sign",
        "print_ir_after": ["first", "last", "set_node_indices", "expand_graph"],
        # "print_ir_before": ["first", "set_node_indices"],
        "print_signature": True,
        "print_pretty_mlir": True,
        "print_indices": True,
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
    ):
        
        torch.manual_seed(0)
        q = torch.full((q_seq_len, qk_head_dim), 0.1, device=get_default_device(), dtype=torch.float16)
        k = torch.full((kv_seq_len, qk_head_dim), 0.2, device=get_default_device(), dtype=torch.float16)
        do = torch.full((q_seq_len, v_head_dim), 0.3, device=get_default_device(), dtype=torch.float16)

        s_ref = torch.matmul(q, k.transpose(-1, -2))
        dv_ref = torch.matmul(s_ref.transpose(-1, -2), do)

        dv = device_zeros(kv_seq_len, v_head_dim)
        mb_bwd = attention_bwd(
            q,
            k,
            do.transpose(-1, -2),
            dv,
        )
            
        if dump_generated_mlir:
            filename = f"wave_chained_gemm_undistributed_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_bwd.module_op.get_asm())
            print(f"IR dumped to {filename}")

        assert_close(dv, dv_ref.to(torch.float32), atol=1e-3, rtol=1e-4)

