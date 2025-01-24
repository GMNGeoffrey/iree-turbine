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
    get_default_device,
)
from iree.turbine.kernel.wave.constraints import MMAType
from ..common.utils import (
    require_e2e,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
from torch.testing import assert_close

big_shapes = [
    # (40, 1024, 64, 64, 1024),
]

shapes_16x16x16 = [
    (2, 64, 128, 32, 256),
    # (40, 1024, 64, 64, 1024),
    # TODO(#374): These are additional demonstrations of this issue.
    # (1, 16, 16, 8, 16),
    # (1, 16, 16, 64, 16),
    (1, 16, 16, 16, 32),
    (1, 16, 16, 32, 16),
    (1, 16, 32, 16, 16),
    (1, 32, 16, 16, 16),
    (2, 16, 16, 16, 16),
    (1, 16, 16, 16, 16),
]

shapes_32x32x32 = [tuple(dim if i == 0 else 2*dim for i, dim in enumerate(shape)) for shape in shapes_16x16x16]

def get_param_id(val):
    if isinstance(val, tuple) and all(isinstance(el, int) for el in val):
        return "x".join(str(el) for el in val)
    elif isinstance(val, MMAType):
        return f"MMA_{val.name}"

param_mfma_shape = pytest.mark.parametrize(
    "mfma_variant,shape",
    [(MMAType.F32_32x32x8_F16, shape) for shape in shapes_32x32x32 + big_shapes]
    + [(MMAType.F32_16x16x16_F16, shape) for shape in shapes_16x16x16 + big_shapes],
    ids=get_param_id,
)


def attention_flash_fwd_loops(q, k, v):
    B, M_qs, K1_qkd = q.shape
    _, K2_kvs, _ = k.shape
    assert k.shape == (B, K2_kvs, K1_qkd)
    _, _, N_vd = v.shape
    assert v.shape == (B, K2_kvs, N_vd)

    o = device_zeros(B, M_qs, N_vd)
    lse = device_zeros(B, M_qs)
    s = device_zeros(B, M_qs, K2_kvs)

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
            end_m = start_m + BLOCK_M_Br
            q_i = q[batch, start_m:end_m, :]
            assert q_i.shape == (BLOCK_M_Br, K1_qkd)
            o_i = device_zeros(BLOCK_M_Br, N_vd)
            l_i = device_zeros(BLOCK_M_Br)
            m_i = to_default_device(torch.full((BLOCK_M_Br,), -torch.inf))
            for start_k2 in range(0, K2_kvs, BLOCK_K2_Bc):
                end_k2 = start_k2 + BLOCK_K2_Bc
                k_j = k[batch, start_k2:end_k2, :]
                assert k_j.shape == (BLOCK_K2_Bc, K1_qkd)
                v_j = v[batch, start_k2:end_k2, :]
                assert v_j.shape == (BLOCK_K2_Bc, N_vd)
                s_ij = torch.matmul(q_i, k_j.transpose(-1, -2))
                assert s_ij.shape == (BLOCK_M_Br, BLOCK_K2_Bc)
                s[batch, start_m:end_m, start_k2:end_k2] = s_ij
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
            o[batch, start_m:end_m, :] = o_i
            lse[batch, start_m:end_m] = L_i

    return o, lse, s


def attention_flash_bwd_loops(
    q,
    k,
    v,
    do,
    o,
    lse,
):
    B, M_qs, K1_qkd = q.shape
    _, K2_kvs, _ = k.shape
    assert k.shape == (B, K2_kvs, K1_qkd)
    _, _, N_vd = v.shape
    assert v.shape == (B, K2_kvs, N_vd)
    assert do.shape == (B, M_qs, N_vd)
    assert lse.shape == (B, M_qs)

    D = torch.sum(do * o, -1)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    ds = device_zeros(B, M_qs, K2_kvs)
    dp = torch.zeros_like(ds)

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
        for start_k2 in range(0, K2_kvs, BLOCK_K2_Bc):
            end_k2 = start_k2 + BLOCK_K2_Bc
            k_j = k[batch, start_k2:end_k2, :]
            assert k_j.shape == (BLOCK_K2_Bc, K1_qkd)
            v_j = v[batch, start_k2:end_k2, :]
            assert v_j.shape == (BLOCK_K2_Bc, N_vd)

            dv_j = device_zeros(BLOCK_K2_Bc, N_vd)
            dk_j = device_zeros(BLOCK_K2_Bc, K1_qkd)

            for start_m in range(0, M_qs, BLOCK_M_Br):
                end_m = start_m + BLOCK_M_Br
                q_i = q[batch, start_m:end_m, :]
                assert q_i.shape == (BLOCK_M_Br, K1_qkd)
                do_i = do[batch, start_m:end_m, :]
                assert do_i.shape == (BLOCK_M_Br, N_vd)
                dq_i = dq[batch, start_m:end_m, :]
                assert dq_i.shape == (BLOCK_M_Br, K1_qkd)
                lse_i = lse[batch, start_m:end_m]
                assert lse_i.shape == (BLOCK_M_Br,)
                D_i = D[batch, start_m:end_m]
                assert D_i.shape == (BLOCK_M_Br,)
                s_ij = torch.matmul(q_i, k_j.transpose(-1, -2))
                assert s_ij.shape == (BLOCK_M_Br, BLOCK_K2_Bc)
                p_ij = torch.exp(s_ij - lse_i.reshape(BLOCK_M_Br, 1))
                assert p_ij.shape == (BLOCK_M_Br, BLOCK_K2_Bc)
                dv_j += torch.matmul(p_ij.transpose(-1, -2), do_i)
                dp_ij = torch.matmul(do_i, v_j.transpose(-1, -2))
                assert dp_ij.shape == (BLOCK_M_Br, BLOCK_K2_Bc)
                dp[batch, start_m:end_m, start_k2:end_k2] = dp_ij
                ds_ij = p_ij * (dp_ij - D_i.reshape(BLOCK_M_Br, 1))
                assert ds_ij.shape == (BLOCK_M_Br, BLOCK_K2_Bc)
                ds[batch, start_m:end_m, start_k2:end_k2] = ds_ij
                dq_i += torch.matmul(ds_ij, k_j)
                dk_j += torch.matmul(ds_ij.transpose(-1, -2), q_i)

                dv[batch, start_k2:end_k2, :] = dv_j

            dk[batch, start_k2:end_k2, :] = dk_j

    return dq, dk, dv, ds, dp


def attention_torch_builtin_ref(q, k, v, do, scale=1):
    q = q.clone().requires_grad_()
    v = v.clone().requires_grad_()
    k = k.clone().requires_grad_()

    o = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, scale=1
    )

    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    o.backward(do)

    dq = q.grad.detach()
    dk = k.grad.detach()
    dv = v.grad.detach()
    return o, dq, dk, dv


def attention_torch_ops_ref(q, k, v, do, scale=1):
    """Attention computed with individual torch operations so we can get intermediates."""
    q = q.clone().requires_grad_()
    v = v.clone().requires_grad_()
    k = k.clone().requires_grad_()

    s = scale * torch.matmul(q, k.transpose(-1, -2))
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
    ds = s.grad.detach()
    dp = p.grad.detach()
    o = o.detach()
    s = s.detach()
    p = p.detach()

    return o, dq, dk, dv, s, p, ds, dp


def attention_bwd_jacobian(q, k, v, do):
    # TODO: add scale
    # Manual backward pass for normal attention
    s = torch.matmul(q, k.transpose(-1, -2))
    p = torch.softmax(s, dim=-1)

    dv = torch.matmul(p.transpose(-1, -2), do)
    dp = torch.matmul(do, v.transpose(-1, -2))
    ds = torch.zeros_like(s)

    batch, q_seq_len, kv_seq_len = s.shape
    for b in range(batch):
        for row in range(q_seq_len):
            p_row = p[b, row, :]
            jacobian = p_row.unsqueeze(0) * (
                to_default_device(torch.eye(kv_seq_len, dtype=torch.float16))
                - p_row.unsqueeze(-1)
            )

            ds[b, row, :] = torch.matmul(dp[b, row, :], jacobian.transpose(-1, -2))

    dq = torch.matmul(ds, k)
    dk = torch.matmul(ds.transpose(-1, -2), q)

    return dq, dk, dv, ds, dp


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
        vec_size = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(WAVES_PER_BLOCK_M, WAVES_PER_BLOCK_N, WAVES_PER_BLOCK_B),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M_qs: vec_size, N_vd: vec_size},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_n_m_write_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, N_vd: j, M_qs: k},
        outputs={B: i, M_qs: k, N_vd: j},
    )

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
        init_m = tkl.Register[B, M_qs, tkl.f32](-1e6)
        init_l = tkl.Register[B, M_qs, tkl.f32](0.0)
        init_o = tkl.Register[B, N_vd, M_qs, tkl.f32](0.0)

        @tkw.reduction(K2_kvs, init_args=[init_m, init_l, init_o])
        def repeat(
            m_prev: tkl.Register[B, M_qs, tkl.f32],
            l_prev: tkl.Register[B, M_qs, tkl.f32],
            o_prev: tkl.Register[B, N_vd, M_qs, tkl.f32],
        ) -> tuple[
            tkl.Register[B, M_qs, tkl.f32],
            tkl.Register[B, M_qs, tkl.f32],
            tkl.Register[B, N_vd, M_qs, tkl.f32],
        ]:
            s_acc = tkl.Register[B, K2_kvs, M_qs, tkl.f32](0.0)
            log2e = tkl.Register[B, M_qs, tkl.f32](1.44269504089)
            q_i = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_j = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
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
            return m_ij, l_ij, o_ij

        log2e = tkl.Register[B, M_qs, tkl.f32](1.44269504089)
        m_i, l_i, o_i = repeat
        o_i = tkw.reciprocal(l_i) * o_i
        tkw.write(
            tkw.cast(o_i, tkl.f16),
            o,
            mapping=flip_n_m_write_mapping,
            elements_per_thread=STORE_ELEMS_PER_THREAD,
        )
        lse_i = m_i + (tkw.log2(l_i) / log2e)
        tkw.write(tkw.cast(lse_i, tkl.f16), lse, elements_per_thread=1)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M_r_qs: vec_size,
        BLOCK_N_vd: vec_size,
        BLOCK_K2_c_kvs: vec_size,
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
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K1 = tkl.sym.BLOCK_K1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    elif mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(K2_kvs, BLOCK_K2, 0),
        # Degenerate distribution seems to fix some bugs
        tkw.WorkgroupConstraint(K1_qkd, BLOCK_K1, 1),
        tkw.WorkgroupConstraint(N_vd, BLOCK_N, 2),
        # Can only have 3 dimensions distributed in actual blocks or the
        # compiler tries to index too far into waves_per_block (and if that is
        # made longer there's just a fatal crash), so batch dimension needs to
        # be last.
        tkw.WorkgroupConstraint(B, BLOCK_B, 3),
        tkw.TilingConstraint(M_qs, BLOCK_M),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, N_vd: vec_size},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_k2_m_write_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, K2_kvs: j, M_qs: k},
        outputs={B: i, M_qs: k, K2_kvs: j},
    )

    flip_m_n_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, N_vd: j},
        outputs={B: i, N_vd: j, M_qs: k},
    )
    flip_m_k1_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, K1_qkd: j},
        outputs={B: i, K1_qkd: j, M_qs: k},
    )
    flip_m_k2_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, K2_kvs: j},
        outputs={B: i, K2_kvs: j, M_qs: k},
    )
    flip_k2_k1_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, K2_kvs: k, K1_qkd: j},
        outputs={B: i, K1_qkd: j, K2_kvs: k},
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

        @tkw.reduction(
            M_qs,
            init_args=[
                dv_init,
                dk_init,
            ],
        )
        def loop_q_seq_len(
            dv_prev: tkl.Register[B, K2_kvs, N_vd, tkl.f32],
            dk_prev: tkl.Register[B, K2_kvs, K1_qkd, tkl.f32],
        ):
            k_j = tkw.read(k, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            q_i = tkw.read(q, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)

            s_acc = tkl.Register[B, M_qs, K2_kvs, tkl.f32](0.0)
            log2e = tkl.Register[B, M_qs, K2_kvs, tkl.f16](1.44269504089)
            s_ij = tkw.mma(q_i, k_j, s_acc)
            tkw.write(s_ij, s, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            s_ij = tkw.permute(s_ij, [B, K2_kvs, M_qs])
            lse_i = tkw.read(lse, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            p_ij = tkw.exp2(log2e * (tkw.cast(s_ij, tkl.f16) - lse_i))
            tkw.write(
                p_ij,
                p,
                mapping=flip_k2_m_write_mapping,
                elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD,
            )

            do_i_for_dv = tkw.read(
                do,
                mapping=flip_m_n_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            dv_j = tkw.mma(p_ij, do_i_for_dv, dv_prev)

            v_j = tkw.read(v, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            # This has to be loaded separately so we have N last
            do_i_for_dp = tkw.read(do, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            dp_acc = tkl.Register[B, K2_kvs, M_qs, tkl.f32](0.0)
            dp_ij = tkw.mma(v_j, do_i_for_dp, dp_acc)
            dp_ij = tkw.permute(dp_ij, [B, M_qs, K2_kvs])
            tkw.write(dp_ij, dp, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            D_i = tkw.read(D, elements_per_thread=1)
            dp_ij_sub = tkw.cast(dp_ij, tkl.f16) - tkw.broadcast(D_i, [B, M_qs, K2_kvs])
            tkw.write(dp_ij_sub, dp_sub, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            # Just multiplying p_ij * dp_ij_sub breaks the previously calculated
            # dp. We have to load back p in the required layout.
            p_ij_for_ds = tkw.read(p, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            ds_ij = p_ij_for_ds * dp_ij_sub
            tkw.write(ds_ij, ds, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            ds_ij_for_dk = tkw.read(
                ds,
                mapping=flip_m_k2_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            q_i_for_dk = tkw.read(
                q,
                mapping=flip_m_k1_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            dk_j = tkw.mma(ds_ij_for_dk, q_i_for_dk, dk_prev)

            k_j_for_dq = tkw.read(
                k,
                mapping=flip_k2_k1_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            dq_prev = tkw.read(dq, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            dq_i = tkw.mma(ds_ij, k_j_for_dq, tkw.cast(dq_prev, tkl.f32))
            tkw.write(dq_i, dq, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            return (
                dv_j,
                dk_j,
            )

        (
            dv_j,
            dk_j,
        ) = loop_q_seq_len
        tkw.write(
            tkw.cast(dv_j, tkl.f16), dv, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD
        )
        tkw.write(
            tkw.cast(dk_j, tkl.f16), dk, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        MFMA_INPUT_ELS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        MFMA_OUTPUT_ELS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: vec_size,
        # TODO(#381) and TODO(#384): We need the degenerate tiling to make Wave
        # recognize the dims, but it needs to be at least the vector size or we
        # get errors about tile size being divisible by vector size.
        BLOCK_N: max(v_head_dim, vec_size),
        BLOCK_K1: max(qk_head_dim, vec_size),
        # TODO: Not actually what we want probably, but I couldn't get nested
        # loops to work (the only option is a reduction, it insists on having
        # args to reduce, and then it can't figure out the vector shapes for
        # them if I give it dummy args) and I don't think our read/write scheme
        # for dq is thread safe. So force the distribution of K2 to be degenerate.
        BLOCK_K2: max(kv_seq_len, vec_size),
        B: batch,
        M_qs: q_seq_len,
        N_vd: v_head_dim,
        K1_qkd: qk_head_dim,
        K2_kvs: kv_seq_len,
    }

    return attention_bwd, hyperparams


def get_attention_bwd_dv_kernel(
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
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K1 = tkl.sym.BLOCK_K1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    elif mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(K2_kvs, BLOCK_K2, 0),
        # Degenerate distribution seems to fix some bugs
        tkw.WorkgroupConstraint(K1_qkd, BLOCK_K1, 1),
        tkw.WorkgroupConstraint(N_vd, BLOCK_N, 2),
        # Can only have 3 dimensions distributed in actual blocks or the
        # compiler tries to index too far into waves_per_block (and if that is
        # made longer there's just a fatal crash), so batch dimension needs to
        # be last.
        tkw.WorkgroupConstraint(B, BLOCK_B, 3),
        tkw.TilingConstraint(M_qs, BLOCK_M),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, N_vd: vec_size},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)

    flip_m_n_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, N_vd: j},
        outputs={B: i, N_vd: j, M_qs: k},
    )

    @tkw.wave(constraints)
    def attention_bwd(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        do: tkl.Memory[B, M_qs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        lse: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dv: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        p: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        dv_init = tkl.Register[B, K2_kvs, N_vd, tkl.f32](0.0)

        @tkw.reduction(M_qs, init_args=[dv_init])
        def loop_q_seq_len(dv_prev: tkl.Register[B, K2_kvs, N_vd, tkl.f32]):
            k_j = tkw.read(k, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            q_i = tkw.read(q, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)

            s_acc = tkl.Register[B, M_qs, K2_kvs, tkl.f32](0.0)
            log2e = tkl.Register[B, M_qs, K2_kvs, tkl.f16](1.44269504089)
            s_ij = tkw.mma(q_i, k_j, s_acc)
            tkw.write(s_ij, s, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            # a no-op permute here gets past a compiler error resolving node
            # indices. I think it just hides the K1 dimension from the index.
            s_ij = tkw.permute(s_ij, [B, M_qs, K2_kvs])
            lse_i = tkw.read(lse, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            p_ij = tkw.exp2(log2e * (tkw.cast(s_ij, tkl.f16) - lse_i))
            tkw.write(p_ij, p, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            p_ij = tkw.permute(p_ij, [B, K2_kvs, M_qs])

            do_i = tkw.read(
                do,
                mapping=flip_m_n_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            dv_j = tkw.mma(p_ij, do_i, dv_prev)
            return dv_j

        tkw.write(
            tkw.cast(loop_q_seq_len, tkl.f16),
            dv,
            elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD,
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        MFMA_INPUT_ELS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        MFMA_OUTPUT_ELS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: vec_size,
        # TODO(#381) and TODO(#384): We need the degenerate tiling to make Wave
        # recognize the dims, but it needs to be at least the vector size or we
        # get errors about tile size being divisible by vector size.
        BLOCK_N: max(v_head_dim, vec_size),
        BLOCK_K1: max(qk_head_dim, vec_size),
        BLOCK_K2: vec_size, # distributing this speeds things up a ton
        B: batch,
        M_qs: q_seq_len,
        N_vd: v_head_dim,
        K1_qkd: qk_head_dim,
        K2_kvs: kv_seq_len,
    }

    return attention_bwd, hyperparams


def get_attention_bwd_dk_kernel(
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
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K1 = tkl.sym.BLOCK_K1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.MFMA_INPUT_ELS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.MFMA_OUTPUT_ELS_PER_THREAD

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    elif mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(K2_kvs, BLOCK_K2, 0),
        # Degenerate distribution seems to fix some bugs
        tkw.WorkgroupConstraint(K1_qkd, BLOCK_K1, 1),
        tkw.WorkgroupConstraint(N_vd, BLOCK_N, 2),
        # Can only have 3 dimensions distributed in actual blocks or the
        # compiler tries to index too far into waves_per_block (and if that is
        # made longer there's just a fatal crash), so batch dimension needs to
        # be last.
        tkw.WorkgroupConstraint(B, BLOCK_B, 3),
        tkw.TilingConstraint(M_qs, BLOCK_M),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_k2_m_write_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, K2_kvs: j, M_qs: k},
        outputs={B: i, M_qs: k, K2_kvs: j},
    )

    flip_m_n_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, N_vd: j},
        outputs={B: i, N_vd: j, M_qs: k},
    )
    flip_m_k1_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, K1_qkd: j},
        outputs={B: i, K1_qkd: j, M_qs: k},
    )
    flip_m_k2_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, K2_kvs: j},
        outputs={B: i, K2_kvs: j, M_qs: k},
    )
    flip_k2_k1_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, K2_kvs: k, K1_qkd: j},
        outputs={B: i, K1_qkd: j, K2_kvs: k},
    )

    @tkw.wave(constraints)
    def attention_bwd_dk(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        do: tkl.Memory[B, M_qs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        lse: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        D: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dk: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        p: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        ds: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dp: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        dp_sub: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        dk_init = tkl.Register[B, K2_kvs, K1_qkd, tkl.f32](0.0)

        @tkw.reduction(M_qs, init_args=[dk_init])
        def loop_q_seq_len(dk_prev: tkl.Register[B, K2_kvs, K1_qkd, tkl.f32]):
            k_j = tkw.read(k, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            q_i = tkw.read(q, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)

            s_acc = tkl.Register[B, M_qs, K2_kvs, tkl.f32](0.0)
            log2e = tkl.Register[B, M_qs, K2_kvs, tkl.f16](1.44269504089)
            s_ij = tkw.mma(q_i, k_j, s_acc)
            tkw.write(s_ij, s, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            # a no-op permute here gets past a compiler error resolving node
            # indices. I think it just hides the K1 dimension from the index.
            s_ij = tkw.permute(s_ij, [B, M_qs, K2_kvs])
            lse_i = tkw.read(lse, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            p_ij = tkw.exp2(log2e * (tkw.cast(s_ij, tkl.f16) - lse_i))
            tkw.write(p_ij, p, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            p_ij = tkw.permute(p_ij, [B, K2_kvs, M_qs])

            dp_acc = tkl.Register[B, M_qs, K2_kvs, tkl.f32](0.0)
            v_j = tkw.read(v, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            do_i = tkw.read(do, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            dp_ij = tkw.mma(do_i, v_j, dp_acc)
            # This no-op permute fixes a compiler error by hiding the N index of
            # the mma from the cast that uses it. Otherwise, the cast operation
            # fails to update the op it uses during expansion.
            dp_ij = tkw.permute(dp_ij, [B, M_qs, K2_kvs])
            tkw.write(dp_ij, dp, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            D_i = tkw.read(D, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            dp_ij_sub = tkw.cast(dp_ij, tkl.f16) - D_i
            tkw.write(dp_ij_sub, dp_sub, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            dp_ij_sub = tkw.permute(dp_ij_sub, [B, K2_kvs, M_qs])

            ds_ij = p_ij * dp_ij_sub
            tkw.write(ds_ij, ds, mapping=flip_k2_m_write_mapping, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            q_i_for_dk = tkw.read(
                q,
                mapping=flip_m_k1_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            dk_j = tkw.mma(ds_ij, q_i_for_dk, dk_prev)

            return dk_j

        tkw.write(
            tkw.cast(loop_q_seq_len, tkl.f16),
            dk,
            elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD,
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        MFMA_INPUT_ELS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        MFMA_OUTPUT_ELS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: vec_size,
        # TODO(#381) and TODO(#384): We need the degenerate tiling to make Wave
        # recognize the dims, but it needs to be at least the vector size or we
        # get errors about tile size being divisible by vector size.
        BLOCK_N: max(v_head_dim, vec_size),
        BLOCK_K1: max(qk_head_dim, vec_size),
        BLOCK_K2: vec_size,
        B: batch,
        M_qs: q_seq_len,
        N_vd: v_head_dim,
        K1_qkd: qk_head_dim,
        K2_kvs: kv_seq_len,
    }

    return attention_bwd_dk, hyperparams



def get_attention_bwd_dq_kernel(
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
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K1 = tkl.sym.BLOCK_K1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    elif mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(K2_kvs, BLOCK_K2, 0),
        # Degenerate distribution seems to fix some bugs
        tkw.WorkgroupConstraint(K1_qkd, BLOCK_K1, 1),
        tkw.WorkgroupConstraint(N_vd, BLOCK_N, 2),
        # Can only have 3 dimensions distributed in actual blocks or the
        # compiler tries to index too far into waves_per_block (and if that is
        # made longer there's just a fatal crash), so batch dimension needs to
        # be last.
        tkw.WorkgroupConstraint(B, BLOCK_B, 3),
        tkw.TilingConstraint(M_qs, BLOCK_M),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, N_vd: vec_size},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_k2_m_write_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, K2_kvs: j, M_qs: k},
        outputs={B: i, M_qs: k, K2_kvs: j},
    )

    flip_m_n_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, N_vd: j},
        outputs={B: i, N_vd: j, M_qs: k},
    )
    flip_m_k1_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, K1_qkd: j},
        outputs={B: i, K1_qkd: j, M_qs: k},
    )
    flip_m_k2_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, K2_kvs: j},
        outputs={B: i, K2_kvs: j, M_qs: k},
    )
    flip_k2_k1_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, K2_kvs: k, K1_qkd: j},
        outputs={B: i, K1_qkd: j, K2_kvs: k},
    )

    @tkw.wave(constraints)
    def attention_bwd_dq(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        do: tkl.Memory[B, M_qs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        lse: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # D: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # dq: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        s_sub: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        p: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # ds: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dp: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        # dp_sub: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        # TODO(#364): Workaround for missing non-reduction loop. This needs to
        # have only dimensions that are in the vector shapes or it doesn't have
        # vector shapes for its indexing dims.
        dummy_init = tkl.Register[B, tkl.f16](0.0)

        @tkw.reduction(M_qs, init_args=[dummy_init])
        def loop_q_seq_len(dummy_prev: tkl.Register[B, tkl.f16]):
            k_j = tkw.read(k, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            q_i = tkw.read(q, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)

            s_acc = tkl.Register[B, K2_kvs, M_qs, tkl.f32](0.0)
            s_ij = tkw.mma(k_j, q_i, s_acc)
            # permuting and then writing without a mapping breaks whichever of s
            # and dp is used later in the kernel iff we multiply p_ij and
            # dp_ij_sub to compute ds_ij.
            tkw.write(s_ij, s, mapping=flip_k2_m_write_mapping, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            s_ij = tkw.permute(s_ij, [B, M_qs, K2_kvs])
            lse_i = tkw.read(lse, elements_per_thread=1)
            s_ij_sub = tkw.cast(s_ij, tkl.f16) - lse_i
            tkw.write(s_ij_sub, s_sub, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            log2e = tkl.Register[B, K2_kvs, M_qs, tkl.f16](1.44269504089)
            log2e = tkw.permute(log2e, [B, M_qs, K2_kvs])
            p_ij = tkw.exp2(log2e * s_ij_sub)
            tkw.write(p_ij, p, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            v_j = tkw.read(v, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            do_i = tkw.read(do, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            dp_acc = tkl.Register[B, K2_kvs, M_qs, tkl.f32](0.0)
            dp_ij = tkw.mma(v_j, do_i, dp_acc)
            # permuting and then writing without a mapping breaks whichever of s
            # and dp is used later in the kernel iff we multiply p_ij and
            # dp_ij_sub to compute ds_ij.
            tkw.write(dp_ij, dp, mapping=flip_k2_m_write_mapping, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            # dp_ij = tkw.permute(dp_ij, [B, M_qs, K2_kvs])
            # D_i = tkw.read(D, elements_per_thread=1)
            # dp_ij_sub = tkw.cast(dp_ij, tkl.f16) - D_i
            # tkw.write(dp_ij_sub, dp_sub, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            # ds_ij = p_ij * dp_ij_sub
            # tkw.write(ds_ij, ds, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            # # permuting doesn't work here. We need to read it in directly in the correct layout.
            # # k_j_for_dq = tkw.permute(k_j, [B, K1_qkd, K2_kvs])
            # k_j_for_dq = tkw.read(
            #     k,
            #     mapping=flip_k2_k1_read_mapping,
            #     elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            # )
            # dq_prev = tkw.read(dq, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            # dq_i = tkw.mma(ds_ij, k_j_for_dq, tkw.cast(dq_prev, tkl.f32)) # B x M x K2 @ B x K1 x K2
            # tkw.write(dq_i, dq, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            return dummy_prev


    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        MFMA_INPUT_ELS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        MFMA_OUTPUT_ELS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: vec_size,
        # TODO(#381) and TODO(#384): We need the degenerate tiling to make Wave
        # recognize the dims, but it needs to be at least the vector size or we
        # get errors about tile size being divisible by vector size.
        BLOCK_N: max(v_head_dim, vec_size),
        BLOCK_K1: max(qk_head_dim, vec_size),
        # TODO: Not actually what we want probably, but I couldn't get nested
        # loops to work (the only option is a reduction, it insists on having
        # args to reduce, and then it can't figure out the vector shapes for
        # them if I give it dummy args) and I don't think our read/write scheme
        # for dq is thread safe. So force the distribution of K2 to be degenerate.
        BLOCK_K2: max(kv_seq_len, vec_size),
        B: batch,
        M_qs: q_seq_len,
        N_vd: v_head_dim,
        K1_qkd: qk_head_dim,
        K2_kvs: kv_seq_len,
    }

    return attention_bwd_dq, hyperparams



@require_e2e
@param_mfma_shape
def testAttentionBackwardParts(mfma_variant: MMAType, shape: tuple[int], request):
    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
    small_shape = math.prod(shape) < 500_000
    # Our float tolerances here are really bad on mi210
    tols = dict(atol=3e-2, rtol=5e-3) if small_shape else dict(atol=2e-1, rtol=5e-2)

    torch.manual_seed(0)
    # doing all this manual stuff in float32 or we lose too much precision. We
    # generate the random numbers in float16 though so we at least start with
    # something that is representable in that.

    q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16).to(
        torch.float32
    )
    # q = torch.full((batch, q_seq_len, qk_head_dim), 0.1, device=get_default_device())
    # q = torch.arange(
    #     (batch * q_seq_len * qk_head_dim), dtype=torch.float32, device=get_default_device()
    # ).reshape((batch, q_seq_len, qk_head_dim)) / (batch * q_seq_len * qk_head_dim)
    # q[0, :, (qk_head_dim//2):] = 0

    k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16).to(
        torch.float32
    )
    # k = torch.full((batch, kv_seq_len, qk_head_dim), 0.2, device=get_default_device())
    # k = torch.arange(
    #     (batch * kv_seq_len * qk_head_dim), dtype=torch.float32
    # ).reshape((batch, kv_seq_len, qk_head_dim)) / (batch * kv_seq_len * qk_head_dim)
    # k[0, :, (qk_head_dim//2):] = 0

    v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16).to(
        torch.float32
    )
    # v = torch.full((batch, kv_seq_len, v_head_dim), 0.3, device=get_default_device())
    # v = torch.arange(
    #     (batch * kv_seq_len * v_head_dim), dtype=torch.float32
    # ).reshape((batch, kv_seq_len, v_head_dim)) / 64

    # Fake derivative for whatever the loss function might be
    # do = to_default_device(
    #     torch.full((batch, q_seq_len, v_head_dim), 10, dtype=torch.float32)
    # )
    # do = to_default_device(
    #     torch.arange(batch*q_seq_len, dtype=torch.float32).reshape((batch, q_seq_len, 1)).expand(batch, q_seq_len, v_head_dim)
    # )
    do = device_randn(batch, q_seq_len, v_head_dim, dtype=torch.float16).to(
        torch.float32
    )

    o_ref, dq_ref, dk_ref, dv_ref, s_ref, p_ref, ds_ref, dp_ref = (
        attention_torch_ops_ref(q, k, v, do)
    )

    # We validated these elsewhere
    o_loops, lse, s_loops = attention_flash_fwd_loops(q, k, v)
    assert_close(o_loops, o_ref)
    assert_close(s_loops, s_ref)

    q = q.to(torch.float16)
    k = k.to(torch.float16)
    v = v.to(torch.float16)
    do = do.to(torch.float16)
    lse = lse.to(torch.float16)
    p_ref = p_ref.to(torch.float16)
    dq_ref = dq_ref.to(torch.float16)
    dk_ref = dk_ref.to(torch.float16)
    dv_ref = dv_ref.to(torch.float16)
    ds_ref = ds_ref.to(torch.float16)


    attention_bwd_dv, hyperparams_dv = get_attention_bwd_dv_kernel(
        batch=batch,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        q_seq_len=q_seq_len,
        v_head_dim=v_head_dim,
        mfma_variant=mfma_variant,
    )
    hyperparams_dv.update(get_default_scheduling_params())
    config = get_default_run_config()
    # config["print_ir_after_all"] = True
    compile_config_dv = {
        "waves_per_eu": 2,
        "denorm_fp_math_f32": "preserve-sign",
        # "print_ir_after": ["last", "set_node_indices"],
        # "print_ir_before": ["first"],
        # "print_signature": True,
        # "print_pretty_mlir": True,
        # "print_indices": True,
    }

    with tk.gen.TestLaunchContext(
        hyperparams_dv,
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        compile_config=compile_config_dv,
        schedule=False,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):


        dv = torch.zeros_like(v)
        s = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
        p = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)

        mb_bwd_dv = attention_bwd_dv(q, k, do, lse, dv, s, p)

        if dump_generated_mlir:
            filename = f"out/wave_attention_bwd_dv_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_bwd_dv.module_op.get_asm())
            print(f"IR dumped to {filename}")

        assert_close(s, s_ref, **tols)
        assert_close(p, p_ref, **tols)
        assert_close(dv, dv_ref, **tols)


    attention_bwd_dk, hyperparams_dk = get_attention_bwd_dk_kernel(
        batch=batch,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        q_seq_len=q_seq_len,
        v_head_dim=v_head_dim,
        mfma_variant=mfma_variant,
    )
    hyperparams_dk.update(get_default_scheduling_params())
    config = get_default_run_config()
    # config["print_ir_after_all"] = True
    compile_config_dk = {
        "waves_per_eu": 2,
        "denorm_fp_math_f32": "preserve-sign",
        # "print_ir_before": ["first", "set_node_indices"],
        # "print_ir_after": ["last"],
        # "print_signature": True,
        # "print_pretty_mlir": True,
        # "print_indices": True,
    }

    with tk.gen.TestLaunchContext(
        hyperparams_dk,
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        compile_config=compile_config_dk,
        schedule=False,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        
        D = torch.sum(do * o_ref, -1).to(torch.float16)
        dk = torch.zeros_like(k)
        s = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
        p = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
        ds = torch.zeros_like(p)
        dp = torch.zeros_like(s)
        dp_sub = torch.zeros_like(p)

        mb_bwd_dk = attention_bwd_dk(
            q,
            k,
            v,
            do,
            lse,
            D,
            dk,
            s,
            p,
            ds,
            dp,
            dp_sub,
        )

        if dump_generated_mlir:
            filename = f"out/wave_attention_bwd_dk_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_bwd_dk.module_op.get_asm())
            print(f"IR dumped to {filename}")

        dp_sub_ref = (dp_ref - D.reshape((batch, q_seq_len, 1))).to(torch.float16)

        assert_close(s, s_ref, **tols)
        assert_close(p, p_ref, **tols)
        assert_close(dp, dp_ref, **tols)
        assert_close(dp_sub, dp_sub_ref, **tols)
        assert_close(ds, ds_ref, **tols)
        assert_close(dk, dk_ref, **tols)


    attention_bwd_dq, hyperparams_dq = get_attention_bwd_dq_kernel(
        batch=batch,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        q_seq_len=q_seq_len,
        v_head_dim=v_head_dim,
        mfma_variant=mfma_variant,
    )
    hyperparams_dq.update(get_default_scheduling_params())
    config = get_default_run_config()
    # config["print_ir_after_all"] = True
    compile_config_dq = {
        "waves_per_eu": 2,
        "denorm_fp_math_f32": "preserve-sign",
        "print_ir_before": ["expand_graph"],
        "print_ir_after": ["last"],
        # "print_signature": True,
        "print_pretty_mlir": True,
        "print_indices": True,
    }

    with tk.gen.TestLaunchContext(
        hyperparams_dq,
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        compile_config=compile_config_dq,
        schedule=False,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        
        D = torch.sum(do * o_ref, -1).to(torch.float16)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        s = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
        p = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
        s_sub = torch.zeros_like(p)
        ds = torch.zeros_like(p)
        dp = torch.zeros_like(s)
        dp_sub = torch.zeros_like(p)

        mb_bwd_dq = attention_bwd_dq(
            q,
            k,
            v,
            do,
            lse,
            # D,
            # dq,
            s,
            s_sub,
            p,
            # ds,
            dp,
            # dp_sub,
        )

        if dump_generated_mlir:
            filename = f"out/wave_attention_bwd_dq_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_bwd_dq.module_op.get_asm())
            print(f"IR dumped to {filename}")

        s_sub_ref = s_ref.to(torch.float16) - lse.reshape((batch, q_seq_len, 1)).expand(batch, q_seq_len, kv_seq_len)
        dp_sub_ref = (dp_ref - D.reshape((batch, q_seq_len, 1))).to(torch.float16)

        # print(small_tensor_string(s, "s"))
        # print(small_tensor_string(lse, "lse"))
        assert_close(s, s_ref, **tols)
        assert_close(s_sub, s_sub_ref, **tols)
        assert_close(p, p_ref, **tols)
        assert_close(dp, dp_ref, **tols)
        assert_close(dp_sub, dp_sub_ref, **tols)
        assert_close(ds, ds_ref, **tols)
        assert_close(dq, dq_ref, **tols)

def small_tensor_string(t, name="", row_limit=50, col_limit=150, min_important_value=1e-5):
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
    sci_mode = torch.any(torch.logical_and(abs > min_important_value, abs < 1e-3)) or torch.max(abs) > 1e3
    
    def fallback():
        with torch._tensor_str.printoptions(precision=2 if sci_mode else 3, linewidth=col_limit, sci_mode=sci_mode, threshold=0):
            return f"{name}[{shape}], {t.dtype}:\n{t}"

    if len(t.shape) > 2 or t.shape[0] > row_limit:
        return fallback()

    def f_entry(d, width=0):
        return f"{d: {width}.2e}" if sci_mode else f"{d: {width}.3f}"

    width = max(len(f_entry(d)) for d in t.flatten().tolist())

    row_width = len(" ".join(f_entry(d, width) for d in t[0].tolist()))

    if row_width > col_limit:
        return fallback()

    rows = []
    for row in t:
        rows.append(" ".join(f_entry(d, width) for d in row.tolist()))

    nl = "\n"
    return f"{name}[{shape}], {t.dtype}:\n{nl.join(rows)}"


def bool_tensor_string(t, print_limit=2048):
    if t.shape.numel() > print_limit:
        return ""
    
    if len(t.shape) > 2:
        t = t.squeeze()
    if len(t.shape) < 2:
        t = t.unsqueeze(0)
    if len(t.shape) > 2:
        return ""

    rows = []
    for row in t:
        rows.append(" ".join("." if el else "F" for el in row.tolist()))

    return "\n".join(rows)


@require_e2e
@param_mfma_shape
def testAttentionMine(mfma_variant: MMAType, shape: tuple[int], request):
    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
    small_shape = math.prod(shape) < 500_000
    extra_verification = small_shape
    # Our float tolerances here are really bad on mi210
    tols = dict(atol=3e-2, rtol=5e-3) if small_shape else dict(atol=2e-1, rtol=5e-2)

    torch.manual_seed(0)
    # doing all this manual stuff in float32 or we lose too much precision. We
    # generate the random numbers in float16 though so we at least start with
    # something that is representable in that.

    q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16).to(
        torch.float32
    )
    # q = torch.full((batch, q_seq_len, qk_head_dim), 0.1, device=get_default_device())
    # q = torch.arange(
    #     (batch * q_seq_len * qk_head_dim), dtype=torch.float32, device=get_default_device()
    # ).reshape((batch, q_seq_len, qk_head_dim)) / (batch * q_seq_len * qk_head_dim)
    # q[0, :, (qk_head_dim//2):] = 0

    k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16).to(
        torch.float32
    )
    # k = torch.full((batch, kv_seq_len, qk_head_dim), 0.2, device=get_default_device())
    # k = torch.arange(
    #     (batch * kv_seq_len * qk_head_dim), dtype=torch.float32
    # ).reshape((batch, kv_seq_len, qk_head_dim)) / (batch * kv_seq_len * qk_head_dim)
    # k[0, :, (qk_head_dim//2):] = 0

    v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16).to(
        torch.float32
    )
    # v = torch.arange(
    #     (batch * kv_seq_len * v_head_dim), dtype=torch.float32
    # ).reshape((batch, kv_seq_len, v_head_dim)) / 64

    # Fake derivative for whatever the loss function might be
    # do = to_default_device(
    #     torch.full((batch, q_seq_len, v_head_dim), 10, dtype=torch.float32)
    # )
    # do = to_default_device(
    #     torch.arange(batch*q_seq_len, dtype=torch.float32).reshape((batch, q_seq_len, 1)).expand(batch, q_seq_len, v_head_dim)
    # )
    do = device_randn(batch, q_seq_len, v_head_dim, dtype=torch.float16).to(
        torch.float32
    )

    # temporarily not scaling so I don't have to figure out how to get that into the backward kernel
    # dk_sqrt = math.sqrt(1.0 / qk_head_dim)
    o_ref, dq_ref, dk_ref, dv_ref = attention_torch_builtin_ref(q, k, v, do)

    if extra_verification:
        o_ops, dq_ops, dk_ops, dv_ops, s_ops, p_ops, ds_ops, dp_ops = (
            attention_torch_ops_ref(q, k, v, do)
        )

        assert_close(o_ops, o_ref)
        assert_close(dq_ops, dq_ref)
        assert_close(dk_ops, dk_ref)
        assert_close(dv_ops, dv_ref)

        # These are presumably correct at this point and now we can use them as
        # a reference for other intermediates. And then also use the other
        # outputs of this as our references so that they're consistent.
        o_ref = o_ops
        dq_ref = dq_ops
        dk_ref = dk_ops
        dv_ref = dv_ops
        s_ref = s_ops
        p_ref = p_ops
        ds_ref = ds_ops
        dp_ref = dp_ops

    if extra_verification:
        dq_jac, dk_jac, dv_jac, ds_jac, dp_jac = attention_bwd_jacobian(q, k, v, do)

        assert_close(dv_jac, dv_ref)
        assert_close(dp_jac, dp_ref)
        assert_close(ds_jac, ds_ref)
        assert_close(dq_jac, dq_ref)
        assert_close(dk_jac, dk_ref)

        # Flash attention implemented as loops
        o_loops, lse_loops, s_loops = attention_flash_fwd_loops(q, k, v)

        assert_close(o_loops, o_ref)
        assert_close(s_loops, s_ref)
        # And now we have a reference for lse
        lse_ref = lse_loops

        dq_loops = torch.zeros_like(q)
        dk_loops = torch.zeros_like(k)
        dv_loops = torch.zeros_like(v)

        dq_loops, dk_loops, dv_loops, ds_loops, dp_loops = attention_flash_bwd_loops(
            q, k, v, do, o_loops, lse_loops
        )

        assert_close(ds_loops, ds_ref)
        assert_close(dp_loops, dp_ref)
        assert_close(dq_loops, dq_ref)
        assert_close(dk_loops, dk_ref)
        assert_close(dv_loops, dv_ref)

    # Alright, back to float16, which wave requires

    if extra_verification:
        lse_ref = lse_ref.to(torch.float16)
        # s_ref and dp_ref are matrix accumulators, so still f32
        p_ref = p_ref.to(torch.float16)
        ds_ref = ds_ref.to(torch.float16)

    o_ref = o_ref.to(torch.float16)
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

        o = device_zeros(batch, q_seq_len, v_head_dim, dtype=torch.float16)
        lse = device_zeros(batch, q_seq_len, dtype=torch.float16)
        s = device_zeros(batch, q_seq_len, kv_seq_len)

        mb_fwd = attention_fwd(q, k, v.transpose(-1, -2), s, o, lse)

        if dump_generated_mlir:
            filename = f"out/wave_attention_fwd_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_fwd.module_op.get_asm())
            print(f"IR dumped to {filename}")

        if extra_verification:
            assert_close(s, s_ref, **tols)
            # Can't check P, since we don't actually compute the "real" thing and rescale as we go.
            # Use the manual loops implementation to verify the LSE part of the
            # kernel, which we can't do with the base torch implementaiton.
            assert_close(lse, lse_ref, **tols)

        assert_close(o, o_ref, **tols)

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
        # "print_ir_after": ["last"],
        # "print_ir_before": ["first"],
        # "print_signature": True,
        # "print_pretty_mlir": True,
        # "print_indices": True,
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
        # TODO: maybe compute this within the kernel? The description of the
        # backward pass in the Flash Attention 2 paper has it computed external
        # to both loops though. It also seems there's a typo because it says it
        # has dimensionality of the head dimension, which doesn't make much
        # sense give that it is subtracted from dP which only has sequence
        # length dimensions. I'm assuming that's a typo (there are actually
        # quite a few in the paper, unfortunately).
        D = torch.sum(do * o, -1)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        s = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
        p = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
        ds = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
        dp = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
        dp_sub = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)

        expanded_lse = lse.reshape((batch, q_seq_len, 1)).expand(
            batch, q_seq_len, kv_seq_len
        )

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
            filename = f"out/wave_attention_bwd_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_bwd.module_op.get_asm())
            print(f"IR dumped to {filename}")

        assert_close(dv, dv_ref, **tols)

        if extra_verification:
            dp_sub_ref = (dp_ref - D.reshape((batch, q_seq_len, 1))).to(torch.float16)
            assert_close(dp, dp_ref, **tols)
            assert_close(dp_sub, dp_sub_ref, **tols)

            # I don't actually understand why this way of calculating ds works at all.
            assert_close(ds_ref, p_ref * dp_sub_ref, **tols)
            assert_close(ds, ds_ref, **tols)

        # the numerics here are ridiculously bad...
        assert_close(dk, dk_ref, **tols)
        assert_close(dq, dq_ref, **tols)


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
        # "print_grid": True,
        # "print_ir_after": ["set_node_indices", "expand_graph", "first", "last"],
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

        mb_fwd = attention_fwd(q.to(torch.float16), k.to(torch.float16), s, m)
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
        ),
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
        q = torch.full(
            (q_seq_len, qk_head_dim),
            0.1,
            device=get_default_device(),
            dtype=torch.float16,
        )
        k = torch.full(
            (kv_seq_len, qk_head_dim),
            0.2,
            device=get_default_device(),
            dtype=torch.float16,
        )
        do = torch.full(
            (q_seq_len, v_head_dim),
            0.3,
            device=get_default_device(),
            dtype=torch.float16,
        )

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
            filename = (
                f"wave_chained_gemm_undistributed_{'x'.join(map(str, shape))}.mlir"
            )
            with open(filename, "w") as f:
                f.write(mb_bwd.module_op.get_asm())
            print(f"IR dumped to {filename}")

        assert_close(dv, dv_ref.to(torch.float32), atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("shape", shapes_16x16x16, ids=get_param_id)
def testReproExpansionOrthogonalToReduction(shape):
    # shape = (1, 16, 32, 16, 16)
    _, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
    mfma_variant = MMAType.F32_16x16x16_F16
    # Input sizes
    M = tkl.sym.M  # query sequence length
    N = tkl.sym.N  # value head dimension
    K1 = tkl.sym.K1  # query/key head dimension
    K2 = tkl.sym.K2  # key/value sequence length

    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(K2, K2, 0),
        tkw.WorkgroupConstraint(N, N, 1),
        tkw.TilingConstraint(M, M),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            # Without setting vector shape for N, the second result of the reduction gets trampled
            vector_shapes={N: 16, M: 16, K1: 16, K2: 16},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    flip_m_k1_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: j, K1: i},
        outputs={K1: i, M: j},
    )

    @tkw.wave(constraints)
    def attention_bwd(
        q: tkl.Memory[M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[K2, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        do: tkl.Memory[N, M, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dk: tkl.Memory[K2, K1, GLOBAL_ADDRESS_SPACE, tkl.f32],
        dv: tkl.Memory[K2, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        ds: tkl.Memory[K2, M, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        dv_init = tkl.Register[K2, N, tkl.f32](0.0)
        dk_init = tkl.Register[K2, K1, tkl.f32](0.0)

        @tkw.reduction(M, init_args=[dv_init, dk_init])
        def loop_q_seq_len(
            dv_prev: tkl.Register[K2, N, tkl.f32],
            dk_prev: tkl.Register[K2, K1, tkl.f32],
        ):
            k_j = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            q_i = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)

            s_acc = tkl.Register[M, K2, tkl.f32](0.0)
            s_ij = tkw.mma(q_i, k_j, s_acc)
            s_ij = tkw.permute(s_ij, [K2, M])

            do_i = tkw.read(do, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            dv_j = tkw.mma(tkw.cast(s_ij, tkl.f16), do_i, dv_prev)

            ds_ij = tkw.read(ds, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            q_i_for_dk = tkw.read(
                q,
                mapping=flip_m_k1_read_mapping,
                elements_per_thread=LOAD_ELEMS_PER_THREAD,
            )
            dk_j = tkw.mma(ds_ij, q_i_for_dk, dk_prev)
            return (dv_j, dk_j)

        (dv_j, dk_j) = loop_q_seq_len
        tkw.write(dv_j, dv, elements_per_thread=STORE_ELEMS_PER_THREAD)
        tkw.write(dk_j, dk, elements_per_thread=STORE_ELEMS_PER_THREAD)

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
        "print_ir_after": ["expand_graph"],
        "print_ir_before": ["expand_graph"],
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
    ):

        torch.manual_seed(0)
        q = device_randn(q_seq_len, qk_head_dim, dtype=torch.float16)
        k = device_randn(kv_seq_len, qk_head_dim, dtype=torch.float16)
        do = device_randn(q_seq_len, v_head_dim, dtype=torch.float16)
        ds = device_randn(kv_seq_len, q_seq_len, dtype=torch.float16)

        s_ref = torch.matmul(q, k.transpose(-1, -2))
        dv_ref = torch.matmul(s_ref.transpose(-1, -2), do)

        dk = device_zeros(kv_seq_len, qk_head_dim, dtype=torch.float32)
        dv = device_zeros(kv_seq_len, v_head_dim, dtype=torch.float32)

        mb_bwd = attention_bwd(
            q,
            k,
            do.transpose(-1, -2),
            dk,
            dv,
            ds,
        )

        assert_close(dv, dv_ref.to(torch.float32), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "mfma_variant",
    [MMAType.F32_16x16x16_F16, MMAType.F32_32x32x8_F16],
    ids = get_param_id,
)
def testReproNonSquareMMAWithElementwise(mfma_variant):
    shape = (1, 32, 32, 32, 32)
    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
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
    MFMA_INPUT_ELS = tkl.sym.LOAD_ELEMS_PER_THREAD
    MFMA_OUTPUT_ELS = tkl.sym.STORE_ELEMS_PER_THREAD

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    elif mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(K2_kvs, BLOCK_K2, 0),
        # Degenerate distribution seems to fix some bugs
        tkw.WorkgroupConstraint(K1_qkd, K1_qkd, 1),
        tkw.WorkgroupConstraint(N_vd, N_vd, 2),
        # Can only have 3 dimensions distributed in actual blocks or the
        # compiler tries to index too far into waves_per_block (and if that is
        # made longer there's just a fatal crash), so batch dimension needs to
        # be last.
        tkw.WorkgroupConstraint(B, BLOCK_B, 3),
        tkw.TilingConstraint(M_qs, BLOCK_M),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={
                B: 0,
                N_vd: vec_size,
                K2_kvs: vec_size,
            },  # M_qs: vec_size, , K2_kvs: vec_size},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_k2_m_write_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, K2_kvs: j, M_qs: k},
        outputs={B: i, M_qs: k, K2_kvs: j},
    )

    flip_m_n_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, N_vd: j},
        outputs={B: i, N_vd: j, M_qs: k},
    )

    @tkw.wave(constraints)
    def attention_bwd(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        do: tkl.Memory[B, M_qs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # This has to be loaded in this way or set_node_indices fails with resolving thread shapes
        lse: tkl.Memory[B, K2_kvs, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dv: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        p: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        dv_init = tkl.Register[B, K2_kvs, N_vd, tkl.f32](0.0)

        @tkw.reduction(M_qs, init_args=[dv_init])
        def loop_q_seq_len(dv_prev: tkl.Register[B, K2_kvs, N_vd, tkl.f32]):
            k_j_for_s = tkw.read(k, elements_per_thread=MFMA_INPUT_ELS)
            q_i_for_s = tkw.read(q, elements_per_thread=MFMA_INPUT_ELS)

            s_acc = tkl.Register[B, M_qs, K2_kvs, tkl.f32](0.0)
            log2e = tkl.Register[B, K2_kvs, M_qs, tkl.f16](1.44269504089)
            s_ij = tkw.mma(q_i_for_s, k_j_for_s, s_acc)
            s_ij = tkw.permute(s_ij, [B, K2_kvs, M_qs])
            tkw.write(
                s_ij,
                s,
                mapping=flip_k2_m_write_mapping,
                elements_per_thread=MFMA_OUTPUT_ELS,
            )
            # Have to read for the MMA output here instead of input
            lse_i = tkw.read(lse, elements_per_thread=MFMA_OUTPUT_ELS)
            p_ij = tkw.exp2(log2e * (tkw.cast(s_ij, tkl.f16) - lse_i))
            tkw.write(
                p_ij,
                p,
                mapping=flip_k2_m_write_mapping,
                elements_per_thread=MFMA_OUTPUT_ELS,
            )

            # p_ij_for_dv = tkw.read(p, mapping=flip_m_k2_read_mapping, elements_per_thread=MFMA_INPUT_ELS)
            do_i = tkw.read(
                do, mapping=flip_m_n_read_mapping, elements_per_thread=MFMA_INPUT_ELS
            )
            dv_j = tkw.mma(p_ij, do_i, dv_prev)

            return dv_j

        tkw.write(
            tkw.cast(loop_q_seq_len, tkl.f16), dv, elements_per_thread=MFMA_OUTPUT_ELS
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        MFMA_INPUT_ELS: get_mfma_load_elems_per_thread(mfma_variant),
        MFMA_OUTPUT_ELS: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: vec_size,
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

    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    # config["print_ir_after_all"] = True
    compile_config = {
        "waves_per_eu": 2,
        "denorm_fp_math_f32": "preserve-sign",
        "print_pretty_mlir": True,
        # "print_grid": True,
        "print_ir_after": ["last"],
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
        q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16)
        k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16)
        v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16)

        s_ref = torch.matmul(q, k.transpose(-1, -2))
        m_ref = torch.max(s_ref, dim=-1)[0]
        p_squiggle_ref = torch.exp(s_ref - m_ref)
        l_ref = torch.sum(p_squiggle_ref, dim=-1)
        o_squiggle_ref = torch.matmul(p_squiggle_ref, v)
        o_ref = o_squiggle_ref / l_ref
        lse_ref = m_ref + torch.log(l_ref)

        do = device_randn(batch, q_seq_len, v_head_dim, dtype=torch.float16)
        D = torch.sum(do * o_ref, dim=-1)
        p_ref = torch.exp(
            s_ref - lse_ref.unsqueeze(-1).expand(batch, q_seq_len, kv_seq_len)
        )
        dv_ref = torch.matmul(p_ref.transpose(-1, -2), do)
        dp_ref = torch.matmul(do, v.transpose(-1, -2))

        dv = torch.zeros_like(v)
        s = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
        p = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)

        expanded_lse = lse_ref.reshape((batch, q_seq_len, 1)).expand(
            batch, q_seq_len, kv_seq_len
        )

        mb_bwd = attention_bwd(
            q,
            k,
            do,
            expanded_lse.transpose(-1, -2),
            dv,
            s,
            p,
        )

        if dump_generated_mlir:
            filename = f"out/wave_attention_repro_non_square_subtract_2_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb_bwd.module_op.get_asm())
            print(f"IR dumped to {filename}")

        assert_close(s.to(torch.float16), s_ref, atol=5e-2, rtol=1e-2)
        assert_close(p, p_ref, atol=5e-2, rtol=1e-2)
        assert_close(dv, dv_ref, atol=5e-2, rtol=1e-2)
