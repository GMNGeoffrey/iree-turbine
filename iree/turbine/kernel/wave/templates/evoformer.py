# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel._support.dtype import DataType
from iree.turbine.kernel.wave.utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)


def get_evoformer_kernel(
    batch: tuple[int, int],
    n: tuple[int, int],
    kv_seq_len: tuple[int, int],
    heads: tuple[int, int],
    head_dim: tuple[int, int],
    q_seq_len: tuple[int, int],
    v_dim: tuple[int, int],
    mfma_variant: MMAType,
    datatype: DataType,
):
    assert datatype in [tkl.f16, tkl.bf16], f"Unsupported datatype: {datatype}"

    # Input sizes
    B = tkl.sym.B
    BN = tkl.sym.BN
    M = tkl.sym.M
    H = tkl.sym.H
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_BN = tkl.sym.BLOCK_BN
    BLOCK_H = tkl.sym.BLOCK_H
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD


    # Expose user-constraints
    ratio_m = 2
    ratio_n = 1
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)] # batch
    constraints += [tkw.WorkgroupConstraint(BN, BLOCK_BN, 3)] # batch
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 4)] # batch
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / ratio_m)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / ratio_n)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(ratio_m, ratio_n, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, BN: 0, H: 0, M: Mvec, N: Nvec},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)
    m = tkw.IndexMapping.iterator(4)
    # [B, BN, M, H, K1] -> [B, BN, H, M, K1]
    q_mapping = tkw.IndexMapping(
        num_iterators=5,
        inputs={B: i, BN: j, H: k, M: l, K1: m},
        outputs={B: i, BN: j, H: k, M: l, K1: m},
    )
    # [B, BN, K2, H, K1] -> [B, BN, H, K2, K1]
    k_mapping = tkw.IndexMapping(
        num_iterators=5,
        inputs={B: i, BN: j, H: k, K2: l, K1: m},
        outputs={B: i, BN: j, H: k, K2: l, K1: m},
    )
    # [B, BN, N, H, K2] -> [B, BN, H, N, K2]
    v_mapping = tkw.IndexMapping(
        num_iterators=5,
        inputs={B: i, BN: j, H: k, N: l, K2: m},
        outputs={B: i, BN: j, H: k, N: l, K2: m},
    )
    # [B, BN, H, N, M] -> [B, BN, M, H, N]
    o_mapping = tkw.IndexMapping(
        num_iterators=5,
        inputs={B: i, BN: j, H: k, N: l, M: m},
        outputs={B: i, BN: j, H: k, N: l, M: m},
    )

    @tkw.wave(constraints)
    def evoformer_fwd(
        q: tkl.Memory[B, BN, M, H, K1, GLOBAL_ADDRESS_SPACE, datatype],
        k: tkl.Memory[B, BN, K2, H, K1, ADDRESS_SPACE, datatype],
        v: tkl.Memory[B, BN, N, H, K2, ADDRESS_SPACE, datatype],
        mask: tkl.Memory[B, BN, K2, GLOBAL_ADDRESS_SPACE, datatype],
        bias: tkl.Memory[B, H, M, K2, GLOBAL_ADDRESS_SPACE, datatype],
        c: tkl.Memory[B, BN, M, H, N, GLOBAL_ADDRESS_SPACE, datatype],
        lse: tkl.Memory[B, BN, H, M, GLOBAL_ADDRESS_SPACE, datatype],
    ):
        c_reg = tkl.Register[B, BN, H, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, BN, H, M, tkl.f32](0.0)
        init_max = tkl.Register[B, BN, H, M, tkl.f32](-1e6)

        @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, BN, H, M, tkl.f32],
            partial_sum: tkl.Register[B, BN, H, M, tkl.f32],
            acc: tkl.Register[B, BN, H, N, M, tkl.f32],
        ) -> (
            tkl.Register[B, BN, H, M, tkl.f32],
            tkl.Register[B, BN, H, M, tkl.f32],
            tkl.Register[B, BN, H, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, BN, H, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(
                q, mapping=q_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )
            if datatype == tkl.bf16:
                q_reg = tkw.cast(tkw.cast(q_reg, tkl.f32), tkl.f16)
            k_reg = tkw.read(
                k, mapping=k_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )
            if datatype == tkl.bf16:
                k_reg = tkw.cast(tkw.cast(k_reg, tkl.f32), tkl.f16)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, BN, H, M, K2])
            mask_reg = tkw.read(mask, elements_per_thread=STORE_ELEMS_PER_THREAD)
            casted_mask_reg = tkw.cast(mask_reg, tkl.f32)
            y_j = x_j + casted_mask_reg
            bias_reg = tkw.read(bias, elements_per_thread=STORE_ELEMS_PER_THREAD)
            casted_bias_reg = tkw.cast(bias_reg, tkl.f32)
            z_j = y_j + casted_bias_reg
            m_j = tkw.max(z_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(z_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(
                v, mapping=v_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )
            if datatype == tkl.bf16:
                v_reg = tkw.cast(tkw.cast(v_reg, tkl.f32), tkl.f16)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        res_casted = tkw.cast(res, datatype)
        tkw.write(
            res_casted, c, mapping=o_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )
        res_lse = res_max + tkw.log2(res_sum)
        res_lse_casted = tkw.cast(res_lse, datatype)
        tkw.write(res_lse_casted, lse, elements_per_thread=1)
        

    # TODO(gcmn): figure out address spaces
    @tkw.wave(constraints)
    def evoformer_bwd(
        # input
        do: tkl.Memory[B, BN, H, M, N, GLOBAL_ADDRESS_SPACE, datatype],
        D: tkl.Memory[B, BN, H, M, GLOBAL_ADDRESS_SPACE, datatype],
        q: tkl.Memory[B, BN, H, M, K1, GLOBAL_ADDRESS_SPACE, datatype],
        k: tkl.Memory[B, BN, H, K2, K1, ADDRESS_SPACE, datatype],
        v: tkl.Memory[B, BN, H, N, K2, ADDRESS_SPACE, datatype],
        # mask: tkl.Memory[B, BN, K2, GLOBAL_ADDRESS_SPACE, datatype],
        # bias: tkl.Memory[B, H, M, K2, GLOBAL_ADDRESS_SPACE, datatype],
        # o: tkl.Memory[B, BN, M, H, N, GLOBAL_ADDRESS_SPACE, datatype],
        lse: tkl.Memory[B, BN, H, M, GLOBAL_ADDRESS_SPACE, datatype],
        # output
        dq: tkl.Memory[B, BN, H, M, K1, GLOBAL_ADDRESS_SPACE, datatype],
        dk: tkl.Memory[B, BN, H, K2, K1, GLOBAL_ADDRESS_SPACE, datatype],
        dv: tkl.Memory[B, BN, H, N, K2, GLOBAL_ADDRESS_SPACE, datatype],
        # dbias: tkl.Memory[B, H, M, K2, GLOBAL_ADDRESS_SPACE, datatype],
    ):
        init_dq = tkl.Register[B, BN, H, M, K1, tkl.f32](0.0)
        init_dk = tkl.Register[B, BN, H, K2, K1, tkl.f32](0.0)
        init_dv = tkl.Register[B, BN, H, N, K2, tkl.f32](0.0)

        @tkw.reduction(K2, init_args=[
            init_dq,
            init_dk,
            init_dv,
        ]) 
        def repeat(
            partial_dq: tkl.Register[B, BN, H, M, K1, tkl.f32],
            partial_dk: tkl.Register[B, BN, H, K2, K1, tkl.f32],
            partial_dv: tkl.Register[B, BN, H, N, K2, tkl.f32],
        ) -> (
            tkl.Register[B, BN, H, M, K1, tkl.f32],
            tkl.Register[B, BN, H, K2, K1, tkl.f32],
            tkl.Register[B, BN, H, N, K2, tkl.f32],
        ):
            # Load things
            q_reg = tkw.read(q, mapping=q_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # if datatype == tkl.bf16:
                # q_reg = tkw.cast(tkw.cast(q_reg, tkl.f32), tkl.f16)
            
            # o_reg = tkw.read(
            #     o, mapping=o_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD
            # )
            # if datatype == tkl.bf16:
                # o_reg = tkw.cast(tkw.cast(o_reg, tkl.f32), tkl.f16)

            do_reg = tkw.read(do, mapping=o_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # if datatype == tkl.bf16:
                # do_reg = tkw.cast(tkw.cast(do_reg, tkl.f32), tkl.f16)
            
            lse_reg = tkw.read(lse, elements_per_thread=1)
            lse_reg = tkw.cast(lse_reg, tkl.f32)
            # if datatype == tkl.bf16:
                # lse_reg = tkw.cast(tkw.cast(lse_reg, tkl.f32), tkl.f16)

            D_reg = tkw.read(D, elements_per_thread=1)
            D_reg = tkw.cast(D_reg, tkl.f32)
            # if datatype == tkl.bf16:
                # D_reg = tkw.cast(tkw.cast(D_reg, tkl.f32), tkl.f16)
            
            k_reg = tkw.read(k, mapping=k_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # if datatype == tkl.bf16:
                # k_reg = tkw.cast(tkw.cast(k_reg, tkl.f32), tkl.f16)

            v_reg = tkw.read(v, mapping=v_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # if datatype == tkl.bf16:
                # v_reg = tkw.cast(tkw.cast(v_reg, tkl.f32), tkl.f16)
            # end loads


            S_imm_reg = tkl.Register[B, BN, H, K2, M, tkl.f32](0.0)
            q_perm = q_reg # tkw.permute(q_reg, target_shape=[B, BN, H, M, K1])
            kT = tkw.permute(k_reg, target_shape=[B, BN, H, K1, K2]) # f16
            do_perm = do_reg # tkw.permute(do_reg, target_shape=[B, BN, H, M, N]) f16

            S = tkw.mma(q_perm, kT, S_imm_reg) # [B, BN, H, M, K2] f32
            P = tkw.exp2(S - lse_reg) # [B, BN, H, M, K2] f32
            PT = tkw.permute(P, target_shape=[B, BN, H, K2, M]) # f32
            PT = tkw.cast(PT, tkl.f16)
            partial_dv_perm = partial_dv # tkw.permute(partial_dv, target_shape=[B, BN, H, K2, N]) f32
            new_partial_dv = tkw.mma(PT, do_perm, partial_dv_perm) # [B, BN, H, K2, N] f32
            new_partial_dv = new_partial_dv # tkw.permute(new_partial_dv, target_shape=[B, BN, N, H, K2])

            vT = tkw.permute(v_reg, target_shape=[B, BN, H, N, K2]) # f16
            dP_imm_reg = tkl.Register[B, BN, H, M, K2, tkl.f32](0.0)
            dP = tkw.mma(do_perm, vT, dP_imm_reg) # [B, BN, H, M, K2] f32
            # D_perm = tkw.permute(D_reg, target_shape=[B, BN, H, M]) # [B, BN, M, H]
            dS = P * (dP - D_reg) # [B, BN, H, M, K2] # f32
            dS = tkw.cast(dS, tkl.f16) # f16

            partial_dq_perm = partial_dq # tkw.permute(partial_dq, target_shape=[B, BN, H, M, K1])
            
            k_perm = k_reg # tkw.permute(k_reg, target_shape=[B, BN, H, K2, K1]) f16
            new_partial_dq = tkw.mma(dS, k_perm, partial_dq_perm) # [B, BN, H, M, K1] f32
            new_partial_dq = new_partial_dq # tkw.permute(new_partial_dq, target_shape=[B, BN, M, H, K1])

            dST = tkw.permute(dS, target_shape=[B, BN, H, K2, M]) # f16
            new_partial_dk = partial_dk # tkw.permute(partial_dk, target_shape=[B, BN, H, K2, K1])
            new_partial_dk = tkw.mma(dST, q_reg, new_partial_dk) # [B, BN, H, K2, K1] f32
            new_partial_dk = new_partial_dk # tkw.permute(new_partial_dk, target_shape=[B, BN, K2, H, K1])

            return (
                new_partial_dq,
                new_partial_dk,
                new_partial_dv,
            ) 

        (
            dq_res,
            dk_res,
            dv_res,
        ) = repeat
        dq_res_casted = tkw.cast(dq_res, datatype)
        tkw.write(
            dq_res_casted, dq, mapping=q_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )
        dk_res_casted = tkw.cast(dk_res, datatype)
        tkw.write(
            dk_res_casted, dk, mapping=k_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )
        dv_res_casted = tkw.cast(dv_res, datatype)
        tkw.write(
            dv_res_casted, dv, mapping=v_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )


    SHAPE = 0
    TILE_SIZE = 1

    symbols = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        B: batch[SHAPE],
        BN: n[SHAPE],
        K2: kv_seq_len[SHAPE],
        H: heads[SHAPE],
        K1: head_dim[SHAPE],
        M: q_seq_len[SHAPE],
        N: v_dim[SHAPE],
        BLOCK_B: batch[TILE_SIZE],
        BLOCK_BN: n[TILE_SIZE],
        BLOCK_H: heads[TILE_SIZE],
        BLOCK_M: q_seq_len[TILE_SIZE],
        BLOCK_N: v_dim[TILE_SIZE],
        BLOCK_K2: kv_seq_len[TILE_SIZE],
    }

    return evoformer_fwd, evoformer_bwd, symbols
