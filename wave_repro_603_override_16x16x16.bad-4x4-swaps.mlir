#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>
module attributes {transform.with_named_sequence} {
  stream.executable private @repro_603 {
    stream.executable.export public @repro_603 workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @repro_603(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c64 = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %thread_id_x = gpu.thread_id  x
        %0 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %1 = arith.muli %workgroup_id_0, %c16 overflow<nsw, nuw> : index
        %x_mod_16 = arith.remsi %thread_id_x, %c16 : index
        %3 = arith.addi %x_mod_16, %1 overflow<nsw, nuw> : index
        %4 = arith.remsi %thread_id_x, %c64 : index
        %x_div_16 = arith.divsi %4, %c16 : index
        %6 = arith.muli %x_div_16, %c4 overflow<nsw, nuw> : index
        %7 = vector.load %0[%3, %6] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        %8 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %9 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<16x16xf32, strided<[16, 1], offset: ?>>
        %10 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %11 = vector.load %8[%x_mod_16, %6] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        %12 = amdgpu.mfma %11 * %7 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %13 = vector.extract_strided_slice %12 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %13, %9[%6, %3] : memref<16x16xf32, strided<[16, 1], offset: ?>>, vector<1xf32>
        %14 = vector.extract_strided_slice %12 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %15 = arith.addi %6, %c1 overflow<nsw, nuw> : index
        vector.store %14, %9[%15, %3] : memref<16x16xf32, strided<[16, 1], offset: ?>>, vector<1xf32>
        %16 = vector.extract_strided_slice %12 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %17 = arith.addi %6, %c2 overflow<nsw, nuw> : index
        vector.store %16, %9[%17, %3] : memref<16x16xf32, strided<[16, 1], offset: ?>>, vector<1xf32>
        %18 = vector.extract_strided_slice %12 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %19 = arith.addi %6, %c3 overflow<nsw, nuw> : index
        vector.store %18, %9[%19, %3] : memref<16x16xf32, strided<[16, 1], offset: ?>>, vector<1xf32>
        %20 = vector.load %10[%3, %6] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        // Do a shuffle here


        // %shuffleResult0, %valid0 = gpu.shuffle idx %40, %shuff1_idx, %c64_i32 : f32
        // %shuffleResult1, %valid1 = gpu.shuffle idx %41, %c4_i32, %c64_i32 : f32
        // %shuffleResult2, %valid2 = gpu.shuffle idx %42, %c8_i32, %c64_i32 : f32
        // %shuffleResult3, %valid3 = gpu.shuffle idx %43, %c12_i32, %c64_i32 : f32

        // %a_reslice = vector.from_elements %shuffleResult0, %shuffleResult1, %shuffleResult2, %shuffleResult3 : vector<4xf32>
        // %a_reslice_trunc = arith.truncf %a_reslice : vector<4xf32> to vector<4xf16>

        %c64_i32 = arith.constant 64 : i32
        %c0_i32 = arith.constant 0 : i32
        %c4_i32 = arith.constant 4 : i32
        %c8_i32 = arith.constant 8 : i32
        %c12_i32 = arith.constant 12 : i32

        // We have to do shuffles in 32-bit
        %a_ext = arith.extf %11 : vector<4xf16> to vector<4xf32>

        %k_block_idx = arith.muli %x_div_16, %c16 : index
        %x_mod_4 = arith.remsi %thread_id_x, %c4 : index
        %inblock_idx = arith.divsi %x_mod_16, %c4 : index

        %t0_x_idx = arith.addi %k_block_idx, %x_mod_4 : index
        %t0_x = arith.index_cast %t0_x_idx : index to i32
        %t1_x = arith.addi %t0_x, %c4_i32 : i32
        %t2_x = arith.addi %t0_x, %c8_i32 : i32
        %t3_x = arith.addi %t0_x, %c12_i32 : i32


        %v_0 = vector.extract %a_ext[0] : f32 from vector<4xf32>
        %v_1 = vector.extract %a_ext[1] : f32 from vector<4xf32>
        %v_2 = vector.extract %a_ext[2] : f32 from vector<4xf32>
        %v_3 = vector.extract %a_ext[3] : f32 from vector<4xf32>

        %v_00, %valid_00 = gpu.shuffle idx %v_0, %t0_x, %c64_i32 : f32
        %v_01, %valid_01 = gpu.shuffle idx %v_1, %t0_x, %c64_i32 : f32
        %v_02, %valid_02 = gpu.shuffle idx %v_2, %t0_x, %c64_i32 : f32
        %v_03, %valid_03 = gpu.shuffle idx %v_3, %t0_x, %c64_i32 : f32

        %v_10, %valid_10 = gpu.shuffle idx %v_0, %t1_x, %c64_i32 : f32
        %v_11, %valid_11 = gpu.shuffle idx %v_1, %t1_x, %c64_i32 : f32
        %v_12, %valid_12 = gpu.shuffle idx %v_2, %t1_x, %c64_i32 : f32
        %v_13, %valid_13 = gpu.shuffle idx %v_3, %t1_x, %c64_i32 : f32

        %v_20, %valid_20 = gpu.shuffle idx %v_0, %t2_x, %c64_i32 : f32
        %v_21, %valid_21 = gpu.shuffle idx %v_1, %t2_x, %c64_i32 : f32
        %v_22, %valid_22 = gpu.shuffle idx %v_2, %t2_x, %c64_i32 : f32
        %v_23, %valid_23 = gpu.shuffle idx %v_3, %t2_x, %c64_i32 : f32

        %v_30, %valid_30 = gpu.shuffle idx %v_0, %t3_x, %c64_i32 : f32
        %v_31, %valid_31 = gpu.shuffle idx %v_1, %t3_x, %c64_i32 : f32
        %v_32, %valid_32 = gpu.shuffle idx %v_2, %t3_x, %c64_i32 : f32
        %v_33, %valid_33 = gpu.shuffle idx %v_3, %t3_x, %c64_i32 : f32

        // %a_tile = vector.from_elements %v_00, %v_01, %v_02, %v_03, %v_10, %v_11, %v_12, %v_13, %v_20, %v_21, %v_22, %v_23, %v_30, %v_31, %v_32, %v_33 : vector<4x4xf32>

        // vector.from_elements doesn't work for some reason. Need to instead create the vector first and then make 16 insert calls
        %a_tile_empty = arith.constant dense<0.000000e+00> : vector<4x4xf32>
        %a_tile_00 = vector.insert %v_00, %a_tile_empty[0, 0] : f32 into vector<4x4xf32>
        %a_tile_01 = vector.insert %v_01, %a_tile_00[0, 1] : f32 into vector<4x4xf32>
        %a_tile_02 = vector.insert %v_02, %a_tile_01[0, 2] : f32 into vector<4x4xf32>
        %a_tile_03 = vector.insert %v_03, %a_tile_02[0, 3] : f32 into vector<4x4xf32>
        %a_tile_10 = vector.insert %v_10, %a_tile_03[1, 0] : f32 into vector<4x4xf32>
        %a_tile_11 = vector.insert %v_11, %a_tile_10[1, 1] : f32 into vector<4x4xf32>
        %a_tile_12 = vector.insert %v_12, %a_tile_11[1, 2] : f32 into vector<4x4xf32>
        %a_tile_13 = vector.insert %v_13, %a_tile_12[1, 3] : f32 into vector<4x4xf32>
        %a_tile_20 = vector.insert %v_20, %a_tile_13[2, 0] : f32 into vector<4x4xf32>
        %a_tile_21 = vector.insert %v_21, %a_tile_20[2, 1] : f32 into vector<4x4xf32>
        %a_tile_22 = vector.insert %v_22, %a_tile_21[2, 2] : f32 into vector<4x4xf32>
        %a_tile_23 = vector.insert %v_23, %a_tile_22[2, 3] : f32 into vector<4x4xf32>
        %a_tile_30 = vector.insert %v_30, %a_tile_23[3, 0] : f32 into vector<4x4xf32>
        %a_tile_31 = vector.insert %v_31, %a_tile_30[3, 1] : f32 into vector<4x4xf32>
        %a_tile_32 = vector.insert %v_32, %a_tile_31[3, 2] : f32 into vector<4x4xf32>
        %a_tile_33 = vector.insert %v_33, %a_tile_32[3, 3] : f32 into vector<4x4xf32>

        %a_tile_trunc = arith.truncf %a_tile_33 : vector<4x4xf32> to vector<4x4xf16>

        %new_v_0 = vector.extract %a_tile_trunc[0, %inblock_idx] : f16 from vector<4x4xf16>
        %new_v_1 = vector.extract %a_tile_trunc[1, %inblock_idx] : f16 from vector<4x4xf16>
        %new_v_2 = vector.extract %a_tile_trunc[2, %inblock_idx] : f16 from vector<4x4xf16>
        %new_v_3 = vector.extract %a_tile_trunc[3, %inblock_idx] : f16 from vector<4x4xf16>

        // %a_reslice = vector.from_elements %a_reslice_0, %a_reslice_1, %a_reslice_2, %a_reslice_3 : vector<4xf16>
        // Need to replace vector.from_elements with 4 insert calls
        %a_reslice_empty = arith.constant dense<0.000000e+00> : vector<4xf16>
        %a_reslice_0 = vector.insert %new_v_0, %a_reslice_empty[0] : f16 into vector<4xf16>
        %a_reslice_1 = vector.insert %new_v_1, %a_reslice_0[1] : f16 into vector<4xf16>
        %a_reslice_2 = vector.insert %new_v_2, %a_reslice_1[2] : f16 into vector<4xf16>
        %a_reslice_3 = vector.insert %new_v_3, %a_reslice_2[3] : f16 into vector<4xf16>



        %21 = amdgpu.mfma %20 * %a_reslice_3 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %22 = arith.truncf %21 : vector<4xf32> to vector<4xf16>
        %23 = vector.extract_strided_slice %22 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
        %24 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %25 = arith.addi %1, %6 overflow<nsw, nuw> : index
        vector.store %23, %24[%25, %x_mod_16] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
        %26 = vector.extract_strided_slice %22 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
        %27 = arith.addi %25, %c1 overflow<nsw, nuw> : index
        vector.store %26, %24[%27, %x_mod_16] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
        %28 = vector.extract_strided_slice %22 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
        %29 = arith.addi %25, %c2 overflow<nsw, nuw> : index
        vector.store %28, %24[%29, %x_mod_16] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
        %30 = vector.extract_strided_slice %22 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
        %31 = arith.addi %25, %c3 overflow<nsw, nuw> : index
        vector.store %30, %24[%31, %x_mod_16] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<16x16xf16>, %arg1: tensor<16x16xf16>, %arg2: tensor<16x16xf16>, %arg3: tensor<16x16xf32>, %arg4: tensor<16x16xf16>) -> (tensor<16x16xf32>, tensor<16x16xf16>) {
    %0:2 = flow.dispatch @repro_603::@repro_603(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf32>, tensor<16x16xf16>) -> (%arg3, %arg4)
    return %0#0, %0#1 : tensor<16x16xf32>, tensor<16x16xf16>
  }
}
