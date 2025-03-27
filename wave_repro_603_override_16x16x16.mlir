#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>
module attributes {transform.with_named_sequence} {
  stream.executable private @repro_603 {
    stream.executable.export public @repro_603 workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @repro_603(%arg_a: !stream.binding, %arg_b: !stream.binding, %arg_e: !stream.binding, %arg_a_transpose: !stream.binding, %arg_c: !stream.binding, %arg_d: !stream.binding) attributes {translation_info = #translation} {
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c64 = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %c0_4vf32 = arith.constant dense<0.000000e+00> : vector<4xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %a = stream.binding.subspan %arg_a[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %b = stream.binding.subspan %arg_b[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %c = stream.binding.subspan %arg_c[%c0] : !stream.binding -> memref<16x16xf32, strided<[16, 1], offset: ?>>
        %e = stream.binding.subspan %arg_e[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %a_transpose = stream.binding.subspan %arg_a_transpose[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %d = stream.binding.subspan %arg_d[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %1 = arith.muli %workgroup_id_0, %c16 overflow<nsw, nuw> : index
        %2 = arith.remsi %thread_id_x, %c16 : index
        %3 = arith.addi %2, %1 overflow<nsw, nuw> : index
        %x_mod_tpw = arith.remsi %thread_id_x, %c64 : index
        %5 = arith.divsi %x_mod_tpw, %c16 : index
        %6 = arith.muli %5, %c4 overflow<nsw, nuw> : index
        %7 = arith.muli %workgroup_id_1, %c16 overflow<nsw, nuw> : index
        %8 = arith.addi %7, %6 overflow<nsw, nuw> : index
        %a_reg = vector.load %a[%2, %8] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        %b_reg = vector.load %b[%3, %8] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        %49 = amdgpu.mfma %a_reg * %b_reg + %c0_4vf32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %50 = vector.extract_strided_slice %49 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %50, %c[%6, %3] : memref<16x16xf32, strided<[16, 1], offset: ?>>, vector<1xf32>
        %52 = vector.extract_strided_slice %49 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %53 = arith.addi %6, %c1 overflow<nsw, nuw> : index
        vector.store %52, %c[%53, %3] : memref<16x16xf32, strided<[16, 1], offset: ?>>, vector<1xf32>
        %54 = vector.extract_strided_slice %49 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %55 = arith.addi %6, %c2 overflow<nsw, nuw> : index
        vector.store %54, %c[%55, %3] : memref<16x16xf32, strided<[16, 1], offset: ?>>, vector<1xf32>
        %56 = vector.extract_strided_slice %49 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %57 = arith.addi %6, %c3 overflow<nsw, nuw> : index
        vector.store %56, %c[%57, %3] : memref<16x16xf32, strided<[16, 1], offset: ?>>, vector<1xf32>
        %e_reg = vector.load %e[%3, %6] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>

        // GPU shuffle here
        %wave_size_i32 = arith.constant 64 : i32

        %wave_size = arith.constant 64 : index
        %els_per_thread = arith.constant 4 : index

        %a_reg_ext = arith.extf %a_reg : vector<4xf16> to vector<4xf32>

        %a_t_reg_ext = arith.constant dense<0.000000e+00> : vector<4xf32>

        // constant for the thread
        %x_div_ept = arith.divsi %thread_id_x, %els_per_thread : index // (x // els_per_thread)
        %x_div_ept_times_16 = arith.muli %x_div_ept, %c16 : index // (x // els_per_thread) * 16
        %x_div_16 = arith.divsi %thread_id_x, %c16 : index        // (x // 16)
        %x_div_16_times_ept = arith.muli %els_per_thread, %x_div_16 : index  // els_per_thread * (x // 16)
        %x_plus_ept = arith.addi %thread_id_x, %els_per_thread : index

        // unrolled loop (4)
        // shuff_i = 0
        %shuff0 = arith.constant 0 : index
        %x_plus_shuff0 = arith.addi %thread_id_x, %shuff0 : index  // x + shuff0
        %dest0_idx = arith.remsi %x_plus_shuff0, %els_per_thread : index  // (x + shuff0) % els_per_thread

        // ((x // els_per_thread) * 16 + dest_i) % threads_per_wave + els_per_thread * (x // 16)
        %tmp0_0 = arith.addi %x_div_ept_times_16, %dest0_idx : index // (x // els_per_thread) * 16 + dest_i
        %tmp0_1 = arith.remsi %tmp0_0, %wave_size : index            // ((x // els_per_thread) * 16 + dest_i) % threads_per_wave
        %src_thread0_idx = arith.addi %tmp0_1, %x_div_16_times_ept : index
        %src_thread0_i32 = arith.index_cast %src_thread0_idx : index to i32

        // (x + els_per_thread - shuff_i) % els_per_thread
        %tmp0_2 = arith.subi %x_plus_ept, %shuff0 : index
        %offer0_idx = arith.remsi %tmp0_2, %els_per_thread : index
        %offer0 = vector.extract %a_reg_ext[%offer0_idx] : f32 from vector<4xf32>

        %get0, %valid0 = gpu.shuffle idx %offer0, %src_thread0_i32, %wave_size_i32 : f32
        %a_t_reg0_ext = vector.insert %get0, %a_t_reg_ext[%dest0_idx] : f32 into vector<4xf32>

        // shuff_i = 1
        %shuff1 = arith.constant 1 : index
        %x_plus_shuff1 = arith.addi %thread_id_x, %shuff1 : index
        %dest1_idx = arith.remsi %x_plus_shuff1, %els_per_thread : index

        %tmp1_0 = arith.addi %x_div_ept_times_16, %dest1_idx : index
        %tmp1_1 = arith.remsi %tmp1_0, %wave_size : index
        %src_thread1_idx = arith.addi %tmp1_1, %x_div_16_times_ept : index
        %src_thread1_i32 = arith.index_cast %src_thread1_idx : index to i32

        %tmp1_2 = arith.subi %x_plus_ept, %shuff1 : index
        %offer1_idx = arith.remsi %tmp1_2, %els_per_thread : index
        %offer1 = vector.extract %a_reg_ext[%offer1_idx] : f32 from vector<4xf32>

        %get1, %valid1 = gpu.shuffle idx %offer1, %src_thread1_i32, %wave_size_i32 : f32
        %a_t_reg1_ext = vector.insert %get1, %a_t_reg0_ext[%dest1_idx] : f32 into vector<4xf32>

        // shuff_i = 2
        %shuff2 = arith.constant 2 : index
        %x_plus_shuff2 = arith.addi %thread_id_x, %shuff2 : index
        %dest2_idx = arith.remsi %x_plus_shuff2, %els_per_thread : index

        %tmp2_0 = arith.addi %x_div_ept_times_16, %dest2_idx : index
        %tmp2_1 = arith.remsi %tmp2_0, %wave_size : index
        %src_thread2_idx = arith.addi %tmp2_1, %x_div_16_times_ept : index
        %src_thread2_i32 = arith.index_cast %src_thread2_idx : index to i32

        %tmp2_2 = arith.subi %x_plus_ept, %shuff2 : index
        %offer2_idx = arith.remsi %tmp2_2, %els_per_thread : index
        %offer2 = vector.extract %a_reg_ext[%offer2_idx] : f32 from vector<4xf32>

        %get2, %valid2 = gpu.shuffle idx %offer2, %src_thread2_i32, %wave_size_i32 : f32
        %a_t_reg2_ext = vector.insert %get2, %a_t_reg1_ext[%dest2_idx] : f32 into vector<4xf32>

        // shuff_i = 3
        %shuff3 = arith.constant 3 : index
        %x_plus_shuff3 = arith.addi %thread_id_x, %shuff3 : index
        %dest3_idx = arith.remsi %x_plus_shuff3, %els_per_thread : index

        %tmp3_0 = arith.addi %x_div_ept_times_16, %dest3_idx : index
        %tmp3_1 = arith.remsi %tmp3_0, %wave_size : index
        %src_thread3_idx = arith.addi %tmp3_1, %x_div_16_times_ept : index
        %src_thread3_i32 = arith.index_cast %src_thread3_idx : index to i32

        %tmp3_2 = arith.subi %x_plus_ept, %shuff3 : index
        %offer3_idx = arith.remsi %tmp3_2, %els_per_thread : index
        %offer3 = vector.extract %a_reg_ext[%offer3_idx] : f32 from vector<4xf32>

        %get3, %valid3 = gpu.shuffle idx %offer3, %src_thread3_i32, %wave_size_i32 : f32
        %a_t_reg3_ext = vector.insert %get3, %a_t_reg2_ext[%dest3_idx] : f32 into vector<4xf32>

        %a_t_reg = arith.truncf %a_t_reg3_ext : vector<4xf32> to vector<4xf16>
        // End GPU Shuffle

        vector.store %a_t_reg, %a_transpose[%2, %8] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        %34 = amdgpu.mfma %e_reg * %a_t_reg + %c0_4vf32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %35 = arith.truncf %34 : vector<4xf32> to vector<4xf16>
        %36 = vector.extract_strided_slice %35 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
        %38 = arith.addi %1, %6 overflow<nsw, nuw> : index
        %39 = arith.addi %2, %7 overflow<nsw, nuw> : index
        vector.store %36, %d[%38, %39] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
        %40 = vector.extract_strided_slice %35 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
        %41 = arith.addi %38, %c1 overflow<nsw, nuw> : index
        vector.store %40, %d[%41, %39] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
        %42 = vector.extract_strided_slice %35 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
        %43 = arith.addi %38, %c2 overflow<nsw, nuw> : index
        vector.store %42, %d[%43, %39] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
        %44 = vector.extract_strided_slice %35 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
        %45 = arith.addi %38, %c3 overflow<nsw, nuw> : index
        vector.store %44, %d[%45, %39] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg_a: tensor<16x16xf16>, %arg_b: tensor<16x16xf16>, %arg_e: tensor<16x16xf16>, %arg_a_transpose: tensor<16x16xf16>, %arg_c: tensor<16x16xf32>, %arg_d: tensor<16x16xf16>) -> (tensor<16x16xf16>, tensor<16x16xf32>, tensor<16x16xf16>) {
    %b:3 = flow.dispatch @repro_603::@repro_603(%arg_a, %arg_b, %arg_e, %arg_a_transpose, %arg_c, %arg_d) : (tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf32>, tensor<16x16xf16>) -> (%arg_a_transpose, %arg_c, %arg_d)
    return %b#0, %b#1, %b#2 : tensor<16x16xf16>, tensor<16x16xf32>, tensor<16x16xf16>
  }
}
