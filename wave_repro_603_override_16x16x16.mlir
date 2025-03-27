#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>
module attributes {transform.with_named_sequence} {
  stream.executable private @repro_603 {
    stream.executable.export public @repro_603 workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @repro_603(%arg_a: !stream.binding, %arg_b: !stream.binding, %arg_e: !stream.binding, %arg_c: !stream.binding, %arg_d: !stream.binding) attributes {translation_info = #translation} {
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c64 = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %c0_4xf32 = arith.constant dense<0.000000e+00> : vector<4xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %thread_id_x = gpu.thread_id  x
        %b = stream.binding.subspan %arg_b[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %block_id_is_0 = arith.muli %workgroup_id_0, %c16 overflow<nsw, nuw> : index // We know we're only using 1 block, so this is just 0
        %x_mod_16 = arith.remsi %thread_id_x, %c16 : index
        %x_mod_16_2 = arith.addi %x_mod_16, %block_id_is_0 overflow<nsw, nuw> : index
        %x_mod_64 = arith.remsi %thread_id_x, %c64 : index // We know we only have 64 threads per workgroup, so this is just x
        %x_div_16 = arith.divsi %x_mod_64, %c16 : index
        %6 = arith.muli %x_div_16, %c4 overflow<nsw, nuw> : index
        %b_reg = vector.load %b[%x_mod_16_2, %6] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        %a = stream.binding.subspan %arg_a[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %c = stream.binding.subspan %arg_c[%c0] : !stream.binding -> memref<16x16xf32, strided<[16, 1], offset: ?>>
        %e = stream.binding.subspan %arg_e[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %a_reg = vector.load %a[%x_mod_16, %6] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        %c_reg = amdgpu.mfma %a_reg * %b_reg + %c0_4xf32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>

        // store c
        %13 = vector.extract_strided_slice %c_reg {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %13, %c[%6, %x_mod_16_2] : memref<16x16xf32, strided<[16, 1], offset: ?>>, vector<1xf32>
        %14 = vector.extract_strided_slice %c_reg {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %15 = arith.addi %6, %c1 overflow<nsw, nuw> : index
        vector.store %14, %c[%15, %x_mod_16_2] : memref<16x16xf32, strided<[16, 1], offset: ?>>, vector<1xf32>
        %16 = vector.extract_strided_slice %c_reg {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %17 = arith.addi %6, %c2 overflow<nsw, nuw> : index
        vector.store %16, %c[%17, %x_mod_16_2] : memref<16x16xf32, strided<[16, 1], offset: ?>>, vector<1xf32>
        %18 = vector.extract_strided_slice %c_reg {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %19 = arith.addi %6, %c3 overflow<nsw, nuw> : index
        vector.store %18, %c[%19, %x_mod_16_2] : memref<16x16xf32, strided<[16, 1], offset: ?>>, vector<1xf32>

        %e_reg = vector.load %e[%x_mod_16_2, %6] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<4xf16>
        // GPU shuffle here

        %c64_i32 = arith.constant 64 : i32

        %a_reg_ext = arith.extf %a_reg : vector<4xf16> to vector<4xf32>

        %a_reg_0 = vector.extract %a_reg_ext[0] : f32 from vector<4xf32>
        %a_reg_1 = vector.extract %a_reg_ext[1] : f32 from vector<4xf32>
        %a_reg_2 = vector.extract %a_reg_ext[2] : f32 from vector<4xf32>
        %a_reg_3 = vector.extract %a_reg_ext[3] : f32 from vector<4xf32>

        %x_mod_16_mul_16 = arith.muli %x_mod_16, %c16 overflow<nsw, nuw> : index
        %srcIdx0_idx = arith.addi %x_mod_16_mul_16, %x_div_16 overflow<nsw, nuw> : index
        %srcIdx0_i32 = arith.index_cast %srcIdx0_idx : index to i32

        %a_t_reg_0, %valid00 = gpu.shuffle idx %a_reg_0, %srcIdx0_i32, %c64_i32 : f32



        %d_reg = amdgpu.mfma %e_reg * %a_reg + %c0_4xf32 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %d_reg_trunc = arith.truncf %d_reg : vector<4xf32> to vector<4xf16>
        // store d
        %23 = vector.extract_strided_slice %d_reg_trunc {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
        %d = stream.binding.subspan %arg_d[%c0] : !stream.binding -> memref<16x16xf16, strided<[16, 1], offset: ?>>
        %25 = arith.addi %block_id_is_0, %6 overflow<nsw, nuw> : index
        vector.store %23, %d[%25, %x_mod_16] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
        %26 = vector.extract_strided_slice %d_reg_trunc {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
        %27 = arith.addi %25, %c1 overflow<nsw, nuw> : index
        vector.store %26, %d[%27, %x_mod_16] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
        %28 = vector.extract_strided_slice %d_reg_trunc {offsets = [2], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
        %29 = arith.addi %25, %c2 overflow<nsw, nuw> : index
        vector.store %28, %d[%29, %x_mod_16] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
        %30 = vector.extract_strided_slice %d_reg_trunc {offsets = [3], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
        %31 = arith.addi %25, %c3 overflow<nsw, nuw> : index
        vector.store %30, %d[%31, %x_mod_16] : memref<16x16xf16, strided<[16, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg_a: tensor<16x16xf16>, %arg_b: tensor<16x16xf16>, %arg_e: tensor<16x16xf16>, %arg_c: tensor<16x16xf32>, %arg_d: tensor<16x16xf16>) -> (tensor<16x16xf32>, tensor<16x16xf16>) {
    %b:2 = flow.dispatch @repro_603::@repro_603(%arg_a, %arg_b, %arg_e, %arg_c, %arg_d) : (tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf32>, tensor<16x16xf16>) -> (%arg_c, %arg_d)
    return %b#0, %b#1 : tensor<16x16xf32>, tensor<16x16xf16>
  }
}
