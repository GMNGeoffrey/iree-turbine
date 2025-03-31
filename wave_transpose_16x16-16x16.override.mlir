#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [16, 4, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>
module attributes {transform.with_named_sequence} {
  stream.executable private @transpose {
    stream.executable.export public @transpose workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @transpose(%arg_a: !stream.binding, %arg_a_transpose: !stream.binding) attributes {translation_info = #translation} {
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %a = stream.binding.subspan %arg_a[%c0] : !stream.binding -> memref<16x16xi32, strided<[16, 1], offset: ?>>
        %a_transpose = stream.binding.subspan %arg_a_transpose[%c0] : !stream.binding -> memref<16x16xi32, strided<[16, 1], offset: ?>>
        %1 = arith.muli %workgroup_id_0, %c16 overflow<nsw, nuw> : index
        %2 = arith.addi %1, %thread_id_x overflow<nsw, nuw> : index
        %3 = arith.muli %thread_id_y, %c4 overflow<nsw, nuw> : index
        %4 = arith.muli %workgroup_id_1, %c16 overflow<nsw, nuw> : index
        %5 = arith.addi %4, %3 overflow<nsw, nuw> : index
        %a_reg = vector.load %a[%2, %5] : memref<16x16xi32, strided<[16, 1], offset: ?>>, vector<4xi32>

        %a_t_reg_empty = arith.constant dense<0> : vector<4xi32>

        %wave_size_i32 = arith.constant 64 : i32
        %els_per_thread = arith.constant 4 : index

        %a_t_reg = scf.for %shuff_i = %c0 to %c4 step %c1 iter_args(%a_t_reg_prev = %a_t_reg_empty) -> (vector<4xi32>) {
          %tmp_00 = arith.addi %thread_id_x, %els_per_thread : index
          %tmp_01 = arith.subi %tmp_00, %shuff_i : index
          %offer_offset = arith.remsi %tmp_01, %els_per_thread : index

          %tmp_02 = arith.addi %thread_id_x, %shuff_i : index
          %dest_offset = arith.remsi %tmp_02, %els_per_thread : index

          %src_thread_x_base = arith.muli %thread_id_y, %els_per_thread : index
          %src_thread_x = arith.addi %src_thread_x_base, %dest_offset : index
          %src_thread_y = arith.divsi %thread_id_x, %c4 : index

          %src_thread_y_offset = arith.muli %src_thread_y, %c16 : index
          %src_thread_linear = arith.addi %src_thread_x, %src_thread_y_offset : index
          %src_thread_linear_i32 = arith.index_cast %src_thread_linear : index to i32

          %offer_val = vector.extract %a_reg[%offer_offset] : i32 from vector<4xi32>

          %shuffVal0, %valid0 = gpu.shuffle idx %offer_val, %src_thread_linear_i32, %wave_size_i32 : i32

          %a_t_reg_next = vector.insert %shuffVal0, %a_t_reg_prev[%dest_offset] : i32 into vector<4xi32>
          scf.yield %a_t_reg_next : vector<4xi32>
        }

        vector.store %a_t_reg, %a_transpose[%2, %5] : memref<16x16xi32, strided<[16, 1], offset: ?>>, vector<4xi32>

        return
      }
    }
  }
  func.func @isolated_benchmark(%arg_a: tensor<16x16xi32>, %arg_a_transpose: tensor<16x16xi32>) -> tensor<16x16xi32> {
    %a = flow.dispatch @transpose::@transpose(%arg_a, %arg_a_transpose) : (tensor<16x16xi32>, tensor<16x16xi32>) -> %arg_a_transpose
    return %a : tensor<16x16xi32>
  }
}
