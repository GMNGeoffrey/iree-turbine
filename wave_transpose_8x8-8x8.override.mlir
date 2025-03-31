#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [8, 8, 1] subgroup_size = 64, {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}}>
module attributes {transform.with_named_sequence} {
  stream.executable private @transpose {
    stream.executable.export public @transpose workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @transpose(%arg_a: !stream.binding, %arg_a_transpose: !stream.binding) attributes {translation_info = #translation} {
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %a = stream.binding.subspan %arg_a[%c0] : !stream.binding -> memref<8x8xi32, strided<[8, 1], offset: ?>>
        %a_transpose = stream.binding.subspan %arg_a_transpose[%c0] : !stream.binding -> memref<8x8xi32, strided<[8, 1], offset: ?>>

        %wg0_offset = arith.muli %workgroup_id_0, %c8 overflow<nsw, nuw> : index
        %x_index = arith.addi %wg0_offset, %thread_id_x overflow<nsw, nuw> : index
        %wg1_offset = arith.muli %workgroup_id_1, %c8 overflow<nsw, nuw> : index
        %y_index = arith.addi %wg1_offset, %thread_id_y overflow<nsw, nuw> : index
        %a_reg = vector.load %a[%x_index, %y_index] : memref<8x8xi32, strided<[8, 1], offset: ?>>, vector<1xi32>

        %row_offset = arith.muli %x_index, %c8 : index
        %flat_offset = arith.addi %row_offset, %y_index : index
        %flat_offset_i32 = arith.index_cast %flat_offset : index to i32
        %threads_per_wave_i32 = arith.constant 64 : i32

        %val = vector.extract %a_reg[0] : i32 from vector<1xi32>

        %shuffledVal, %valid = gpu.shuffle idx %val, %flat_offset_i32, %threads_per_wave_i32 : i32

        %shuffledVec = vector.from_elements %shuffledVal : vector<1xi32>

        vector.store %shuffledVec, %a_transpose[%x_index, %y_index] : memref<8x8xi32, strided<[8, 1], offset: ?>>, vector<1xi32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg_a: tensor<8x8xi32>, %arg_a_transpose: tensor<8x8xi32>) -> tensor<8x8xi32> {
    %a = flow.dispatch @transpose::@transpose(%arg_a, %arg_a_transpose) : (tensor<8x8xi32>, tensor<8x8xi32>) -> %arg_a_transpose
    return %a : tensor<8x8xi32>
  }
}
