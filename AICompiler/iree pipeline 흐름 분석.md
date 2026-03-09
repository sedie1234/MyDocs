- 아래의 example.mlir을 변환 시키고, 변환되는 과정을 dump한 파일을 확인하며 iree pipeline의 흐름을 분석
```
func.func @simple_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.add ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
                  outs(%arg0 : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
```
- 변환 및 dump
```
$ ./iree-compile example.mlir \
--iree-hal-target-backends=llvm-cpu \
--mlir-print-ir-after-all \
--mlir-elide-elementsattrs-if-larger=32 \
-o test.vmfb > full_pipeline.mlir 2>&1
```
- dump file
[full_pipeline.mlir](full_pipeline.mlir)

# ConvertToLLVMPass
-  flow, stream, hal 변환 결과를 llvm으로 고정
- 전
```
// -----// IR Dump After CSE (cse) //----- //

func.func @simple_add_dispatch_0_elementwise_4_f32() {

  %c0 = arith.constant 0 : index

  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<4xf32>

  %assume_align = memref.assume_alignment %0, 64 : memref<4xf32>

  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<4xf32>

  %assume_align_0 = memref.assume_alignment %1, 64 : memref<4xf32>

  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<4xf32>

  %assume_align_1 = memref.assume_alignment %2, 64 : memref<4xf32>

  %3 = vector.load %assume_align[%c0] : memref<4xf32>, vector<4xf32>

  %4 = vector.load %assume_align_0[%c0] : memref<4xf32>, vector<4xf32>

  %5 = arith.addf %3, %4 : vector<4xf32>

  vector.store %5, %assume_align_1[%c0] : memref<4xf32>, vector<4xf32>

  return

}
```
- 후
```
// -----// IR Dump After ConvertToLLVMPass (iree-convert-to-llvm) //----- //

module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-unknown-eabi-elf"} {

  llvm.func @simple_add_dispatch_0_elementwise_4_f32(%arg0: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg2: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}) -> i32 {

    %0 = llvm.mlir.constant(0 : i32) : i32

    %1 = llvm.mlir.constant(64 : index) : i64

    %2 = llvm.mlir.constant(true) : i1

    %3 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

    %4 = llvm.extractvalue %3[10] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

    %5 = llvm.load %4 : !llvm.ptr -> !llvm.ptr

    llvm.intr.assume %2 ["align"(%5, %1 : !llvm.ptr, i64)] : i1

    %6 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

    %7 = llvm.extractvalue %6[10] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

    %8 = llvm.getelementptr %7[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr

    %9 = llvm.load %8 : !llvm.ptr -> !llvm.ptr

    llvm.intr.assume %2 ["align"(%9, %1 : !llvm.ptr, i64)] : i1

    %10 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

    %11 = llvm.extractvalue %10[10] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

    %12 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr

    %13 = llvm.load %12 : !llvm.ptr -> !llvm.ptr

    llvm.intr.assume %2 ["align"(%13, %1 : !llvm.ptr, i64)] : i1

    %14 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>

    %15 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>

    %16 = llvm.fadd %14, %15 : vector<4xf32>

    llvm.store %16, %13 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr

    llvm.return %0 : i32

  }

}
```
# ConvertToHALPass
- llvm으로 고정시켜 둔 IR로부터 HAL을 생성성

- 전
```
// -----// IR Dump After ConvertToLLVMPass (iree-convert-to-llvm) //----- //

module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-unknown-eabi-elf"} {

  llvm.func @simple_add_dispatch_0_elementwise_4_f32(%arg0: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg2: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}) -> i32 {

    %0 = llvm.mlir.constant(0 : i32) : i32

    %1 = llvm.mlir.constant(64 : index) : i64

    %2 = llvm.mlir.constant(true) : i1

    %3 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

    %4 = llvm.extractvalue %3[10] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

    %5 = llvm.load %4 : !llvm.ptr -> !llvm.ptr

    llvm.intr.assume %2 ["align"(%5, %1 : !llvm.ptr, i64)] : i1

    %6 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

    %7 = llvm.extractvalue %6[10] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

    %8 = llvm.getelementptr %7[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr

    %9 = llvm.load %8 : !llvm.ptr -> !llvm.ptr

    llvm.intr.assume %2 ["align"(%9, %1 : !llvm.ptr, i64)] : i1

    %10 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

    %11 = llvm.extractvalue %10[10] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

    %12 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr

    %13 = llvm.load %12 : !llvm.ptr -> !llvm.ptr

    llvm.intr.assume %2 ["align"(%13, %1 : !llvm.ptr, i64)] : i1

    %14 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>

    %15 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>

    %16 = llvm.fadd %14, %15 : vector<4xf32>

    llvm.store %16, %13 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr

    llvm.return %0 : i32

  }

}
```
- 후
```
// -----// IR Dump After ConvertToHALPass (iree-hal-conversion) //----- //

#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

#device_target_local = #hal.device.target<"local", [#executable_target_embedded_elf_x86_64]> : !hal.device

module {

  util.global private @__device_0 = #device_target_local

  hal.executable private @simple_add_dispatch_0 {

    hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64) {

      hal.executable.export public @simple_add_dispatch_0_elementwise_4_f32 ordinal(0) layout(#pipeline_layout) attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}

      builtin.module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-unknown-eabi-elf"} {

        llvm.func @simple_add_dispatch_0_elementwise_4_f32(%arg0: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}, %arg2: !llvm.ptr {llvm.align = 16 : i64, llvm.noalias, llvm.nonnull, llvm.noundef}) -> i32 {

          %0 = llvm.mlir.constant(0 : i32) : i32

          %1 = llvm.mlir.constant(64 : index) : i64

          %2 = llvm.mlir.constant(true) : i1

          %3 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

          %4 = llvm.extractvalue %3[10] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

          %5 = llvm.load %4 : !llvm.ptr -> !llvm.ptr

          llvm.intr.assume %2 ["align"(%5, %1 : !llvm.ptr, i64)] : i1

          %6 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

          %7 = llvm.extractvalue %6[10] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

          %8 = llvm.getelementptr %7[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr

          %9 = llvm.load %8 : !llvm.ptr -> !llvm.ptr

          llvm.intr.assume %2 ["align"(%9, %1 : !llvm.ptr, i64)] : i1

          %10 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

          %11 = llvm.extractvalue %10[10] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t", (i32, i32, i16, i16, i32, i32, i16, i8, i8, ptr, ptr, ptr)>

          %12 = llvm.getelementptr %11[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr

          %13 = llvm.load %12 : !llvm.ptr -> !llvm.ptr

          llvm.intr.assume %2 ["align"(%13, %1 : !llvm.ptr, i64)] : i1

          %14 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>

          %15 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> vector<4xf32>

          %16 = llvm.fadd %14, %15 {fastmathFlags = #llvm.fastmath<contract>} : vector<4xf32>

          llvm.store %16, %13 {alignment = 4 : i64} : vector<4xf32>, !llvm.ptr

          llvm.return %0 : i32

        }

      }

    }

  }

  util.func public @simple_add(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @simple_add(%input0: tensor<4xf32>, %input1: tensor<4xf32>) -> (%output0: tensor<4xf32>)"}} {

    %c0 = arith.constant 0 : index

    %c16 = arith.constant 16 : index

    %c4 = arith.constant 4 : index

    %element_type_f32 = hal.element_type<f32> : i32

    %dense_row_major = hal.encoding_type<dense_row_major> : i32

    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c4]) type(%element_type_f32) encoding(%dense_row_major)

    %buffer = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer

    %__device_0 = util.global.load immutable @__device_0 : !hal.device

    %allocator = hal.device.allocator<%__device_0 : !hal.device> : !hal.allocator

    hal.buffer.assert<%buffer : !hal.buffer> message("tensor") allocator(%allocator : !hal.allocator) minimum_length(%c16) type(DeviceVisible) usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage")

    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%c4]) type(%element_type_f32) encoding(%dense_row_major)

    %buffer_0 = hal.buffer_view.buffer<%arg1 : !hal.buffer_view> : !hal.buffer

    %__device_0_1 = util.global.load immutable @__device_0 : !hal.device

    %allocator_2 = hal.device.allocator<%__device_0_1 : !hal.device> : !hal.allocator

    hal.buffer.assert<%buffer_0 : !hal.buffer> message("tensor") allocator(%allocator_2 : !hal.allocator) minimum_length(%c16) type(DeviceVisible) usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage")

    %memory_types, %buffer_usage = hal.allocator.resolve_memory_properties for(#hal.device.affinity<@__device_0>) lifetime(external) : i32, i32

    %__device_0_3 = util.global.load immutable @__device_0 : !hal.device

    %c-1_i64 = arith.constant -1 : i64

    %0 = util.null : !hal.fence

    %fence = hal.fence.create device(%__device_0_3 : !hal.device) flags("None") : !hal.fence

    %c0_i64 = arith.constant 0 : i64

    %transient_buffer = hal.device.queue.alloca<%__device_0_3 : !hal.device> affinity(%c-1_i64) wait(%0) signal(%fence) pool(%c0_i64) type(%memory_types) usage(%buffer_usage) flags("None") : !hal.buffer{%c16}

    %__device_0_4 = util.global.load immutable @__device_0 : !hal.device

    %c-1_i64_5 = arith.constant -1 : i64

    %c0_6 = arith.constant 0 : index

    %c1 = arith.constant 1 : index

    %c2 = arith.constant 2 : index

    %1 = hal.device.memoize<%__device_0_4 : !hal.device> affinity(%c-1_i64_5) -> !hal.command_buffer {

      %c3 = arith.constant 3 : index

      %cmd = hal.command_buffer.create device(%__device_0_4 : !hal.device) mode("None") categories("Transfer|Dispatch") affinity(%c-1_i64_5) bindings(%c3) : !hal.command_buffer

      %2 = hal.command_buffer.device<%cmd : !hal.command_buffer> : !hal.device

      %exe = hal.executable.lookup device(%2 : !hal.device) executable(@simple_add_dispatch_0) : !hal.executable

      %ordinal = hal.executable.export.ordinal target(@simple_add_dispatch_0::@embedded_elf_x86_64::@simple_add_dispatch_0_elementwise_4_f32) : index

      %c1_12 = arith.constant 1 : index

      %c1_13 = arith.constant 1 : index

      %c1_14 = arith.constant 1 : index

      hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%c1_12, %c1_13, %c1_14]) bindings([

        (%c0_6 : index)[%c0, %c16],

        (%c1 : index)[%c0, %c16],

        (%c2 : index)[%c0, %c16]

      ]) flags("None")

      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|Transfer|CommandRetire") target("CommandIssue|Dispatch|Transfer") flags("None")

      hal.command_buffer.finalize<%cmd : !hal.command_buffer>

      hal.return %cmd : !hal.command_buffer

    }

    %fence_7 = hal.fence.create device(%__device_0_4 : !hal.device) flags("None") : !hal.fence

    hal.device.queue.execute.indirect<%__device_0_4 : !hal.device> affinity(%c-1_i64_5) wait(%fence) signal(%fence_7) commands(%1) bindings([

      (%buffer : !hal.buffer)[%c0_6, %c16],

      (%buffer_0 : !hal.buffer)[%c0_6, %c16],

      (%transient_buffer : !hal.buffer)[%c0_6, %c16]

    ]) flags("None")

    %c-1_i32 = arith.constant -1 : i32

    %status = hal.fence.await until([%fence_7]) timeout_millis(%c-1_i32) flags("None") : i32

    util.status.check_ok %status, "failed to wait on timepoint"

    %dense_row_major_8 = hal.encoding_type<dense_row_major> : i32

    %element_type_f32_9 = hal.element_type<f32> : i32

    %c4_10 = arith.constant 4 : index

    %c0_11 = arith.constant 0 : index

    %view = hal.buffer_view.create buffer(%transient_buffer : !hal.buffer)[%c0_11, %c16] shape([%c4_10]) type(%element_type_f32_9) encoding(%dense_row_major_8) : !hal.buffer_view

    util.return %view : !hal.buffer_view

  }

}
```


# ConversionPass (iree-vm-conversion)
- hal 및 그 상위 레벨에서 변환 및 최적화를 수행하던 IR로부터 실제 vm code를 만들어냄
- 코드가 본격적으로 hal과 vm으로 나뉘기 시작하며, hal명령을 vm에 등록하고 vm은 등록된 hal을 조합하여 연산실행계획을 세운다.
- vm이 호출할 수 있는 hal 명령어는 이 때 고정되기 때문에 backend를 구현하기 위해서는 이 구간의 hal 명령어들을 확인하여 pass를 구현
- 전
```
// -----// IR Dump After FuseGlobalsPass (iree-util-fuse-globals) //----- //

module {

  util.global private @__device_0 : !hal.device

  util.global private @__device_0_executable_0_simple_add_dispatch_0 : !hal.executable

  util.global private @__simple_add_memoize_result_0_device_0 : !hal.command_buffer

  util.initializer {

    %0 = util.null : !hal.executable

    %c14_i32 = arith.constant 14 : i32

    %c-1 = arith.constant -1 : index

    %c-1_i64 = arith.constant -1 : i64

    %c18_i32 = arith.constant 18 : i32

    %false = arith.constant false

    %c0 = arith.constant 0 : index

    %c1 = arith.constant 1 : index

    %1 = util.null : !hal.device

    %device_count = hal.devices.count : index

    cf.br ^bb1(%c0, %c0, %1 : index, index, !hal.device)

  ^bb1(%2: index, %3: index, %4: !hal.device):  // 2 preds: ^bb0, ^bb4

    %5 = util.cmp.eq %4, %1 : !hal.device

    %6 = arith.cmpi slt, %2, %device_count : index

    %7 = arith.andi %5, %6 : i1

    cf.cond_br %7, ^bb2, ^bb5

  ^bb2:  // pred: ^bb1

    %device_n = hal.devices.get %2 : !hal.device

    %ok, %value = hal.device.query<%device_n : !hal.device> key("hal.device.id" :: "local*") : i1, i1 = false

    cf.cond_br %value, ^bb3, ^bb4(%false : i1)

  ^bb3:  // pred: ^bb2

    %ok_0, %value_1 = hal.device.query<%device_n : !hal.device> key("hal.executable.format" :: "embedded-elf-x86_64") : i1, i1 = false

    cf.br ^bb4(%value_1 : i1)

  ^bb4(%8: i1):  // 2 preds: ^bb2, ^bb3

    %9 = arith.cmpi eq, %3, %c0 : index

    %10 = arith.select %8, %c1, %c0 : index

    %11 = arith.addi %3, %10 : index

    %12 = arith.andi %8, %9 : i1

    %13 = arith.select %12, %device_n, %1 : !hal.device

    %14 = arith.addi %2, %c1 : index

    cf.br ^bb1(%14, %11, %13 : index, index, !hal.device)

  ^bb5:  // pred: ^bb1

    cf.cond_br %5, ^bb6, ^bb7

  ^bb6:  // pred: ^bb5

    util.status.check_ok %c18_i32, "HAL device `__device_0` not found or unavailable: #hal.device.target<\22local\22, [#hal.executable.target<\22llvm-cpu\22, \22embedded-elf-x86_64\22, {cpu = \22\22, cpu_features = \22\22, data_layout = \22e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\22, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = \22x86_64-unknown-unknown-eabi-elf\22}>]>"

    cf.br ^bb7

  ^bb7:  // 2 preds: ^bb5, ^bb6

    %ok_2, %value_3 = hal.device.query<%4 : !hal.device> key("hal.executable.format" :: "embedded-elf-x86_64") : i1, i1 = false

    %15 = arith.select %value_3, %c0, %c-1 : index

    %16 = arith.cmpi eq, %15, %c0 : index

    util.global.store %4, @__device_0 : !hal.device

    cf.cond_br %16, ^bb8, ^bb9

  ^bb8:  // pred: ^bb7

    %executable = hal.executable.create device(%4 : !hal.device) affinity(%c-1_i64) target(@simple_add_dispatch_0::@embedded_elf_x86_64) : !hal.executable

    cf.br ^bb10(%executable : !hal.executable)

  ^bb9:  // pred: ^bb7

    util.status.check_ok %c14_i32, "HAL device `__device_0` does not support any variant of executable `simple_add_dispatch_0`; available formats: [embedded-elf-x86_64]"

    cf.br ^bb10(%0 : !hal.executable)

  ^bb10(%17: !hal.executable):  // 2 preds: ^bb8, ^bb9

    util.global.store %17, @__device_0_executable_0_simple_add_dispatch_0 : !hal.executable

    %18 = util.call @__simple_add_memoize_apply() : () -> !hal.command_buffer

    util.global.store %18, @__simple_add_memoize_result_0_device_0 : !hal.command_buffer

    util.return

  }

  hal.executable private @simple_add_dispatch_0 {

    hal.executable.binary public @embedded_elf_x86_64 attributes {data = dense_resource<__elided__> : vector<3592xi8>, format = "embedded-elf-x86_64", mime_type = "application/x-elf"}

  }

  util.func private @__simple_add_memoize_apply() -> !hal.command_buffer attributes {inlining_policy = #util.inline.never} {

    %c16 = arith.constant 16 : index

    %c2 = arith.constant 2 : index

    %c1 = arith.constant 1 : index

    %c0 = arith.constant 0 : index

    %c3 = arith.constant 3 : index

    %c-1_i64 = arith.constant -1 : i64

    %__device_0 = util.global.load immutable @__device_0 : !hal.device

    %__device_0_executable_0_simple_add_dispatch_0 = util.global.load immutable @__device_0_executable_0_simple_add_dispatch_0 : !hal.executable

    %cmd = hal.command_buffer.create device(%__device_0 : !hal.device) mode("None") categories("Transfer|Dispatch") affinity(%c-1_i64) bindings(%c3) : !hal.command_buffer

    hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%__device_0_executable_0_simple_add_dispatch_0 : !hal.executable)[%c0] workgroups([%c1, %c1, %c1]) bindings([

      (%c0 : index)[%c0, %c16],

      (%c1 : index)[%c0, %c16],

      (%c2 : index)[%c0, %c16]

    ]) flags("None")

    hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|Transfer|CommandRetire") target("CommandIssue|Dispatch|Transfer") flags("None")

    hal.command_buffer.finalize<%cmd : !hal.command_buffer>

    util.return %cmd : !hal.command_buffer

  }

  util.func public @simple_add(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @simple_add(%input0: tensor<4xf32>, %input1: tensor<4xf32>) -> (%output0: tensor<4xf32>)"}} {

    %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage"> : i32

    %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32

    %c4 = arith.constant 4 : index

    %c16 = arith.constant 16 : index

    %c0 = arith.constant 0 : index

    %c-1_i64 = arith.constant -1 : i64

    %0 = util.null : !hal.fence

    %c0_i64 = arith.constant 0 : i64

    %c-1_i32 = arith.constant -1 : i32

    %__device_0 = util.global.load immutable @__device_0 : !hal.device

    %__simple_add_memoize_result_0_device_0 = util.global.load immutable @__simple_add_memoize_result_0_device_0 : !hal.command_buffer

    %element_type_f32 = hal.element_type<f32> : i32

    %dense_row_major = hal.encoding_type<dense_row_major> : i32

    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c4]) type(%element_type_f32) encoding(%dense_row_major)

    %buffer = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer

    %allocator = hal.device.allocator<%__device_0 : !hal.device> : !hal.allocator

    hal.buffer.assert<%buffer : !hal.buffer> message("tensor") allocator(%allocator : !hal.allocator) minimum_length(%c16) type(DeviceVisible) usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage")

    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%c4]) type(%element_type_f32) encoding(%dense_row_major)

    %buffer_0 = hal.buffer_view.buffer<%arg1 : !hal.buffer_view> : !hal.buffer

    hal.buffer.assert<%buffer_0 : !hal.buffer> message("tensor") allocator(%allocator : !hal.allocator) minimum_length(%c16) type(DeviceVisible) usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage")

    %fence = hal.fence.create device(%__device_0 : !hal.device) flags("None") : !hal.fence

    %transient_buffer = hal.device.queue.alloca<%__device_0 : !hal.device> affinity(%c-1_i64) wait(%0) signal(%fence) pool(%c0_i64) type(%memory_type) usage(%buffer_usage) flags("None") : !hal.buffer{%c16}

    %fence_1 = hal.fence.create device(%__device_0 : !hal.device) flags("None") : !hal.fence

    hal.device.queue.execute.indirect<%__device_0 : !hal.device> affinity(%c-1_i64) wait(%fence) signal(%fence_1) commands(%__simple_add_memoize_result_0_device_0) bindings([

      (%buffer : !hal.buffer)[%c0, %c16],

      (%buffer_0 : !hal.buffer)[%c0, %c16],

      (%transient_buffer : !hal.buffer)[%c0, %c16]

    ]) flags("None")

    %status = hal.fence.await until([%fence_1]) timeout_millis(%c-1_i32) flags("None") : i32

    util.status.check_ok %status, "failed to wait on timepoint"

    %view = hal.buffer_view.create buffer(%transient_buffer : !hal.buffer)[%c0, %c16] shape([%c4]) type(%element_type_f32) encoding(%dense_row_major) : !hal.buffer_view

    util.return %view : !hal.buffer_view

  }

}
```
- 후
```
// -----// IR Dump After ConversionPass (iree-vm-conversion) //----- //

module attributes {vm.toplevel} {

  vm.module public @module {

    vm.global.ref private @__device_0 : !vm.ref<!hal.device>

    vm.global.ref private @__device_0_executable_0_simple_add_dispatch_0 : !vm.ref<!hal.executable>

    vm.global.ref private @__simple_add_memoize_result_0_device_0 : !vm.ref<!hal.command_buffer>

    vm.initializer {

      %null = vm.const.ref.zero : !vm.ref<!hal.executable>

      %c14 = vm.const.i32 14

      %c-1 = vm.const.i64 -1

      %c-1_0 = vm.const.i64 -1

      %c18 = vm.const.i32 18

      %zero = vm.const.i32.zero

      %zero_1 = vm.const.i64.zero

      %c1 = vm.const.i64 1

      %null_2 = vm.const.ref.zero : !vm.ref<!hal.device>

      %0 = vm.call @hal.devices.count() {nosideeffects} : () -> i32

      %1 = vm.ext.i32.i64.s %0 : i32 -> i64

      vm.br ^bb1(%zero_1, %zero_1, %null_2 : i64, i64, !vm.ref<!hal.device>)

    ^bb1(%2: i64, %3: i64, %4: !vm.ref<!hal.device>):  // 2 preds: ^bb0, ^bb4

      %req = vm.cmp.eq.ref %4, %null_2 : !vm.ref<!hal.device>

      %slt = vm.cmp.lt.i64.s %2, %1 : i64

      %5 = vm.and.i32 %req, %slt : i32

      vm.cond_br %5, ^bb2, ^bb5

    ^bb2:  // pred: ^bb1

      %6 = vm.trunc.i64.i32 %2 : i64 -> i32

      %ref = vm.call @hal.devices.get(%6) {nosideeffects} : (i32) -> !vm.ref<!hal.device>

      %buffer = vm.rodata.inline "_utf8_hal_device_id_C6650FF277232B5A" {alignment = 1 : i64} : !vm.buffer = "hal.device.id"

      %buffer_3 = vm.rodata.inline "_utf8_local_1A8FF0278D7661D8" {alignment = 1 : i64} : !vm.buffer = "local*"

      %7:2 = vm.call @hal.device.query.i64(%ref, %buffer, %buffer_3) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i64)

      %nz = vm.cmp.nz.i64 %7#1 : i64

      %zero_4 = vm.const.i32.zero

      %8 = vm.select.i32 %7#0, %nz, %zero_4 : i32

      %c1_5 = vm.const.i32 1

      vm.cond_br %8, ^bb3, ^bb4(%zero : i32)

    ^bb3:  // pred: ^bb2

      %buffer_6 = vm.rodata.inline "_utf8_hal_executable_format_E03EECB63A2AAF52" {alignment = 1 : i64} : !vm.buffer = "hal.executable.format"

      %buffer_7 = vm.rodata.inline "_utf8_embedded_elf_x86_64_FF16E34B4A5F9C83" {alignment = 1 : i64} : !vm.buffer = "embedded-elf-x86_64"

      %9:2 = vm.call @hal.device.query.i64(%ref, %buffer_6, %buffer_7) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i64)

      %nz_8 = vm.cmp.nz.i64 %9#1 : i64

      %zero_9 = vm.const.i32.zero

      %10 = vm.select.i32 %9#0, %nz_8, %zero_9 : i32

      %c1_10 = vm.const.i32 1

      vm.br ^bb4(%10 : i32)

    ^bb4(%11: i32):  // 2 preds: ^bb2, ^bb3

      %eq = vm.cmp.eq.i64 %3, %zero_1 : i64

      %12 = vm.select.i64 %11, %c1, %zero_1 : i64

      %13 = vm.add.i64 %3, %12 : i64

      %14 = vm.and.i32 %11, %eq : i32

      %ref_11 = vm.select.ref %14, %ref, %null_2 : !vm.ref<!hal.device>

      %15 = vm.add.i64 %2, %c1 : i64

      vm.br ^bb1(%15, %13, %ref_11 : i64, i64, !vm.ref<!hal.device>)

    ^bb5:  // pred: ^bb1

      vm.cond_br %req, ^bb6, ^bb7

    ^bb6:  // pred: ^bb5

      vm.cond_fail %c18, "HAL device `__device_0` not found or unavailable: #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>]>"

      vm.br ^bb7

    ^bb7:  // 2 preds: ^bb5, ^bb6

      %buffer_12 = vm.rodata.inline "_utf8_hal_executable_format_E03EECB63A2AAF52" {alignment = 1 : i64} : !vm.buffer = "hal.executable.format"

      %buffer_13 = vm.rodata.inline "_utf8_embedded_elf_x86_64_FF16E34B4A5F9C83" {alignment = 1 : i64} : !vm.buffer = "embedded-elf-x86_64"

      %16:2 = vm.call @hal.device.query.i64(%4, %buffer_12, %buffer_13) {nosideeffects} : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i64)

      %nz_14 = vm.cmp.nz.i64 %16#1 : i64

      %zero_15 = vm.const.i32.zero

      %17 = vm.select.i32 %16#0, %nz_14, %zero_15 : i32

      %c1_16 = vm.const.i32 1

      %18 = vm.select.i64 %17, %zero_1, %c-1 : i64

      %eq_17 = vm.cmp.eq.i64 %18, %zero_1 : i64

      vm.global.store.ref %4, @__device_0 : !vm.ref<!hal.device>

      vm.cond_br %eq_17, ^bb8, ^bb9

    ^bb8:  // pred: ^bb7

      %buffer_18 = vm.rodata.inline "simple_add_dispatch_0_embedded_elf_x86_64" {alignment = 16 : i64, mime_type = "application/x-elf"} : !vm.buffer = dense_resource<__elided__> : vector<3592xi8>

      %buffer_19 = vm.rodata.inline "_utf8_embedded_elf_x86_64_FF16E34B4A5F9C83" {alignment = 1 : i64} : !vm.buffer = "embedded-elf-x86_64"

      %null_20 = vm.const.ref.zero : !vm.buffer

      %ref_21 = vm.call @hal.executable.create(%4, %c-1_0, %buffer_19, %buffer_18, %null_20) {nosideeffects} : (!vm.ref<!hal.device>, i64, !vm.buffer, !vm.buffer, !vm.buffer) -> !vm.ref<!hal.executable>

      vm.br ^bb10(%ref_21 : !vm.ref<!hal.executable>)

    ^bb9:  // pred: ^bb7

      vm.cond_fail %c14, "HAL device `__device_0` does not support any variant of executable `simple_add_dispatch_0`; available formats: [embedded-elf-x86_64]"

      vm.br ^bb10(%null : !vm.ref<!hal.executable>)

    ^bb10(%19: !vm.ref<!hal.executable>):  // 2 preds: ^bb8, ^bb9

      vm.global.store.ref %19, @__device_0_executable_0_simple_add_dispatch_0 : !vm.ref<!hal.executable>

      %ref_22 = vm.call @__simple_add_memoize_apply() : () -> !vm.ref<!hal.command_buffer>

      vm.global.store.ref %ref_22, @__simple_add_memoize_result_0_device_0 : !vm.ref<!hal.command_buffer>

      vm.return

    }

    vm.func private @__simple_add_memoize_apply() -> !vm.ref<!hal.command_buffer> attributes {inlining_policy = #util.inline.never} {

      %c16 = vm.const.i64 16

      %c2 = vm.const.i64 2

      %c1 = vm.const.i64 1

      %zero = vm.const.i64.zero

      %c3 = vm.const.i64 3

      %c-1 = vm.const.i64 -1

      %__device_0 = vm.global.load.ref immutable @__device_0 : !vm.ref<!hal.device>

      %__device_0_executable_0_simple_add_dispatch_0 = vm.global.load.ref immutable @__device_0_executable_0_simple_add_dispatch_0 : !vm.ref<!hal.executable>

      %zero_0 = vm.const.i32.zero

      %c3_1 = vm.const.i32 3

      %c3_2 = vm.const.i32 3

      %ref = vm.call @hal.command_buffer.create(%__device_0, %zero_0, %c3_1, %c-1, %c3_2) : (!vm.ref<!hal.device>, i32, i32, i64, i32) -> !vm.ref<!hal.command_buffer>

      %zero_3 = vm.const.i32.zero

      %zero_4 = vm.const.i32.zero

      %c1_5 = vm.const.i32 1

      %c1_6 = vm.const.i32 1

      %c1_7 = vm.const.i32 1

      %zero_8 = vm.const.i64 0

      %zero_9 = vm.const.i32.zero

      %null = vm.const.ref.zero : !vm.ref<!hal.buffer>

      %c1_10 = vm.const.i32 1

      %null_11 = vm.const.ref.zero : !vm.ref<!hal.buffer>

      %c2_12 = vm.const.i32 2

      %null_13 = vm.const.ref.zero : !vm.ref<!hal.buffer>

      vm.call.variadic @hal.command_buffer.dispatch(%ref, %__device_0_executable_0_simple_add_dispatch_0, %zero_4, %c1_5, %c1_6, %c1_7, %zero_8, [], [(%zero_3, %zero_9, %null, %zero, %c16), (%zero_3, %c1_10, %null_11, %zero, %c16), (%zero_3, %c2_12, %null_13, %zero, %c16)]) : (!vm.ref<!hal.command_buffer>, !vm.ref<!hal.executable>, i32, i32, i32, i32, i64, i32 ..., tuple<i32, i32, !vm.ref<!hal.buffer>, i64, i64> ...)

      %c28 = vm.const.i32 28

      %c13 = vm.const.i32 13

      %zero_14 = vm.const.i64.zero

      vm.call @hal.command_buffer.execution_barrier(%ref, %c28, %c13, %zero_14) : (!vm.ref<!hal.command_buffer>, i32, i32, i64) -> ()

      vm.call @hal.command_buffer.finalize(%ref) : (!vm.ref<!hal.command_buffer>) -> ()

      vm.return %ref : !vm.ref<!hal.command_buffer>

    }

    vm.import private @hal.ex.file.from_memory(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %access : i32, %buffer : !vm.buffer, %offset : i64, %length : i64, %flags : i32) -> !vm.ref<!hal.file>

    vm.import private @hal.allocator.select(%memory_types : i32, %buffer_usage : i32, %flags : i64, %from : tuple<!vm.ref<!hal.device>, i64> ...) -> (!vm.ref<!hal.device>, i64) attributes {nosideeffects}

    vm.import private @hal.allocator.allocate(%allocator : !vm.ref<!hal.allocator>, %queue_affinity : i64, %memory_types : i32, %buffer_usage : i32, %allocation_size : i64) -> !vm.ref<!hal.buffer>

    vm.import private @hal.allocator.import(%allocator : !vm.ref<!hal.allocator>, %try : i32, %queue_affinity : i64, %memory_types : i32, %buffer_usage : i32, %source : !vm.buffer, %offset : i64, %length : i64) -> !vm.ref<!hal.buffer>

    vm.import private @hal.buffer.assert(%buffer : !vm.ref<!hal.buffer>, %message : !vm.buffer, %allocator : !vm.ref<!hal.allocator>, %minimum_length : i64, %memory_types : i32, %buffer_usage : i32)

    vm.import private @hal.buffer.allocation.preserve(%buffer : !vm.ref<!hal.buffer>)

    vm.import private @hal.buffer.allocation.discard(%buffer : !vm.ref<!hal.buffer>) -> i32

    vm.import private @hal.buffer.allocation.is_terminal(%buffer : !vm.ref<!hal.buffer>) -> i32

    vm.import private @hal.buffer.subspan(%source_buffer : !vm.ref<!hal.buffer>, %source_offset : i64, %length : i64) -> !vm.ref<!hal.buffer> attributes {nosideeffects}

    vm.import private @hal.buffer.length(%buffer : !vm.ref<!hal.buffer>) -> i64 attributes {nosideeffects}

    vm.import private @hal.buffer.load(%source_buffer : !vm.ref<!hal.buffer>, %source_offset : i64, %length : i32) -> i32

    vm.import private @hal.buffer.store(%value : i32, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i64, %length : i32)

    vm.import private @hal.buffer_view.create(%buffer : !vm.ref<!hal.buffer>, %source_offset : i64, %source_length : i64, %element_type : i32, %encoding_type : i32, %shape : i64 ...) -> !vm.ref<!hal.buffer_view> attributes {nosideeffects}

    vm.import private @hal.buffer_view.assert(%buffer_view : !vm.ref<!hal.buffer_view>, %message : !vm.buffer, %element_type : i32, %encoding_type : i32, %shape : i64 ...)

    vm.import private @hal.buffer_view.buffer(%buffer_view : !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer> attributes {nosideeffects}

    vm.import private @hal.buffer_view.element_type(%buffer_view : !vm.ref<!hal.buffer_view>) -> i32 attributes {nosideeffects}

    vm.import private @hal.buffer_view.encoding_type(%buffer_view : !vm.ref<!hal.buffer_view>) -> i32 attributes {nosideeffects}

    vm.import private @hal.buffer_view.rank(%buffer_view : !vm.ref<!hal.buffer_view>) -> i32 attributes {nosideeffects}

    vm.import private @hal.buffer_view.dim(%buffer_view : !vm.ref<!hal.buffer_view>, %index : i32) -> i64 attributes {nosideeffects}

    vm.import private @hal.buffer_view.trace(%key : !vm.buffer, %operands : !vm.ref<!hal.buffer_view> ...)

    vm.import private @hal.channel.create(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %flags : i64, %id : !vm.buffer, %group : !vm.buffer, %rank : i32, %count : i32) -> !vm.ref<!hal.channel> attributes {nosideeffects}

    vm.import private @hal.channel.split(%channel : !vm.ref<!hal.channel>, %color : i32, %key : i32, %flags : i64) -> !vm.ref<!hal.channel> attributes {nosideeffects}

    vm.import private @hal.channel.rank_and_count(%channel : !vm.ref<!hal.channel>) -> (i32, i32) attributes {nosideeffects}

    vm.import private @hal.command_buffer.create(%device : !vm.ref<!hal.device>, %modes : i32, %command_categories : i32, %queue_affinity : i64, %binding_capacity : i32) -> !vm.ref<!hal.command_buffer> attributes {minimum_version = 6 : i32}

    vm.import private @hal.command_buffer.finalize(%command_buffer : !vm.ref<!hal.command_buffer>)

    vm.import private @hal.command_buffer.begin_debug_group(%command_buffer : !vm.ref<!hal.command_buffer>, %label : !vm.buffer)

    vm.import private @hal.command_buffer.end_debug_group(%command_buffer : !vm.ref<!hal.command_buffer>)

    vm.import private @hal.command_buffer.execution_barrier(%command_buffer : !vm.ref<!hal.command_buffer>, %source_stage_mask : i32, %target_stage_mask : i32, %flags : i64)

    vm.import private @hal.command_buffer.advise_buffer(%command_buffer : !vm.ref<!hal.command_buffer>, %buffer : !vm.ref<!hal.buffer>, %flags : i64, %arg0 : i64, %arg1 : i64, %buffer_slot : i32)

    vm.import private @hal.command_buffer.fill_buffer(%command_buffer : !vm.ref<!hal.command_buffer>, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i64, %length : i64, %target_buffer_slot : i32, %pattern : i64, %pattern_length : i32, %flags : i64)

    vm.import private @hal.command_buffer.update_buffer(%command_buffer : !vm.ref<!hal.command_buffer>, %source_buffer : !vm.buffer, %source_offset : i64, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i64, %length : i64, %target_buffer_slot : i32, %flags : i64)

    vm.import private @hal.command_buffer.copy_buffer(%command_buffer : !vm.ref<!hal.command_buffer>, %source_buffer_slot : i32, %target_buffer_slot : i32, %source_buffer : !vm.ref<!hal.buffer>, %source_offset : i64, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i64, %length : i64, %flags : i64)

    vm.import private @hal.command_buffer.collective(%command_buffer : !vm.ref<!hal.command_buffer>, %channel : !vm.ref<!hal.channel>, %op : i32, %param : i32, %send_buffer_slot : i32, %recv_buffer_slot : i32, %send_buffer : !vm.ref<!hal.buffer>, %recv_buffer : !vm.ref<!hal.buffer>, %send_offset : i64, %send_length : i64, %recv_offset : i64, %recv_length : i64, %element_count : i64)

    vm.import private @hal.command_buffer.dispatch(%command_buffer : !vm.ref<!hal.command_buffer>, %executable : !vm.ref<!hal.executable>, %entry_point : i32, %workgroup_x : i32, %workgroup_y : i32, %workgroup_z : i32, %flags : i64, %constants : i32 ..., %bindings : tuple<i32, i32, !vm.ref<!hal.buffer>, i64, i64> ...)

    vm.import private @hal.command_buffer.dispatch.indirect(%command_buffer : !vm.ref<!hal.command_buffer>, %executable : !vm.ref<!hal.executable>, %entry_point : i32, %workgroups_buffer_slot : i32, %workgroups_buffer : !vm.ref<!hal.buffer>, %workgroups_offset : i64, %flags : i64, %constants : i32 ..., %bindings : tuple<i32, i32, !vm.ref<!hal.buffer>, i64, i64> ...)

    vm.import private @hal.device.allocator(%device : !vm.ref<!hal.device>) -> !vm.ref<!hal.allocator> attributes {nosideeffects}

    vm.import private @hal.device.query.i64(%device : !vm.ref<!hal.device>, %category : !vm.buffer, %key : !vm.buffer) -> (i32, i64) attributes {nosideeffects}

    vm.import private @hal.device.queue.alloca(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %wait_fence : !vm.ref<!hal.fence>, %signal_fence : !vm.ref<!hal.fence>, %pool : i64, %memory_types : i32, %buffer_usage : i32, %allocation_size : i64, %flags : i64) -> !vm.ref<!hal.buffer>

    vm.import private @hal.device.queue.dealloca(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %wait_fence : !vm.ref<!hal.fence>, %signal_fence : !vm.ref<!hal.fence>, %buffer : !vm.ref<!hal.buffer>, %flags : i64)

    vm.import private @hal.device.queue.fill(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %wait_fence : !vm.ref<!hal.fence>, %signal_fence : !vm.ref<!hal.fence>, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i64, %length : i64, %pattern : i64, %pattern_length : i32, %flags : i64)

    vm.import private @hal.device.queue.update(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %wait_fence : !vm.ref<!hal.fence>, %signal_fence : !vm.ref<!hal.fence>, %source_buffer : !vm.buffer, %source_offset : i64, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i64, %length : i64, %flags : i64)

    vm.import private @hal.device.queue.copy(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %wait_fence : !vm.ref<!hal.fence>, %signal_fence : !vm.ref<!hal.fence>, %source_buffer : !vm.ref<!hal.buffer>, %source_offset : i64, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i64, %length : i64, %flags : i64)

    vm.import private @hal.device.queue.read(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %wait_fence : !vm.ref<!hal.fence>, %signal_fence : !vm.ref<!hal.fence>, %source_file : !vm.ref<!hal.file>, %source_offset : i64, %target_buffer : !vm.ref<!hal.buffer>, %target_offset : i64, %length : i64, %flags : i64)

    vm.import private @hal.device.queue.write(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %wait_fence : !vm.ref<!hal.fence>, %signal_fence : !vm.ref<!hal.fence>, %source_buffer : !vm.ref<!hal.buffer>, %source_offset : i64, %target_file : !vm.ref<!hal.file>, %target_offset : i64, %length : i64, %flags : i64)

    vm.import private @hal.device.queue.barrier(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %wait_fence : !vm.ref<!hal.fence>, %signal_fence : !vm.ref<!hal.fence>, %flags : i64)

    vm.import private @hal.device.queue.execute(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %wait_fence : !vm.ref<!hal.fence>, %signal_fence : !vm.ref<!hal.fence>, %command_buffer : !vm.ref<!hal.command_buffer>, %flags : i64)

    vm.import private @hal.device.queue.execute.indirect(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %wait_fence : !vm.ref<!hal.fence>, %signal_fence : !vm.ref<!hal.fence>, %command_buffer : !vm.ref<!hal.command_buffer>, %flags : i64, %binding_table : tuple<!vm.ref<!hal.buffer>, i64, i64> ...)

    vm.import private @hal.device.queue.flush(%device : !vm.ref<!hal.device>, %queue_affinity : i64)

    vm.import private @hal.devices.count() -> i32 attributes {nosideeffects}

    vm.import private @hal.devices.get(%index : i32) -> !vm.ref<!hal.device> attributes {nosideeffects}

    vm.import private @hal.executable.create(%device : !vm.ref<!hal.device>, %queue_affinity : i64, %executable_format : !vm.buffer, %executable_data : !vm.buffer, %constants : !vm.buffer) -> !vm.ref<!hal.executable> attributes {nosideeffects}

    vm.import private @hal.fence.create(%device : !vm.ref<!hal.device>, %flags : i64) -> !vm.ref<!hal.fence>

    vm.import private @hal.fence.join(%flags : i64, %fences : !vm.ref<!hal.fence> ...) -> !vm.ref<!hal.fence> attributes {nosideeffects}

    vm.import private @hal.fence.query(%fence : !vm.ref<!hal.fence>) -> i32

    vm.import private @hal.fence.signal(%fence : !vm.ref<!hal.fence>)

    vm.import private @hal.fence.fail(%fence : !vm.ref<!hal.fence>, %status : i32)

    vm.import private @hal.fence.await(%timeout_millis : i32, %flags : i64, %fences : !vm.ref<!hal.fence> ...) -> i32 attributes {vm.yield}

    vm.func private @simple_add(%arg0: !vm.ref<!hal.buffer_view>, %arg1: !vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer_view> attributes {iree.reflection = {iree.abi.declaration = "sync func @simple_add(%input0: tensor<4xf32>, %input1: tensor<4xf32>) -> (%output0: tensor<4xf32>)"}} {

      %c3075 = vm.const.i32 3075

      %c48 = vm.const.i32 48

      %c4 = vm.const.i64 4

      %c16 = vm.const.i64 16

      %zero = vm.const.i64.zero

      %c-1 = vm.const.i64 -1

      %null = vm.const.ref.zero : !vm.ref<!hal.fence>

      %zero_0 = vm.const.i64.zero

      %c-1_1 = vm.const.i32 -1

      %__device_0 = vm.global.load.ref immutable @__device_0 : !vm.ref<!hal.device>

      %__simple_add_memoize_result_0_device_0 = vm.global.load.ref immutable @__simple_add_memoize_result_0_device_0 : !vm.ref<!hal.command_buffer>

      %c553648160 = vm.const.i32 553648160

      %c1 = vm.const.i32 1

      %buffer = vm.rodata.inline "_utf8_input0_DCE99660CEB3F6B" {alignment = 1 : i64} : !vm.buffer = "input0"

      vm.call.variadic @hal.buffer_view.assert(%arg0, %buffer, %c553648160, %c1, [%c4]) : (!vm.ref<!hal.buffer_view>, !vm.buffer, i32, i32, i64 ...)

      %ref = vm.call @hal.buffer_view.buffer(%arg0) {nosideeffects} : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>

      %ref_2 = vm.call @hal.device.allocator(%__device_0) {nosideeffects} : (!vm.ref<!hal.device>) -> !vm.ref<!hal.allocator>

      %buffer_3 = vm.rodata.inline "_utf8_tensor_FC1814BC4A58F22A" {alignment = 1 : i64} : !vm.buffer = "tensor"

      %c16_4 = vm.const.i32 16

      %c3075_5 = vm.const.i32 3075

      vm.call @hal.buffer.assert(%ref, %buffer_3, %ref_2, %c16, %c16_4, %c3075_5) : (!vm.ref<!hal.buffer>, !vm.buffer, !vm.ref<!hal.allocator>, i64, i32, i32) -> ()

      %buffer_6 = vm.rodata.inline "_utf8_input1_B898B726583C85DA" {alignment = 1 : i64} : !vm.buffer = "input1"

      vm.call.variadic @hal.buffer_view.assert(%arg1, %buffer_6, %c553648160, %c1, [%c4]) : (!vm.ref<!hal.buffer_view>, !vm.buffer, i32, i32, i64 ...)

      %ref_7 = vm.call @hal.buffer_view.buffer(%arg1) {nosideeffects} : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!hal.buffer>

      %buffer_8 = vm.rodata.inline "_utf8_tensor_FC1814BC4A58F22A" {alignment = 1 : i64} : !vm.buffer = "tensor"

      %c16_9 = vm.const.i32 16

      %c3075_10 = vm.const.i32 3075

      vm.call @hal.buffer.assert(%ref_7, %buffer_8, %ref_2, %c16, %c16_9, %c3075_10) : (!vm.ref<!hal.buffer>, !vm.buffer, !vm.ref<!hal.allocator>, i64, i32, i32) -> ()

      %zero_11 = vm.const.i64.zero

      %ref_12 = vm.call @hal.fence.create(%__device_0, %zero_11) : (!vm.ref<!hal.device>, i64) -> !vm.ref<!hal.fence>

      %zero_13 = vm.const.i64.zero

      %ref_14 = vm.call @hal.device.queue.alloca(%__device_0, %c-1, %null, %ref_12, %zero_0, %c48, %c3075, %c16, %zero_13) : (!vm.ref<!hal.device>, i64, !vm.ref<!hal.fence>, !vm.ref<!hal.fence>, i64, i32, i32, i64, i64) -> !vm.ref<!hal.buffer>

      %zero_15 = vm.const.i64.zero

      %ref_16 = vm.call @hal.fence.create(%__device_0, %zero_15) : (!vm.ref<!hal.device>, i64) -> !vm.ref<!hal.fence>

      %zero_17 = vm.const.i64 0

      vm.call.variadic @hal.device.queue.execute.indirect(%__device_0, %c-1, %ref_12, %ref_16, %__simple_add_memoize_result_0_device_0, %zero_17, [(%ref, %zero, %c16), (%ref_7, %zero, %c16), (%ref_14, %zero, %c16)]) : (!vm.ref<!hal.device>, i64, !vm.ref<!hal.fence>, !vm.ref<!hal.fence>, !vm.ref<!hal.command_buffer>, i64, tuple<!vm.ref<!hal.buffer>, i64, i64> ...)

      %zero_18 = vm.const.i64.zero

      %0 = vm.call.variadic @hal.fence.await(%c-1_1, %zero_18, [%ref_16]) : (i32, i64, !vm.ref<!hal.fence> ...) -> i32

      vm.cond_fail %0, "failed to wait on timepoint"

      %ref_19 = vm.call.variadic @hal.buffer_view.create(%ref_14, %zero, %c16, %c553648160, %c1, [%c4]) {nosideeffects} : (!vm.ref<!hal.buffer>, i64, i64, i32, i32, i64 ...) -> !vm.ref<!hal.buffer_view>

      vm.return %ref_19 : !vm.ref<!hal.buffer_view>

    }

    vm.export @simple_add attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @simple_add(%input0: tensor<4xf32>, %input1: tensor<4xf32>) -> (%output0: tensor<4xf32>)"}}

  }

}
```

# 최종 동작방식
- vm.import : vm이 사용할 hal 명령어들을 import 한다. 대부분의 하드웨어를 위한 제어는 해당 명령어들로 구성할 수 있으므로 일반적으로 크게 추가할 부분은 없음
- @hal.command_buffer : 실제 하드웨어 제어에 사용될 op code 및 operand가 기록될 버퍼
- vm.rodata : command buffer에 들어갈 operand를 기록한 곳

- 위 내용들을 조합하여 vm bytecode가 생성됨
- runtime은 vm bytecode를 통해 hal을 분석하고 디바이스 드라이버를 호출한다.
- 호출한 디바이스 드라이버에 @hal.command_buffer에 기록한 값들을 입력시켜 동작시킴

- vm code 및 rodata 세팅
![Pasted image 20260113115644](Pasted%20image%2020260113115644.png)
- vm function, function을 어떤 순서에 따라서 그리고 어떤 값을 넣을 지 계획획
![Pasted image 20260113115726](Pasted%20image%2020260113115726.png)
- 결과를 export
![Pasted image 20260113115919](Pasted%20image%2020260113115919.png)