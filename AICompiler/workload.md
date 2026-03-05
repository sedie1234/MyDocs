
# main workload

1. ~~IREE 및 관련 개념 파악~~
2. ~~예제 실행~~
3. ~~코드 분석~~
4. outline 설계 (진행 중)
5. 작업내용 목록화
6. 작업 파이프라인
7. 초기 시범운영 (mlir 변환 확인)
8. 중간 시범운영 (간단한 모델 구동)
9. 최종 시범운영 (하드웨어와 붙여보기)
10. 고도화 (최적화 부분 / 실험적인 부분 추가하며 성능 고도화)

# pop-up work list

- IREE의 메인영역 (Flow-Stream-HAL이 있는 영역)에서 Custom hardware 전용 최적화 Pass를 포함시키는 방법과 기존 Pass와 충돌났을 때 해결하는 방법 조사
- IREE의 메인영역에서 최종 결과물은 IREE에서 정의하는 형태의 Dialect들로 구성되어 있을 것. iree to custom dialect를 위해 convert pass를 작성해야 하는 IREE의 operation들 목록화하기
- 최종적인 code generation 구조 파악
- plugin 활용 방법 조사하기 : https://github.com/nod-ai/iree-amd-aie?tab=readme-ov-file
- https://iree.dev/guides/deployment-configurations/#listing-available-backends
- https://iree.dev/developers/design-docs/metal-hal-driver/
- https://github.com/nod-ai/mlir-air/tree/2f2345fdb6a40b7a970c42127b70b69a00c03072

- speculative execution을 사용할것인가? -> VLIW에서 basic block 이 커지는 효과가 있어서 효과가 좋다고는 하지만 하드웨어가 자원이 비어있을 때 이야기. -> 코어개수와 모든 예상시간을 측정해본 다음, 적용하는 것이 효과적일 때 적용하는 것이 좋아보임. 
- 

- HW 전용 layout 지정
- -> attribute로 사용
- -> 기존 ins type들을 custom layout으로 가져오는 operation필요
- 

# outline
- iree-air-aie 구조를 참고
- 2개의 git을 구성
- 1개의 git은 하드웨어를 잘 설명하는 IR이 구축된 git으로, IREE 뿐만 아니라 MLIR을 활용하는 다양한 곳에 붙일 수 있도록 할 예정. 사실상 하드웨어의 구성품과 동일하게 취급
- 1개의 git은 IREE로의 plugin 장치
- repH(representation HW), pubH(publish HW)
- repH에는 general하게 사용할 수 있는 dialect와 pass를 구성
- pubH에는 iree specific한 dialect와 pass 및 iree-plugin을 위한 파일들로 구성

# 떠오르는 것들
- 하드웨어에 맞는 IR은 고정하여 들고 다녀야 함. 다른 환경에도 쉽게 붙일 수 있을 것.
- 하드웨어를 표현하는 IR은 라이브러리나 프레임워크에 맞출 것이 아니라 하드웨어에 맞춰야 함




# 임시로 기록
- pubH 받기
```
$ git clone --recursive <pubH git address>
or
$ git clone <pubH git address>
$ git submodule update --init --recursive
```

- 데이터 레이아웃 필요한 정보
	  - 데이터 타입
	  - 데이터 배열 (tile size나 nchw/nhwc.. 등등 각종 배열)
	  - cache
	  - memory bank


- IREE plugin registration
```
IREE_Plugin_Registration/
├── 1. 컴파일러 등록 (Compiler Side Registration)
│   ├── A. Static/Dynamic Registration (IREE_DEFINE_COMPILER_OPTION)
│   ├── B. TargetBackendDescriptor 정의 (고유 ID 및 이름 지정)
│   └── C. BackendFactory 구현 (실제 Backend 객체 생성 루틴)
│
├── 2. 타겟 백엔드 인터페이스 구현 (TargetBackend Interface)
│   ├── A. getDependentDialects (HW1 Dialect 주입)
│   ├── B. buildTranslationPassPipeline (Codegen 파이프라인 연결)
│   └── C. serializeExecutable (최종 바이너리 포맷 결정)
│
├── 3. HAL(Hardware Abstraction Layer) 연동 (Runtime Side)
│   ├── A. Device Capability 선언 (가상 하드웨어 스펙 정의)
│   ├── B. Executable Loader 등록 (컴파일된 코드를 어떻게 읽을 것인가)
│   └── C. Driver/Device Factory (런타임에서 백엔드 식별)
│
└── 4. 빌드 시스템 연결 (Build System Integration)
    ├── A. CMake 프로젝트 구성 (IREE 외부 플러그인 설정)
    ├── B. Dialect 및 Pass 라이브러리 의존성 설정
    └── C. IREE_REGISTER_COMPILER_PLUGIN 매크로 설정
```


# plugin이 작동하는 방식 및 작업흐름
- iree는 hal.executable을 만들어주고 그 아래에 region을 만들어준다. plugin은 그 region의 handle을 가져올 수 있는 것. buildTranslatePassPipeline이라는 함수를 상속받아 사용한다.
- plugin이 적용되는 위치 (그림 상)
<img src="https://imgs.hwan123.synology.me:19999/AICompiler/Apply_Plugin.png" width="500">

- plulgin이 적용되는 위치 (코드 상) : hal.executable.variant가 처음 등장한 직후 builtin.module에 적용
```
#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#map = affine_map<(d0) -> (d0)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#device_target_local = #hal.device.target<"local", [#executable_target_embedded_elf_x86_64]> : !hal.device
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_local
  hal.executable private @simple_add_dispatch_0 {
    hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64) {
      hal.executable.export public @simple_add_dispatch_0_elementwise_4_f32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @simple_add_dispatch_0_elementwise_4_f32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>
          %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>
          %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
          %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
          %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
          %5 = tensor.empty() : tensor<4xf32>
          %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %4 : tensor<4xf32>, tensor<4xf32>) outs(%5 : tensor<4xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %7 = arith.addf %in, %in_0 : f32
            linalg.yield %7 : f32
          } -> tensor<4xf32>
          iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
          return
        }
      }
    }
  }
```
- 작업 flow : 변환해야 할 operation들을 list up 하고 이것을 변환하는 작업을 수행해야 함. 변환해야하는 operation들을 list up 하는 방법
    1) model을 컴파일 (front end만 적용, to hlo / to torch mlir ....)
	2) model을 cpu로 컴파일하고 fullpipeline을 기록 (TODO : 추후에는 필요한 부분만 남길 수 있도록 함)
	3) hal.executable.variant를 검색하고, builtin.module 안의 operation들을 listup에 추가한다.
```
$ ./iree-compile [your_model_front_end].mlir \
--iree-hal-target-backends=llvm-cpu \
--mlir-print-ir-after-all \
--mlir-disable-threading \
--mlir-elide-elementsattrs-if-larger=32 \
-o [your_compiled_model].vmfb \
> [full_pipeline].mlir 2>&1
```



```
cmake -B ./build -S third_party/iree -DIREE_CMAKE_PLUGIN_PATH=$PWD -DIREE_BUILD_PYTHON_BINDINGS=OFF -DIREE_INPUT_STABLEHLO=ON -DIREE_INPUT_TORCH=ON -DIREE_INPUT_TOSA=OFF -DIREE_HAL_DRIVER_DEFAULTS=ON -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_LLVM_CPU=ON -DIREE_BUILD_TESTS=ON -DIREE_EXTERNAL_HAL_DRIVERS=keti-hw1 -DTARGET_DEVICE="HW1"


cmake -B ./build -S third_party/iree -DIREE_CMAKE_PLUGIN_PATH=$PWD -DIREE_BUILD_PYTHON_BINDINGS=OFF -DIREE_INPUT_STABLEHLO=ON -DIREE_INPUT_TORCH=ON -DIREE_INPUT_TOSA=OFF -DIREE_HAL_DRIVER_DEFAULTS=ON -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_LLVM_CPU=ON -DIREE_BUILD_TESTS=ON -DIREE_BUILD_SAMPLES=OFF

cmake --build ./build/ -j$(nproc)


cmake -G Ninja -B ./build -S third_party/iree \ -DIREE_CMAKE_PLUGIN_PATH=$PWD \ -DIREE_BUILD_PYTHON_BINDINGS=OFF \ -DIREE_INPUT_STABLEHLO=ON \ -DIREE_INPUT_TORCH=ON \ -DIREE_INPUT_TOSA=OFF \ -DIREE_HAL_DRIVER_DEFAULTS=ON \ -DIREE_TARGET_BACKEND_DEFAULTS=OFF \ -DIREE_TARGET_BACKEND_LLVM_CPU=ON \ -DIREE_BUILD_TESTS=ON \ -DIREE_BUILD_SAMPLES=OFF



cmake -G Ninja -B ./build -S third_party/iree   -DIREE_CMAKE_PLUGIN_PATHS=$PWD   -DIREE_BUILD_PYTHON_BINDINGS=OFF   -DIREE_INPUT_STABLEHLO=ON   -DIREE_INPUT_TORCH=ON   -DIREE_INPUT_TOSA=OFF   -DIREE_HAL_DRIVER_DEFAULTS=ON   -DIREE_TARGET_BACKEND_DEFAULTS=OFF   -DIREE_TARGET_BACKEND_LLVM_CPU=ON   -DIREE_BUILD_TESTS=ON   -DIREE_BUILD_SAMPLES=OFF  -DTARGET_DEVICE="HW1"



cmake -G Ninja -B ./build_inf -S third_party/iree   -DIREE_CMAKE_PLUGIN_PATHS=$PWD   -DIREE_BUILD_PYTHON_BINDINGS=OFF   -DIREE_INPUT_STABLEHLO=ON   -DIREE_INPUT_TORCH=ON   -DIREE_INPUT_TOSA=OFF   -DIREE_HAL_DRIVER_DEFAULTS=ON   -DIREE_TARGET_BACKEND_DEFAULTS=OFF   -DIREE_TARGET_BACKEND_LLVM_CPU=ON   -DIREE_BUILD_TESTS=ON   -DIREE_BUILD_SAMPLES=OFF  -DTARGET_DEVICE="Infetron_v2"
```



```
$ ./iree-compile ~/workspace/testStage/work1/yolov10_top3.onnx.mlir --iree-hal-target-backends=keti_hw1 --mlir-print-ir-after-all --mlir-disable-threading --compile-to=executable-targets --iree-global-opt-experimental-disable-conv-generalization --iree-opt-generalize-matmul=false --mlir-elide-elementsattrs-if-larger=32 -o ~/workspace/testStage/work1/out.vmfb > ~/workspace/testStage/work1/test_pipeline2.mlir 2>&1
```





# variant 하나로 합치기
variant가 여러 개로 나뉘어져 있어, 원하는 최적화를 수행하지 못하는 문제
-> variant를 하나로 합치는 로직을 iree에 포함시켜 빌드한다.
코드는 publish HW에  "RegionOpUtils_one_variant.cpp" 파일로 남겨둠. 이 파일을 iree에 복사한다.
```
[pubHW]

$ cp RegionOpUtils_one_variant.cpp third_party/iree/compiler//src/iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.cpp

```
-> 추후에는 cmake에서 자동으로 복사해가도록 해야 할 듯.



symbolic execution
