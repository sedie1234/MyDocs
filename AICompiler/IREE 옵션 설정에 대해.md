
# 1. 개요

- IREE의 최적화가 잘되어 있어서 오히려 문제가 발생. 
- 예시로 convolution의 경우, 최적화를 위해 linalg의 루프로 만들어버림. 
- 이것을 해결하고자 앞쪽의 plugin에서 일찍이 custom IR로 변환하는 경우 IREE의 최적화를 적용받지 못할 수 있다. (필요없는 연산 제거가 대표적)
- IREE의 옵션을 보면 --iree-global-opt-experimental-disable-conv-generalization과 같은 옵션이 있다. 이 옵션을 설정하면 이런 패스를 없애거나 순서를 뒤로 미룰 수 있다. (아래 그림은 옵션 여부에 따른 전/후)
![[Pasted image 20260128144858.png]]
![[Pasted image 20260128144926.png]]
- 옵션 설정을 하면 conv가 남아있고, 하드웨어용 IR과 1:1 매칭을 할 수 있다. 
- 이런 경우를 위해 iree의 compile 옵션을 정리한다. 옵션은 버전마다 많이 바뀌는 것으로 보여, 3.9.0 버전을 기준으로 작성함

# 2. 옵션

Obsidian에서 `<...>` 형식을 HTML 태그로 인식하여 내용이 증발하거나 렌더링이 깨지는 문제를 해결했습니다. 모든 꺽쇠 괄호를 백슬래시(`\`)로 이스케이프 처리하거나, 코드 블록(backtick)으로 감싸서 안전하게 수정했습니다.

---

## (IREE 메인 파이프라인 제어) --compile- / -o

- **--compile-from=`<phase>`**
    
    - 컴파일을 시작할 지점을 지정합니다. (start, input, abi, preprocessing, global-optimization, dispatch-creation, flow, stream, executable-sources, executable-configurations, executable-targets, hal, vm)
        
- **--compile-to=`<phase>`**
    
    - 컴파일을 중단하고 결과를 출력할 최종 지점을 지정합니다. (위와 동일한 단계 선택 가능)
        
- **--compile-mode=`<value>`**
    
    - IREE 컴파일 모드를 설정합니다. (std, vm, hal-executable, precompile)
        
- **-o `<filename>`**
    
    - 최종 출력 파일 이름을 지정합니다.
        
- **--output-format=`<value>`**
    
    - 출력 파일의 형식을 지정합니다. (vm-bytecode, vm-c, vm-asm)
        

## (전역 최적화 및 공통 변환) --iree-opt-

- **--iree-opt-level=`<level>`**
    
    - 전체 컴파일 파이프라인에 적용할 최적화 수준을 설정합니다. (O0, O1, O2, O3)
        
- **--iree-opt-const-eval**
    
    - 컴파일 타임에 상수 식을 미리 계산(Eager evaluation)하여 런타임 부하를 줄입니다.
        
- **--iree-opt-const-expr-hoisting**
    
    - 지연된 상수 식 결과를 초기화 단계(immutable global initializers)로 끌어올립니다.
        
- **--iree-opt-const-expr-max-size-increase-threshold=`<long>`**
    
    - 상수 호이스팅 시 허용되는 최대 바이너리 크기 증가 폭을 설정합니다.
        
- **--iree-opt-data-tiling**
    
    - 캐시 효율 향상을 위한 데이터 타일링 최적화를 활성화합니다.
        
- **--iree-opt-aggressively-propagate-transposes**
    
    - 퓨전 기회를 높이기 위해 전치(Transpose) 연산을 적극적으로 전파합니다.
        
- **--iree-opt-generalize-matmul**
    
    - Matmul 연산을 linalg.generic 루프로 변환하여 더 나은 퓨전 기회를 제공합니다.
        
- **--iree-opt-numeric-precision-reduction**
    
    - 가능한 경우 데이터 타입을 낮은 비트(low bit depth)로 축소하여 최적화합니다.
        
- **--iree-opt-outer-dim-concat**
    
    - 모든 Concatenation 연산이 가급적 가장 바깥쪽 차원에서 발생하도록 조정합니다.
        
- **--iree-opt-strip-assertions**
    
    - 유용한 정보를 추출한 뒤 최종 결과물에서 디버그 어설션을 제거합니다.
        
- **--iree-opt-import-parameters / --iree-opt-export-parameters**
    
    - 외부 파라미터 아카이브(.irpa)를 임포트하거나 특정 크기 이상의 상수를 외부로 내보냅니다.
        
- **--iree-opt-splat-parameters=`<string>`**
    
    - 모든 파라미터 백업 글로벌을 스플랫(Splat) 값 아카이브로 생성합니다.
        

## (디스패치 지역 생성 및 연산 퓨전) --iree-dispatch-creation-

- **--iree-dispatch-creation-opt-level=`<level>`**
    
    - 디스패치 형성 단계 전용 최적화 수준을 설정합니다.
        
- **--iree-dispatch-creation-enable-aggressive-fusion**
    
    - 표준보다 훨씬 공격적인 연산 퓨전 전략을 사용합니다. (O2 이상 기본)
        
- **--iree-dispatch-creation-enable-aggressive-reshape-movement**
    
    - Reshape 연산을 자유롭게 이동(bubbling)시켜 퓨전 차단을 방지합니다.
        
- **--iree-dispatch-creation-enable-detensoring**
    
    - 텐서 연산을 스칼라 연산으로 변환하여 불필요한 메모리 접근을 제거합니다.
        
- **--iree-dispatch-creation-enable-early-trunc-fusion**
    
    - 비트 축소(Truncate) 연산을 소비 연산과 조기에 퓨전합니다.
        
- **--iree-dispatch-creation-enable-fuse-horizontal-contractions**
    
    - 공통 피연산자를 가진 여러 수축 연산을 수평적으로 병합합니다.
        
- **--iree-dispatch-creation-enable-fuse-padding-into-linalg-consumer-ops**
    
    - tensor.pad 연산을 후속 Linalg 소비 연산 내부로 통합합니다.
        
- **--iree-dispatch-creation-enable-fuse-padding-into-linalg-producer-ops**
    
    - tensor.pad 연산을 이전 Linalg 생산 연산 내부로 통합합니다.
        
- **--iree-dispatch-creation-enable-split-reduction**
    
    - Reduction 연산을 분할하여 병렬성을 높입니다.
        
- **--iree-dispatch-creation-fuse-multi-use**
    
    - 결과가 여러 곳에서 쓰이는 연산도 퓨전 대상에 포함합니다.
        
- **--iree-dispatch-creation-set-encoding-strategy=`<value>`**
    
    - 연산 인코딩 전략을 설정합니다. (generic, padding)
        
- **--iree-dispatch-creation-data-tiling**
    
    - 디스패치 생성 단계에서 데이터 타일링 최적화를 수행합니다.
        

## (HAL 백엔드 및 실행 타겟 제어) --iree-hal-

- **--iree-hal-target-backends=`<string>`**
    
    - 컴파일할 타겟 백엔드 리스트를 지정합니다. (예: keti_hw1, llvm-cpu, cuda)
        
- **--iree-hal-target-device=`<string>`**
    
    - 타겟 디바이스의 상세 명세(Specification)를 지정합니다.
        
- **--iree-hal-list-target-backends**
    
    - 현재 등록된 모든 타겟 백엔드 목록을 출력합니다.
        
- **--iree-hal-executable-object-search-path=`<string>`**
    
    - 외부 오브젝트(.o, .bc 등) 참조를 위한 추가 검색 경로입니다.
        
- **--iree-hal-dump-executable-sources/binaries/files-to=`<path>`**
    
    - 중간 단계의 IR 소스나 최종 바이너리를 특정 디렉토리에 덤프합니다.
        
- **--iree-hal-link-executables**
    
    - 실행 파일들의 최종 링크 여부를 제어합니다. (비활성 시 바이너리 개별 검사 가능)
        
- **--iree-hal-memoization**
    
    - 커맨드 버퍼 등 디바이스 리소스의 재사용(Memoization)을 활성화합니다.
        
- **--iree-hal-indirect-command-buffers**
    
    - 커맨드 기록 시 버퍼 바인딩을 간접 참조로 변환할지 여부입니다.
        
- **--iree-hal-instrument-dispatches=`<size>`**
    
    - 성능 분석을 위해 디스패치에 인스트루먼테이션을 추가합니다.
        
- **--iree-hal-preprocess-executables-with=`<string>`**
    
    - 각 hal.executable을 외부 명령어 또는 MLIR 패스 파이프라인으로 전처리합니다.
        

## (실행 모델 및 런타임 스케줄링) --iree-execution- / --iree-scheduling-

- **--iree-execution-model=`<value>`**
    
    - 텐서 연산 스케줄링 모델을 선택합니다. (host-only, async-internal, async-external, inline-static, inline-dynamic)
        
- **--iree-scheduling-optimize-bindings**
    
    - 버퍼 바인딩 퓨전 및 디스패치 사이트 특수화(Specialization)를 활성화합니다.
        
- **--iree-scheduling-initialization-mode=`<value>`**
    
    - 파라미터 및 전역 변수 초기화 방식입니다. (sync, async)
        
- **--iree-scheduling-dump-statistics-file/format**
    
    - 스케줄링 관련 통계 데이터를 덤프합니다.
        

## (실험적 전역 최적화 옵션) --iree-global-opt-

- **--iree-global-opt-experimental-disable-conv-generalization**
    
    - Convolution이 linalg.generic 루프로 풀리는 것을 방지합니다. (실험적)
        
- **--iree-global-opt-propagate-transposes**
    
    - 퓨전 향상을 위해 전치 연산을 적극적으로 전파합니다.
        
- **--iree-global-opt-enable-early-materialization**
    
    - 인코딩 정보를 조기에 구체화(Materialization)합니다.
        
- **--iree-global-opt-enable-quantized-matmul-reassociation**
    
    - 양자화된 Matmul 연산의 재결합 최적화를 활성화합니다.
        
- **--iree-global-opt-enable-warn-on-uninitialized-values**
    
    - 초기화되지 않은 값 사용 시 경고를 출력합니다.
        

## (입력 변환 및 타입 조정) --iree-input-

- **--iree-input-type=`<string>`**
    
    - 입력 프로그램의 다이얼렉트 형식을 지정합니다. (none, auto, tosa, stablehlo 등)
        
- **--iree-input-demote-f32-to-f16**
    
    - 모든 f32 연산과 값을 f16으로 강제 하향 변환합니다.
        
- **--iree-input-demote-f64-to-f32**
    
    - 모든 f64 연산과 값을 f32로 강제 하향 변환합니다.
        
- **--iree-input-demote-i64-to-i32**
    
    - 모든 i64 연산과 값을 i32로 강제 하향 변환합니다.
        
- **--iree-input-promote-f16-to-f32 / --iree-input-promote-bf16-to-f32**
    
    - 낮은 정밀도의 부동소수점을 f32로 상향 변환합니다.
        

## (LLVM CPU 코드 생성 최적화) --iree-llvmcpu-

- **--iree-llvmcpu-loop-vectorization / interleaving / unrolling**
    
    - 루프의 벡터화, 인터리빙, 언롤링 최적화를 제어합니다.
        
- **--iree-llvmcpu-slp-vectorization**
    
    - SLP(상향식) 벡터화를 활성화합니다.
        
- **--iree-llvmcpu-enable-ukernels=`<string>`**
    
    - 사전에 최적화된 마이크로커널(ukernels) 사용을 설정합니다. (default, none, all, mmt4d 등)
        
- **--iree-llvmcpu-target-triple / cpu / features**
    
    - 타겟 CPU 아키처 및 상세 기능을 설정합니다.
        
- **--iree-llvmcpu-use-fast-min-max-ops**
    
    - 고속 하드웨어 min/max 명령어를 사용합니다.
        
- **--iree-llvmcpu-reassociate-fp-reductions**
    
    - 부동소수점 Reduction 연산의 결합법칙을 허용하여 최적화합니다.
        
- **--iree-llvmcpu-stack-allocation-limit=`<size>`**
    
    - 스택 할당의 최대 크기를 제한합니다.
        

## (스트림 관리 및 리소스 최적화) --iree-stream-

- **--iree-stream-resource-max-allocation-size / max-range**
    
    - 단일 메모리 할당 및 리소스 바인딩의 최대 범위를 제한합니다.
        
- **--iree-stream-resource-index-bits**
    
    - 리소스 오프셋 참조 시 사용할 인덱스 비트 폭을 설정합니다.
        
- **--iree-stream-partitioning-favor=`<value>`**
    
    - 스트림 분할 우선순위를 설정합니다. (debug, min-peak-memory, max-concurrency)
        
- **--iree-stream-resource-memory-model=`<value>`**
    
    - 호스트-디바이스 메모리 모델을 설정합니다. (unified, discrete)
        
- **--iree-stream-resource-min-offset-alignment**
    
    - 리소스 오프셋의 최소 정렬(Alignment) 조건을 설정합니다.
        

## (VM 및 바이트코드 최적화) --iree-vm-

- **--iree-vm-bytecode-module-optimize**
    
    - VM 바이트코드 생성 전 CSE/Inlining 등 최종 최적화를 수행합니다.
        
- **--iree-vm-target-index-bits**
    
    - VM 내부 인덱스 타입의 비트 폭을 설정합니다.
        
- **--iree-vm-target-extension-f32 / f64**
    
    - f32/f64 타겟 명령어 확장 지원 여부를 설정합니다.
        
- **--iree-vm-bytecode-module-strip-debug-ops / source-map**
    
    - 바이너리에서 디버그 정보나 소스 맵을 제거하여 크기를 줄입니다.
        
- **--iree-vm-emit-polyglot-zip**
    
    - 결과 파일을 Zip 파일 형태로 볼 수 있게 출력합니다.
        

## (MLIR 표준 및 디버깅 옵션) --mlir-

- **--mlir-disable-threading**
    
    - MLIR 내부 멀티스레딩을 끕니다. (출력 순서 고정 및 디버깅 용이)
        
- **--mlir-print-ir-before / after=`<pass-arg>`**
    
    - 특정 패스 실행 전후의 IR을 덤프합니다.
        
- **--mlir-print-ir-after-all**
    
    - 모든 패스 실행 후의 IR을 덤프합니다.
        
- **--mlir-print-ir-module-scope**
    
    - IR 출력 시 항상 최상위 모듈 스코프를 유지합니다.
        
- **--mlir-elide-elementsattrs-if-larger=`<uint>`**
    
    - 큰 텐서 속성값을 "..."으로 생략하여 가독성을 높입니다.
        
- **--mlir-pass-statistics**
    
    - 각 패스의 통계 데이터를 표시합니다.
        
- **--mlir-timing**
    
    - 각 패스의 실행 시간을 측정하여 표시합니다.
        

## (플러그인 및 기타 옵션) --iree-plugin / --color

- **--iree-plugin=`<string>`**
    
    - 로드할 컴파일러 플러그인 파일을 지정합니다.
        
- **--iree-list-plugins**
    
    - 현재 로드된 모든 플러그인 목록을 출력합니다.
        
- **--iree-hw1-debug-pass**
    
    - HW1 백엔드 전용 패스의 디버그 출력을 활성화합니다. (사용자 지정 옵션)
        
- **--color**
    
    - 터미널 출력에 색상을 사용합니다. (default=autodetect)