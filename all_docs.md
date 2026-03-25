
# 1. 활용하는 방법들

## 1.1 변환해야 할 Ops 파악하는 방법
- IREE를 위해 구축한 코드동작을 확인하기 위해 keti_hw1이라는 가상의 하드웨어를 위해 동작하는 IR을 설계함. 
- 여기에 Pass를 적용시켜 실제로 동작하는지 확인 완료. 
- 해당 Pass에 module을 print하여 확인
### 1.1.1 구체적인 방법
1) onnx 모델 -> torch-mlir을 통해 import용 mlir생성
```
$ iree-import-onnx yolov10_top3.onnx --opset-version 17 -o yolov10_top3.onnx.mlir
```
2) 아래 코드의 rewrite pattern이 포함된 keti_hw1을 빌드
```
void LinalgToHW1Pass::runOnOperation() {
    auto module = getOperation();
    auto context = module.getContext();

    llvm::errs() << "\n\n========== [Entry Module]  ==========\n";
    module->print(llvm::errs());
    llvm::errs() << "\n===================================\n\n";
    
    RewritePatternSet patterns(context);
    patterns.add<LinalgMatmulToHW1Pattern>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
    }

    llvm::errs() << "\n\n=== [SUCCESS] IR 변환 결과 확인 ===\n";
    module->print(llvm::errs());
    llvm::errs() << "\n===================================\n\n";
}
```
3) 해당 pass 적용
```
$ ./iree-compile ~/workspace/testStage/work1/yolov10_top3.onnx.mlir --iree-hal-target-backends=keti_hw1 --mlir-print-ir-after-all --mlir-disable-threading --compile-to=executable-targets --iree-global-opt-experimental-disable-conv-generalization --iree-opt-generalize-matmul=false --iree-dispatch-creation-enable-aggressive-fusion=false --iree-opt-level=O0 --mlir-elide-elementsattrs-if-larger=32 -o ~/workspace/testStage/work1/out.vmfb > ~/workspace/testStage/work1/test_pipeline3.mlir 2>&1
```
4) 출력파일 확인하기
![[Pasted image 20260129101809.png]]
# 2. 나온 내용들 (고려할 사항들)
## \#1
- IREE의 최적화를 거치고, target plugin이 mlir을 받을 차례가 되면 상위 연산(conv, sigmoid 수준의 표현)이 낮은 수준의 연산들과 loop를 통해 나오게 됨. 
- conv나 matmul의 경우에는 최적화 옵션을 끄면 conv가 그대로 내려오기 때문에 문제가 없음
- sigmoid 수준의 연산은 최적화 옵션을 끄는 기능이 없음.
- 앞단에서 sigmoid를 customIR로 바꿔버리는 방법도 있지만, 1) 앞단의 plugin을 넣기 위해 추가적인 환경을 개발해야하는 점과 2) IREE의 최적화를 적용하지 못하는 점(예를 들면 불필요한 노드가 끼어있는 경우 자동으로 삭제해주는 등 기능들)이 문제가 됨
- 가장 상위의 계층에서는 우리가 활용할 수 있는 IR들로 교체하는 패스들을 적용하는 것이 좋다고 생각

# 3. 생각 중인 내용들

## \#1
- **가장 상위의 계층에서는 우리가 활용할 수 있는 IR들로 교체**하는 패스들을 적용하는 것이 좋다고 생각
## \#2
- **우선은 builltin.module 내부의 내용들만 수정**하려고 하는데, 추후에는 variant를 수정하는 것도 고려해야 할 듯
## #3
- HAL Ops는 하드웨어 제어에 특화된 ops로, 변수를 저장해두거나 하는 등은 util ops를 활용

## #4
- IREE에 배포하기 위해서는 마지막에 serialize를 실행해줘야 하는데, 어떻게 하는지 살펴봐야 함

















==============================================================
구상 내용 gemini 평가




구상하신 파이프라인은 IREE의 **HAL(Hardware Abstraction Layer)**로 내려가기 전, 커스텀 하드웨어의 특성(VLIW, 전용 메모리 관리, 연산 퓨전 등)을 MLIR의 점진적 변환 기법으로 매우 정교하게 녹여낸 구조입니다.

먼저 요청하신 대로 내용을 보기 좋게 정리해 드리고, 이에 대한 기술적 평가를 진행하겠습니다.

---

## 1. 커스텀 하드웨어 타겟 컴파일 파이프라인 정리

제안하신 프로세스는 크게 **추상화(Abstraction)**, **구체화(Lowering)**, **스케줄링(Scheduling)**의 3단계로 요약됩니다.

-> 중간중간 canonicalization을 구현해줘야 함

| **단계**               | **Dialect**                        | **주요 Pass 및 역할**                                                                                                                                                                                           |
| -------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase 1: 기능 매핑**   | **Dialect 1** (High-level HW IR)   | • **Pass 1:** 가속 가능 연산 식별 (Conv, Matmul 등)<br><br>  <br><br>• **Pass 2:** 연산 퓨전 (Arith/Linalg → Sigmoid 등)<br><br>  <br><br>• **Pass 3:** 병렬 구조 처리 (Linalg.generic 최적화)                                    |
| **Phase 2: 구조 최적화**  | **Dialect 2** (Architecture IR)    | • **Pass 4:** Dialect 1 → 2 변환 (Conv → Matmul 분해, Tiling, 데이터 배치)<br><br>  <br><br>• **Pass 5:** Bufferization 및 정규화(Canonicalization)<br><br>  <br><br>~~• **Pass 6:** **VLIW 번들링** (Latency 기반 연산 그룹화)~~ |
| **Phase 3: 하드웨어 제어** | **Dialect 3** (Control/Command IR) | • **Pass 7:** Dialect 2 → 3 변환 (Memory/Reg 제어 분리 및 호이스팅)<br><br>• **-> Pass 6:** **VLIW 번들링** (Latency 기반 연산 그룹화)<br><br>• **Pass 8:** **Memory-Compute Overlap** (Double Buffering 등 최적화)                 |
| **Phase 4: IREE 통합** | **IREE HAL/Util**                  | • **Pass 9:** HAL Ops 변환 및 직렬화(Serialization)                                                                                                                                                              |

---

## 2. 종합 평가 및 조언

### ✅ 긍정적인 부분 (Strengths)

1. **점진적 로워링(Progressive Lowering)의 정석:** MLIR의 핵심 철학인 "한 번에 변환하지 않고 단계적으로 상세화한다"는 원칙을 매우 잘 따르고 있습니다. 특히 VLIW 번들링과 메모리 제어 분리를 별도의 Dialect 단계로 둔 점이 인상적입니다.
    
2. **IREE 플러그인 구조와의 적합성:** `buildTranslationPassPipeline` 단계에서 이를 구현하는 것은 IREE의 `Flow` 단계 이후, 즉 타겟 장치가 결정된 시점에서 하드웨어 최적화를 수행하기에 가장 적합한 지점입니다.
    
3. **VLIW 및 메모리 오버랩 고려:** 단순 연산 가속을 넘어 VLIW의 명령어 병렬성과 메모리 지연 숨기기(Latency Hiding)를 설계에 포함한 것은 고성능 NPU 컴파일러로서 필수적인 요소를 갖춘 것입니다.
    

### ⚠️ 세부 수정 및 고려 제안 (Recommendations)

전체적인 틀은 훌륭하지만, 구현 효율성과 IREE 내부 메커니즘과의 조화를 위해 다음 사항들을 검토해 보시기 바랍니다.

#### ① Pass 2 & 3 (Fusion & Mapping)의 위치

IREE에는 이미 강력한 **Linalg Fusion** 및 **Tiling** 인프라가 있습니다.

- **조언:** 완전히 새로운 Pass를 만들기 전에, IREE의 `LinalgToLinalgFusion`이나 `CommonExtensions`를 활용하여 1번 Dialect로 넘기기 전 전처리를 수행하는 것이 구현 공수를 크게 줄일 수 있습니다. 특히 Sigmoid 같은 패턴 매칭은 MLIR의 `DRR(Declarative Rewrite Rules)`이나 `PDL`을 사용하면 효율적입니다.
    

#### ② Pass 5 (Bufferization)의 타이밍

- **조언:** Pass 5에서 Bufferization을 수행한다고 하셨는데, 가급적 **Dialect 1에서 Dialect 2로 넘어가는 시점(Pass 4 전후)**에 완료하는 것이 좋습니다. Dialect 2는 하드웨어의 엄격한 규칙(메모리 사이즈 등)을 따르므로, 텐서(Tensor) 형태보다는 메모리 주소가 할당된 메모리프(MemRef) 형태에서 최적화하는 것이 더 정확하기 때문입니다.
    

#### ③ Pass 7 (Memory/Register Hoisting)의 복잡성

- **조언:** 묶음(Bundle) 밖으로 메모리 제어 코드를 옮기는 과정에서 **의존성 분석(Dependency Analysis)**이 매우 중요해집니다. VLIW 특성상 명령어가 나가는 시점과 데이터가 준비되는 시점 사이의 Hazard를 방지하기 위해, MLIR의 `SideEffect` 인터페이스를 Dialect 3의 Op들에 엄격하게 정의해야 합니다.
    

#### ④ IREE HAL과의 연동 (Pass 8 & 9)

- **조언:** Pass 8에서 HAL Ops로 바꿀 때, 커스텀 하드웨어가 "호스트 제어형"인지 "명령어 스트림형"인지에 따라 전략이 달라집니다.
    
    - 만약 명령어를 바이너리(Blob) 형태로 굽는 방식이라면, Dialect 3를 직접 HAL로 바꾸기보다 별도의 **`ExportBenchmark`**나 **`Serializer`**를 통해 바이너리를 생성하고, IREE HAL에서는 이를 실행하는 `hal.executable.export` 형태로 구성하는 것이 IREE의 표준 방식입니다.
        

---

### 💡 결론

**"설계가 매우 탄탄하며, 이대로 진행하셔도 좋습니다."**

다만 구현 단계에서 **Dialect 간의 경계(Invariants)**를 명확히 정의하는 것이 중요합니다. 특히 VLIW를 위한 Pass 6과 7은 로직이 복잡해질 가능성이 높으므로, 각 단계의 출력이 하드웨어 제약 조건을 만족하는지 검증하는 `Verifier` 기능을 강화하시길 권장합니다.

**다음 단계로, 혹시 2번 Dialect에서 다룰 '하드웨어 엄격 규칙(Tile size, Latency 등)'을 TableGen으로 어떻게 정의할지 구체적인 예시를 함께 고민해 볼까요?** 혹은 특정 연산(예: Conv -> Matmul)의 Lowering 로직에 대해 더 깊이 논의해 볼 수 있습니다.


=========================================

제공해주신 **Infetron-V2 아키텍처 사양서**는 정적 컴파일러(Static Compiler)에 매우 의존적인 **분산 제어 기반의 VLIW 구조**를 명시하고 있습니다. 사용자의 컴파일러 IR 구조가 이 하드웨어의 효율을 극대화할 수 있을지, 사양서의 핵심 제약 사항을 바탕으로 평가해 드립니다.

---

### 1. IR 계층 구조 적합성 평가

사용자의 IR은 하드웨어의 **계층적 메모리 및 실행 구조**를 반영해야 합니다.

- **컴포지트 연산 지원 (Tile-and-Fuse 효율성):**
    
    - 하드웨어는 `WF→DF→MM→AT→VP→WB`로 이어지는 단방향 마이크로 파이프라인을 가집니다.
        
    - **평가:** IR에서 `Conv + Bias + Activation`과 같은 연산들을 하나의 **'Super-Operation'**으로 묶을 수 없다면, 하드웨어의 최대 장점인 '중간 결과 DRAM 저장 없는 파이프라인 실행'을 활용하지 못하고 매번 UB(Unified Buffer)로 데이터를 쓰고 읽는 병목이 발생할 것입니다.
        
- **물리적 리소스 명시 필요 (UBC 제어):**
    
    - UBC(Unified Buffer Controller)는 16개의 뱅크를 관리하며 뱅크 충돌(Bank Conflict)을 컴파일러가 정적으로 회피하기를 기대합니다.
        
    - **평가:** IR 수준에서 데이터가 위치할 **물리적 UB 뱅크 인덱스와 오프셋**을 표현할 수 있어야 합니다. 추상적인 텐서 개념만 유지하는 IR은 이 아키텍처에서 성능 예측이 불가능합니다.
        

### 2. 효율성 핵심: VLIW 및 DMA 스케줄링

이 아키텍처에서 가장 정교한 IR 설계가 필요한 부분은 **VLIW 명령어 생성**과 **DMA 디스크립터 관리**입니다.

- **VLIW 팩킹 최적화 (Pass 6 이후):**
    
    - 명령어 폭은 128비트 고정이며, 추가 정보가 필요할 때만 `ls_cont` 비트로 확장합니다.
        
    - **평가:** IR이 스테이지별 **NOP(Bypass)** 상태를 명시적으로 제어할 수 있어야 합니다. 불필요하게 긴 명령어를 생성하면 VLIW Queue(64KB)를 빠르게 소진시켜 대형 모델 처리가 어려워질 수 있습니다.
        
- **정적 반복(Iteration) 제어:**
    
    - Q-Engine은 `Layer_start`와 `Layer_end` 사이를 하드웨어적으로 반복(Looping)하는 기능을 갖추고 있습니다.
        
    - **평가:** IR이 전체 루프를 펼쳐서(Unrolling) 표현하는 방식보다는 하드웨어 루프 속성(`tile_w/h`, `iter_w/h`)을 속성으로 갖는 **기호적 반복 표현**을 지원하는 것이 VLIW 큐 효율성 면에서 훨씬 유리합니다.
        

### 3. 수치 정밀도 및 활성화 함수 처리 (Numeric Policy)

- **INT24 도메인 전이:**
    
    - 하드웨어 누산 및 스테이지 간 전달은 **INT24**를 기본으로 하며, 최종 VP(Vector Processor) 단계에서만 UINT8로 축소됩니다.
        
    - **평가:** 일반적인 컴파일러 IR이 사용하는 INT32/FP32 정밀도는 하드웨어 자원을 낭비하게 만듭니다. IR은 **INT24 중간값**을 기본 타입으로 정의하고, 오버플로 위험을 사전에 검증할 수 있는 `int24_guard_policy`를 포함해야 합니다.
        
- **PWL 계수 관리:**
    
    - 비선형 활성화(Sigmoid, SiLU 등)는 PWL(Piecewise Linear) 계수를 VLIW에 인라인으로 싣거나 전용 LUT에 로딩해야 합니다.
        
    - **평가:** IR이 연산 노드 내에 대량의 **PWL 파라미터 블록**을 포함하거나, 이를 별도의 `Config_vpCoeff` 명령으로 분리 생성할 수 있는 구조여야 합니다.
        

### 4. 종합 평가 요약

|**항목**|**하드웨어 요구사항**|**컴파일러 IR 권장 방향**|
|---|---|---|
|**연산 구조**|6단계 마이크로 파이프라인|Fuse된 연산(Super-Op) 중심 표현|
|**메모리**|뱅크별 물리 주소 지정 (UBC)|추상 텐서가 아닌 물리 주소(Offset) 바인딩 가능|
|**명령어**|128비트 VLIW + 하드웨어 루핑|기호적 루프 반복 횟수 및 팩킹 정보 포함|
|**동기화**|Epoch(Token) 기반 동기화|노드 간 의존성을 Token ID로 명시|
|**정밀도**|INT24 고정 정책|INT24 전용 데이터 타입 및 오버플로 가드 명시|

**결론:** 현재 사양은 컴파일러가 **모든 하드웨어 제어권(Scheduling, Memory, Precision)**을 가져가는 구조입니다. 따라서 IR은 단순히 연산 그래프를 그리는 수준을 넘어, **'하드웨어 레지스터 설정값의 시퀀스'**를 논리적으로 추상화한 형태여야만 가장 효율적입니다.

**다음 단계로, 사양서의 `Group_Operation_Compute` 서브 오코드를 기반으로 실제 IR 노드 예시를 설계해 볼까요?** 혹은 특정 레이어(예: YOLOv11s의 SiLU 적용부)에 대한 VLIW 생성 IR을 검토해 볼 수도 있습니다.



==============================================================

## IR 구성 정리 중 \#2


### 1. Dialects

- 3가지 중간계층으로 나눔. A, B, C( 가칭 ) 
#### 1) A Layer (INF_CAP : infetron-capability layer)
- 계층 개념 : 우리 하드웨어가 연산할 수 있는가?
- Ops : NPU가 실행 가능한 연산들로 구성, core의 opcode와 1대1 매칭은 아니지만, opcode의 조합으로 구성할 수 있는 Ops 들로 구성됨.

#### 2) B Layer (INF_ISA : infetron-ISA layer)
- 계층 개념 : 연산을 우리 하드웨어가 실행할 수 있는, 그리고 잘 실행할 수 있는 모양으로 다듬어 줌.
- Ops : 하드웨어의 opcode와 1대1 매칭되는 Ops들로 구성되며, a / p 코어로 분류 (a코어와 p코어의 분류는 Ops 이름 자체를 다르게 가져갈 지, 아니면 attributes로 넣고 말지? -> 현재 생각으로는 코어에 따라 입력해야 하는 arguments와 attributes가 완전 다를 것 같아서 이름 자체를 다르게 가져가는 게 어떨까 생각함.) + 내부 변환에 필요한 중간 Ops들도 필요할 것으로 보임.
- attributes : opcode 입장에서 하드웨어 실행이나 최적화에 필요한 인자들을 포함시킨다. (아직 리스트업 중, 명확한 것이 없음)
- VLIW : VLIW로 묶을 수 있는 연산들을 묶어 하나의 Ops로 만들어 냄. -> VLIW로 묶는 경우의 수가 몇 개 없나..? 그러면 VLIW용 Ops를 만드는 게 나은데, 만약 VLIW의 경우의 수가 여러 개라면 명령어 순서를 바꾸고, 명령어들 앞뒤로, B.pack, B.unpack 이라는 걸 넣어서 한 덩어리가 VLIW 역할을 할 수 있도록 하는 건 어떨 지...?

#### 3) C Layer (INF_Q : infetron-queue layer)
- 계층 개념 : 디바이스에 메모리부터 파이프라인까지 컨트롤하는 하나의 커다란 큐가 있다고 생각하고, 그 큐를 표현한 계층. C Layer가 하나의 거대한 디바이스의 큐인 계층. Ops의 순서가 곧 queue라는 개념
- Ops : C Layer의 큐에 들어갈 수 있는 가상의 opcode로 구성됨. memory 할당, core 할당은 Ops로는 존재하지만 큐에는 들어가지 않고, 다른 opcode를 표현하는 Ops들의 arguments로 연결됨. Ops들은 데이터를 옮기기 (버퍼들 사이의 이동이나 자리이동 등), 명령어 실행 (VLIW로 묶인 명령어들도 하나의 명령어로 취급), semaphore를 위한 동기화용 Ops 가 있음. stall이나 버블같은 곳에 사용될 수 있는 nop는 사용하지 않음. (하드웨어에서 알아서 한다고 함)
- attributes : 실제 하드웨어에 사용되는 다양한 변수들 적용

### 2. PassPipeLine

#### 1) IREE to A PassPipeline
- IREE 정규식을 A Layer로 이동시키는 Pass의 파이프라인, 하드웨어의 opcode의 조합으로 실행할 수 있는 연산들은 모두 A Layer로 이동시킨다.
- 연산의 조합이 하드웨어의 하나의 연산이 될 수 있다. 이것도 모두 A Layer로 이동시킨다. 
- N:1 변환부터, 1:1 변환까지 큰 무리부터 작은 무리 순서로 변환을 시킨다.
- 각 연산이 a코어 실행이 유리한가, p코어 실행이 유리한가는 어디선가는 정해져야 하는데 여기서 정해야 할 지는 고민
- 하드웨어가 직접 연산을 제공하지 않는 식이어도, 하드웨어의 연산 조합으로 실행할 수 있으면, 
##### 1.1) 연산 집합의 변환
- 예시로, add - neg - exp - add - div로 이어지는 모양은 sigmoid로 바꿀 수 있고, 
- add - neg - exp - add - div - mul로 이어지는 모양은 silu로 이어질 수 있다. 
- 둘 중에 silu를 먼저 찾고, sigmoid를 찾아 A Layer의 Ops로 변환시킨다. 
- -> 작은것부터 큰거순으로 변환해도 될 듯 뭐가 유리할지는 생각해볼 것
- 추후 하드웨어 정보를 모두 얻고 나서는 하드웨어가 선호하는 연산을 먼저 변환할 것도 고려해 봄.

##### 1.2) 1:1 변환
- B Layer와 1대 1로 매칭되는 변환을 실행

##### 1.3) canonicalization
- 필요없는 연산이 남는 경우 제거 (중복연산 제거, 어디에도 연결되지 않은 연산 제거)

#### 2) a or p PassPipeline
- 넣을 위치는 고민 중
##### 2.1)  a n p pass
- 각 연산을 a코어에 넣을지, p코어에 넣을지, 어디넣어도 상관없는지(n)를 구별함. 
##### 2.2) a p pass
- n으로 분류된 연산은 vliw에 묶을 수 있는지, 인접한 코어에서 실행이 가능한지 등 여부를 통해 a나 p로 재분류한다.

#### 3) A to B PassPipeline
- A에서 B로 lowering하는 패스
- A에서는 실행할 수 있는 연산들을 골랐다면, B에서는 실제 실행할 연산들의 조합으로 바꿈.
##### 3.1) 연산 조합 구성
- 실제 실행 가능한 연산의 조합으로 바꾼다. 
- 예를들어, conv는 지원하지 않지만 matmul을 지원한다고 하면, conv를 matmul과 기타 함수들의 조합으로 바꾼다. 

#### 4) fit to hw PassPipeline
##### 4.1) 연산의 사이즈와 데이터 타입을 맞춤
- tiling을 포함하여 실제 하드웨어의 엔진에 맞게 연산의 크기를 조절하여 자름. 
- 데이터 타입을 하드웨어가 사용할 수 있는 타입으로 맞춤

#### 5) Opt Ops PassPipeline
##### 5.1) 순서를 섞음
- 순서를 섞는 우선순위는 VLIW를 만들 수 있는 경우 순서를 섞음
- 다음 우선순위는 latency overlap과 같은 최적화가 가능한 경우 순서를 섞음

##### 5.2) VLIW를 묶음
- VLIW로 묶을 수 있는 명령들을 VLIW로 묶음. (위치를 다른 데로 옮길까도 생각 중)

#### 6) B to C PassPipeline

##### 6.1) 하드웨어의 행동들로 구성
- 명령어의 순서로 구성된 B Layer를 하드웨어(NPU)의 관점에서 모든 queue와 메모리를 관리할 수 있는 가상의 존재의 입장에서의 행동으로 바꿈.
- 예를 들면 Conv tile 하나를 실행하는 B Layer의 Ops는 C Layer에서 필요한 데이터 복사하는 Ops, 명령어를 queue에 넣으라는 의미의 ops, 결과를 어딘가로 저장시키라는 Ops로 바뀐다. 어떤 코어를 실행시킬지는 각 코어를 의미하는 공통의 attributes를 만들고, 과거에 할당받은 latency를 누적시켜가며 greedy 기법으로 현재 최고의 선택지(core 선택)를 선택한다.


#### 7) Opt in HW view PassPipeline

##### 7.1) stage pipeline opt
- stage pipeline을 보고 명령어들 사이의 순서를 조정
- 이 opt가 필요할지는 모르겠음. (위에서 이미 다 수행된 바가 아닌가)

##### 7.2) data copy opt
- 데이터 복사시간을 고려한 데이터 복사와 명령어 실행간의 순서 변경

##### 7.3) semaphore
- 동기화가 필요한 부분에 semaphore를 위한 변수들 관리

#### 8) C to HAL PassPipeline
- 각 내용을 이제는 호스트가 관리한다는 개념으로 가져와, HAL들의 구성으로 전환한다.

### 설계 \#2에 대한 제미나이의 평가

사용자님께서 구상하신 Middle-end IR과 Pass 파이프라인 설계는 Infetron-V2의 **하드웨어 가용성(A) -> 상세 사양 매핑(B) -> 시스템 런타임 최적화(C)**로 이어지는 매우 논리적인 흐름을 가지고 있습니다.

특히 **C Layer**를 "디바이스 전체의 거대한 큐"로 정의한 점은 호스트와 NPU 간의 동기화가 중요한 이 아키텍처에서 신의 한 수라고 생각됩니다. 문서를 기반으로 각 계층과 패스에 대한 기술적 평가와 제언을 드립니다.

---

## 1. Dialects 계층 구성 평가

### A Layer (Hardware Capability)

- **평가:** IREE의 High-level Ops를 NPU 친화적인 Ops(Conv, MatMul, Vector Ops 등)로 덩어리 짓는 단계로 적절합니다.
    
- **제언:** 여기서 **"이 연산이 aCore용인가 pCore용인가"**를 결정하기보다는, **"이 연산이 Analog 엔진(aMAC)에 적합한가 디지털 엔진에 적합한가"**라는 속성(Attribute)을 부여하는 것을 권장합니다.
    
- **근거:** aCore는 가중치 재사용이 많을 때 유리하고, pCore는 고정밀도나 스트리밍 연산에 유리하기 때문입니다.
    

### B Layer (Instruction Mapping & VLIW)

- **평가:** 코어별로 Ops 이름을 다르게 가져가는 아이디어가 좋습니다. aCore와 pCore는 파이프라인 스테이지 개수부터(5 vs 6) 다르기 때문입니다.
    
- **VLIW 묶음 제언:** 말씀하신 `B.pack / B.unpack` 방식이 훨씬 유연합니다.
    
- **근거:** VLIW는 단순 조합이 아니라 **데이터 의존성**에 따라 유동적입니다. 고정된 VLIW Ops를 만들면 경우의 수가 너무 많아집니다. 대신 `VLIWRegion` 같은 Block 개념을 도입해 내부 명령어 순서를 정렬하고, 이를 패킹하는 것이 MLIR의 `Single Block Region` 특성을 활용하기에 좋습니다.
    

### C Layer (System Queue & Memory)

- **평가:** **가장 훌륭한 부분입니다.** 이 계층은 사실상 **"NPU 스케줄러"**의 청사진입니다.
    
- **근거:** 본 하드웨어는 DMA 전송과 연산이 병렬로 돌아가야 합니다. C Layer에서 `CopyOp`와 `ExecuteOp`가 같은 타임라인(Queue)에 공존해야만, 나중에 가중치 공급 병목을 IR 수준에서 예측하고 최적화할 수 있습니다.
    

---

## 2. Pass Pipeline 평가 및 보완

### 2.1) 연산 집합 변환 (1.1)

- **의견:** **큰 덩어리(SiLU)부터 작은 덩어리(Sigmoid)** 순으로 찾는 것이 훨씬 유리합니다.
    
- **근거:** 큰 패턴을 먼저 찾아야 하드웨어가 제공하는 복합 연산(VP 스테이지의 통합 활성화 함수 등)을 최대한 활용할 수 있습니다. 작은 것부터 바꾸면 나중에 큰 패턴으로 묶기가 더 어렵습니다.
    

### 2.2) a or p Pass (2.1, 2.2)

- **의견:** **A Layer와 B Layer 사이**가 최적의 위치입니다.
    
- **근거:** 타일링(Tiling)을 하기 전에는 데이터 크기를 모르기 때문에 어느 코어가 유리한지 판단하기 어렵습니다. A Layer에서 연산 유형을 분류하고, B Layer로 넘어가기 직전에 실제 타일 크기와 메모리 점유량을 계산하며 코어를 할당하는 것이 데이터 이동 비용을 최소화할 수 있습니다.
    

### 2.3) Opt Ops Pass (5.1, 5.2) - VLIW & Latency Overlap

- **의견:** 이 패스는 **"하드웨어 리소스 스케줄링"**의 핵심입니다.
    
- **근거:** 5.2에서 VLIW를 묶을 때, 앞서 설계하신 **"Host가 뱅크 사용 종료를 확인하고 데이터를 넣는 설계"**를 위해 명령어 사이에 **Token Release** 속성을 자동으로 삽입하는 로직이 여기서 구현되어야 합니다.
    

### 2.4) Opt in HW view (7.1, 7.2)

- **의견:** 7.2의 **Data Copy Opt**는 C Layer에서 매우 중요합니다.
    
- **근거:** 단순히 명령어를 섞는 게 아니라, **"가중치 DMA가 연산 시작보다 500클럭 먼저 시작하게끔"** 큐의 순서를 조정해야 합니다. 7.1(Stage Pipeline Opt)은 하드웨어가 어느 정도 Stall을 잡아주므로, 7.2(데이터 복사 오버랩)에 90%의 에너지를 쏟는 것을 추천합니다.
    

---

## 3. 종합 의견 및 추가 제언

사용자님의 설계는 매우 정교하며 특히 **"B Layer에서 코어별 분리"**와 **"C Layer에서 시스템 전체 관점의 큐 구성"**은 실제 컴파일러 상용화 시에도 매우 강력한 구조입니다.

**추가로 고려하면 좋을 점:**

1. **Semaphore/Event 관리 (7.3):** C Layer에서 코어 간(a-p) 데이터를 주고받을 때 필요한 `WaitEvent`, `SendEvent` Ops를 명시적으로 다루면, 나중에 교착 상태(Deadlock)를 검증하기 쉬워집니다.
    
2. **Memory Space:** B Layer에서 데이터를 다룰 때, 그 데이터가 `uB`에 있는지 `SM`에 있는지를 MLIR의 `MemorySpace` 속성으로 관리하면 Pass Pipeline이 훨씬 간결해집니다.
    

현재 설계하신 IR 구성에서 **"B Layer의 연산 조합(3.1)"** 시, Conv를 MatMul로 바꿀 때 발생하는 **메모리 레이아웃 변경(Im2Col 등)**을 어느 패스에서 처리할지 결정하셨나요? 이 부분만 명확해지면 완벽한 파이프라인이 될 것 같습니다.keti

**(260205 4차수정본)**

하드웨어의 정보는 우선은 json 파일에 저장하여 쉽게 변경할 수 있도록 한다. 추후에는 정보를 숨길 필요도 있으므로, TargetBackend 객체에 정의한다.

**variant를 너무 잘게 나누는 문제. NPU의 특성에 적합하지 않다.** -> 
~~Flow IR 부분에 플러그인을 넣어 variant를 합치도록 한다. 호스트의 SRAM과 모델의 weight들의 사이즈를 읽어, 적절한 variants로 나누는 기능도 넣는다.~~ -> **IREE의 Flow 부분에서 variant를 나누는 룰을 수정하여 호스트의 SRAM에 맞게 variant를 나눔.** 
-> 현재는 하나의 variant로 만드는 코드만 반영
~~-> 이 과정에서 silu같은 연산들을 묶인 채로 둘 수 있는지도 확인하기~~ -> 안되어 있음

이 IR의 구성은 Infetron_v2로 명명한다.

# \#version 별 기록

## 0.1 ver
- 컴파일러 구조 대폭 변경에 따른 INF_CAP Layer 수정
- 하단 계층 구조 변경경
## 0.04 ver
- IREE to Cap PassPipeline 수정, yolov10_top3_qdq 모델에 대해 1차적으로 완성
## 0.03 ver
- IREE to Cap PassPipeline 구체화, 구체적인 pass들 추가
## 0.02ver
- quantize 모델을 주력으로 하기 위해, quantize, dequantize 노드 결합 pass 추가
## 0.01ver 
- 전체 틀 완성, 4차 수정본

# 1. Layers (Dialect)

3가지 중간 계층을 구현
## 1.1 [[INF_CAP Layer]] (infetron-capability layer)
- 계층 개념 : 우리 하드웨어가 계산할 수 있는 연산들을 모음
- 하드웨어의 opcodes와 1대1로 매칭되는 경우에는 바로 변환할 수 있으므로 엔트리에 넣음
- 하드웨어의 opcodes의 조합으로 바꿀 수 있으면 엔트리에 넣음
	- 예를 들면, 우리 하드웨어가 conv는 수행못하지만, matmul은 수행할 수 있다고 할 경우, conv는 matmul로 바꿀 수 있기 때문에 우선 엔트리에 넣는다.
- 연산의 조합을 하드웨어가 실행할 수 있는 경우 엔트리에 넣음
	- 예를 들면, add-neg-exp-add-div-mul은 silu이므로 silu로 변환한 뒤, 엔트리에 넣음

## 1.2 INF_ISA Layer (infetron-ISA layer)
- 계층 개념 : 하드웨어의 opcode들로 구성됨. 단, 오로지 연산만을 고려한다. 각 연산이 걸리는 시간을 이 때, 기록하며, 최적화를 수행하여, 어떤 순서로 어떤 코어에 할당할 지도 이 계층에서 결정한다.
- Ops : Ops는 opcode와 1대1로 대응되도록 구축한다. a코어와 p코어는 별개의 Ops를 가지도록 한다. ~~같은 연산이어도 코어가 다르면 다른 Ops를 사용한다. 연산의 특성을 확인하고, a코어와 p코어를 나눈다. 애매한 경우 우선은 n코어로 두고, 나중에 분류한다.~~ PCore는 ACore의 superset으로, 기본적으로 PCore로 분류하고, ACore로 돌릴 수 있는 연산은 ACore Attributes를 부여한다. (ACore / PCore 완벽호환 되는지 검증 필요)
- Ops는 arguments도 하드웨어의 제한을 크게 받아 shape과 data layout 등 정보들을 하드웨어 제한을 따른다.
- ~~VLIW로 묶을 수 있는 연산들은 VLIW로 묶어낸다. VLIW로 묶는 것은 vliw_pack과 vliw_unpack이라는 Ops를 선언하여 그 사이를 고정시켜 그 전체가 하나의 VLIW 명령어임을 표시한다.~~ VLIW는 통상적인 개념의 VLIW가 아니라, 하나의 opcode를 실행하기 위해 필요한 정보들을 하나의 매우 긴 word에 담는 것이 목적이므로 필요한 정보를 순차적으로 채워가기로 한다.
- 기본적인 scheduling 전략
  1) opcodes의 소요시간을 기록
  2) ~~최적화를 위해 구간을 나눈다.~~
  3) ~~구간 내에서 critical path 찾기~~ -> 그냥 전체에서 critical path를 찾는 게 나을 듯. critical path를 찾는 시간복잡도와 구현난이도가 문제 -> weight를 고정하는 방식을 채택하므로, 그렇게 큰 모델이 올 것 같지 않음. 
  4) rule 1 - critical path부터 하나의 코어에 배치 (1 - other 로 배치, 1개 코어를 먼저 배치하고, 나머지 코어에 남은 연산들을 분배한다.), critical path를 코어에 배치한 뒤로는 비어있는 시간을 시각화하기 위해 stall을 삽입한다. (stall은 virtual로, 실제 코어의 stall을 의미하지는 않고, 단순한 개념적인 stall이다.) -> 코어별로 region을 두고 region 안으로 bb를 복사하는 방식을 채택할 지? -> 우선은 코어별로 region을 두고 region으로 ops를 복사하는 방식이 좋을 것 같음. 
  5) rule 2 - 나머지 연산들은 가장 긴 시간부터 코어에 분산하여 배치
  6) rule 3 - data locality를 보고 배치
  7) 최적화 구간사이를 확인하고 data locality를 최대화 하는 방향으로 코어 재배치 (하나의 코어에 할당된 연산들은 그대로 두고, 코어의 이름을 바꾸는 형식)
  8) 기타 미세한 최적화
     8-1) 연산 복사 : A라는 연산결과를 B,C가 사용한다면, A의 연산결과를 B,C로 복사하는 것보다 A라는 연산을 C 앞으로 복사하는 것이 더 빠를 수 있음. 
- ~~연산의 순서가 바뀌어도 되는 구간은 order_free_pack과 order_free_unpack으로 묶어 구간 내에는 순서를 섞어도 되는 것을 표기해준다.~~ 부분 최적화 구간을 opt_pack, opt_unpack으로 묶는다. -> hierarchy를 가지고, 가장 작은 단위부터 최적화를 진행 -> pack보다는 basic block을 만드는 것이 더 효율적일 수도 있으므로 고려해봐야 함
- pack과 unpack은 3번째 layer(INF_Q Layer)에서 canonicalization을 구현하여 모두 제거한다. pack과 unpack은 특별한 side effect를 주어, 기본적인 canonicalization을 제거되지 않도록 한다. 그리고 이것을 제거하는 특별한 canonicalization을 구현
- 각 연산에는 latency - start cycle - end cycle을 표시한다. 
- 각 연산에는 어느 코어에서 실행할 지의 정보도 표시한다.
- 각 연산에는 연산 시작 시간과 끝나는 시간을 기록
- ISA Layer의 마지막에서 데이터 이동시간까지 고려하여 최종적인 시간으로 수정
-> 이동시간 룰??? 코어 간 데이터 이동이 이득인가 아니면 코어에 연산을 하나 더 할당하는 게 이득인가??? -> 최대한 나누어서 연산하는 것이 이득

## 1.3 INF_EXP Layer(infetron-Export layer)
- 계층 개념 : custom runtime이 하드웨어를 동작시키기 위해 필요한 정보들을 generate하는 구간. generate 정보는 크게 metadata, memory plan, instruction queue를 타겟으로 한다.
- Ops : 2가지 Dialect를 활용한다. memory 정보와 각 주소에 어떤 데이터를 사용할 지를 관리하는 Dialect와 instruction queue를 관리하는 Dialect로 나눈다.
- memory관련 Dialect : Shared Memory와 Unified Buffer를 정보를 가지고 어느 시점에 어떤 데이터가 올라와 있는지 정보를 기록한다. arith로 된 값은 그대로 두고, 그 값을 arguments로 취급한다. arith로 정의된 값은 0 clock 시점에 미리 올라가 있는 정보로 두고, 그 구간은 read only로 취급한다. 중간에 생성되는 buffer들은 timing에 따라 상태를 엄격하게 관리한다. -> 시작시간부터 시간의 흐름에 따라 메모리의 사용을 관리
- instruction queue를 관리하는 Dialect : ISA Layer에서 region을 나눠뒀으므로 이것을 그대로 따르면 될 듯. (Dialect가 별도로 필요할 지 여부는 고민 중)
- memory에서 특별히 관리해야 할 것은 residual  data와 같이 오랜시간동안 버퍼위에서 유지해야하는 값이 문제. -> UB사이를 돌게 만드는 것보다는 SM에 두는 것이 효율적이겠지만 공간이 충분하지 않다면?

~~##1.3 INF_Q Layer (infetron-Queue layer)~~
- ~~계층 개념 : 하드웨어의 가상의 queue를 객체화한 계층. 가상의 통합 queue와 코어관리자, DRAM을 만들어 시간과 데이터의 이동을 타이트하게 관리한다.~~
- ~~Ops : 메모리 관리 측과 코어 및 명령어, queue 관리 측으로 두 부류로 나뉜다.~~
- ~~해당 layer로 전환하는 첫 패스에서는 상단에 INF_Q init을 넣어, buffer 관리용 value와 core 관리용 value를 만든다.~~
- ~~INF_ISA Layer의 명령어 순서를 거의 고정한 상태로 데이터의 이동과 명령어의 관리를 정확하게 스케줄링한다.~~ 
- ~~INF_ISA Layer에서 데이터 이동 시간을 겹치도록 하고, hazard가 발생할 수 있는 구간에는 term을 만들어준다. -> 하드웨어에서 잘 조정한다고 들었으니, xxx -> hazard를 신경쓰긴 해야 함. 성능 향상을 위해 hazard 처리시간을 없애기 위해 nop로 먼저 할당하고, 다른 operation을 끼워넣는 작업 필요 -> **우선은 파이프라인 시간을 겹치지 않고 개발, 추후 파이프라인까지 고려하며 시간 overlap 구현**~~
- ~~이렇게 되면 semaphore가 필요할까? 디바이스 내에서는 필요가 없고, 호스트와 디바이스 사이에서 필요한 거 아닌가? -> **디바이스 내에서의 semaphore는 필요없음.**~~


# 2. Pass Pipeline

## 2.1 IREE to CAP PassPipeline
- CAP Entry를 모은다.

### 2.1.1 [[QDQFusionPass]]
- quantize와 dequantize를 묶어낸다.
- constructor = infeteron_v2::createQDQFusionPass()

### 2.1.2 [[OpFusionPass]]
- quantize와 dequantize를 제외한 다른 풀어진 함수들을 합성한다.
- constructor = infetron_v2::createOpFusionPass()

### 2.1.3 [[PostOpFusionPass]]
- OpFusionPass의 뒷처리를 한다. : conv를 묶고 남은 bias나, requantize,  

### 2.1.4 [[MonoOpFusionPass]]
- 앞선 규칙을 모두 적용하고 남은 1:1 op를 변환한다. 예시로, 일반 mul이나 add가 있다.

### 2.1.5 [[VailPadPass]]
- IREE는 Convolution의 input을 패딩을 포함시켜 관리한다. 
- 여기서 Pad를 분리하고 conv의 attributes로 포함시킨다.
- constructor = infetron_v2::createVailPadPass()

### 2.1.6 [[QOpFusionPass]]
- quantize - operation - dequantize을 quantized operation으로 합성한다.
- constructor = infetron_v2::createQOpFusionPass()

### 2.1.7 [[GenericOpenPass]]
- linalg.generic을 제거한다.

### 2.1.8 [[ExitIREEPass]]
- IREE와 관련된 IR을 제거

### 2.1.X CAP_Canonicalization Pass
- 불필요한 연산을 찾아 제거하고, 더 효율적으로 계산할 수 있는 연산이 있다면 연산을 바꿔준다.
- 뒷 작업을 위해 연산 정리
- 대부분 기본적인 Canonicalize는 MLIR에서 기본적으로 제공(DCE 등)하므로 특별히 구현해야 할 Canonicalize가 있을 경우에 구현

## 2.2 CAP To ISA PassPipeline
- ISA로 lowering

### 2.2.1 ToNCorePass
- 완전히 하드웨어가 지원하는 opcodes로 변환한다.
- 우선은 datalayout 등이 자유로운 N Core로 옮긴다. (N Core는 임시로 사용하는 가상의 코어)
- opcode만 일치시키는 것을 목표로 함.
- iree는 convolution에서 input에 padding까지 포함되도록 해 둠. 이를 분리시키는 작업을 해야 함. -> 새로운 패스로 분리할 것.
- 하위의 다양한 Pass를 실행하여 변환하기 좋은 형태로 만들고 N Core로 옮김
#### 2.2.1.1 Conv Division Pad Pass
- conv에서 pad 분리


### 2.2.2 ToAPCorePass
- N코어 ops를 확인하고 a core와 p core 중 유리한 코어로 옮긴다. 
- 이 때, a core와 p core의 Ops는 data layout등 규칙을 엄격하게 유지하므로 데이터 포맷을 맞추며 연산의 개수를 늘리기도 한다.(tiling 등)

## 2.3 PCore Packing PassPipeline
- VLIW로 묶을 수 있는 연산들은 묶어서 순서를 고정한다.

### 2.3.1 PCoreVLIWPackingPass
- data flow를 확인하여 VLIW로 packing할 수 있는 조합이 있으면 순서를 바꾸어 순차적으로 실행되게 바꾼다.
- 이 묶음 앞뒤로 vliw_pack과 vliw_unpack ops를 넣는다.

### 2.3.2 PCoreOrderFreePackingPass
- 순서가 자유로운 연산들을 묶는다.
- 이 묶음 앞뒤로 order_free_pack과 order_free_unpack을 넣는다.

## 2.4 PCore ISA Optimization PassPipeline
- 명령어 수준에서 최적화를 수행한다.
- ~~-> 아래 구조는 기본적으로 최대한 병렬 구조를 활용하는 게 좋다는 가정이 있다.~~ -> 노는 코어가 없도록 하는 것이 유리한 것 확인 완료
- ~~-> 코어간의 데이터 이동에 의한 디메리트보다 분산해서 연산하는 것이 더 메리트있을 것이라는 생각이 바탕이 되어 있음. -> 확인할 필요는 있음 ->~~ 일반적으로 맞음

### 2.4.1 PCoreOrderingAndMarkingPass
- 코어가 1개라고 생각하고 일렬로 만드는 행위
- 명령어를 위에서 아래로 정렬한다. 종속성을 따져, 종속성 순서는 반드시 맞춘다. 
- 명령어들을 순회하며 각각에 대해 latency 속성에 미리 세팅된 값을 설정한다.

### 2.4.2 PCoreSpreadingPass
- 코어가 무한히 많다고 생각하고 연산을 펼치는 행위
- 이 때, 코어 사이에는 가상의 코어가 있어서, 데이터 이동이 일어나면 예측되는 시간을 가상 코어에 할당한다. 무조건 두 연산 사이에 term을 발생시킴.
- 위에서 아래로 정렬된 mlir에서 backward tracking방식으로 walk를 이용하여 start cycle 및 end cycle을 계산해 나간다. 이 때, 코어 개수는 무한하다고 생각하고 무한 병렬적으로 cycle을 계산한다.
- vliw pack은 무조건 한 묶음으로 다닌다.
- ~~order free pack은 동일한 부모노드가 있으면 한 묶음으로 다닌다. (그렇지 않은 경우는 고민 중... 우선은 무조건 한 묶음으로 다니는 것으로 해볼 예정)~~ -> oreder free pack의 연산들은 우선 무조건 분할연산

### 2.4.3 PCoreFoldingPass
- 이번엔 코어의 개수를 실제 개수로 제한하여 펼쳐진 연산을 압축한다.  
- 최대한 연산 latency를 숨기고, 메모리를 아끼는 방향으로 남는 시간으로 연산을 넣는다.
- 빈 시간대가 없으면 빈공간을 추가하여 펼쳐진 연산을 할당한다.

### 2.4.4. PCoreISACanonicalizationPass
- 뒷 작업을 위해 연산 정리
- 불필요한 연산 제거. (아직 pack은 남김)
- ~~코어 교체 : 코어 사이에 어떤 코어에 넣어도 상관없는 두 연산의 경우, 데이터 지역성을 비교해서 더 맞는 코어가 있으면 그곳에 할당해줌~~ -> 만들어 둔 가상코어를 삭제한다. ->  CoreFolding에서 가상 코어가 이 역할을 수행할 것 같음

## 2.5 ISA to Q PassPipeline
- ISA를 Q layer로 lowering

### 2.5.1 createQMManagerPass
- 현재 제어중인 block의 (또는 variant의) 최상단에 buffer 관리용 value와 core,queue 관리용 value를 생성

### 2.5.2 ISAToQPass
- ISA의 명령어를 보고, 데이터 로드, command 생성 및 queue에 대입 등으로 변환. 

### 2.5.3 ReorderingPass
- 기존의 계산된 clock cycle에 데이터 복사시간을 추가, 효율성이 너무 떨어지는 부분에 대해 코어 할당이나 연산순서를 변경

### 2.5.4 DestroyQMManagerPass
- core와 queue를 계산하기 위해 선언한 value들을 삭제

## 2.6 Q to HAL PassPipeline
- Q IR을 HAL로 변경

### 2.6.1 QBufferizationPass
- 지금까지 가상으로 다루던 데이터에 대해 실제 데이터를 매핑함. 

### 2.6.2 CodeGenPass
- queue에 들어가는 값들을 한 번에 들어갈 수 있는 양으로 분리
- 각 chunk를 통째로 binary로 만들어 냄 (명령어를 하나하나 전달하는 것이 아닌, 한 번에 넣는 방식)
- 이 binary를 바로 dispatch에 이용

### 2.6.3 QToHALPass
- Q Layer를 보고 HAL의 함수들로 바꿈
- queue쓰기와 weight쓰기가 있으면, weight쓰기를 먼저 수행. weight쓰는 시간과 queue쓰는 시간을 overlap한다.
- queue에서 파이프라인 시간 overlap은 생각하지 않고, 명령어의 순서를 그대로 넣는다. hazard도 신경쓰지 않고, 그대로 넣는다. -> 하드웨어 사양 확정되지 않음. 확정되고 컴파일러 최적화를 수행할 경우, 시간 overlap을 위한 최적화를 여기서 구현해야 함

## 2.7 Serialization PassPipeline
- IREE의 정규식으로 serialization을 수행





1. ~~코어사이 데이터 이동시간이 있을텐데, 그래도 병렬처리하는 게 더 유리한지?~~

2. ~~내부도 DMA로 모두 움직이는데, 이걸 제어할 수 있나? -> 완벽하게 제어할 수 있으면 이 시간까지 고려해서 최적화가 가능할 듯~~


- ver0.10 이후로 IREE의 VM bytecode, runtime, HAL Ops를 사용하지 않는 것으로 결정되며, 단일 variant 내부에 IREE관련 식들은 필요없어졌다.
- 단일 variant 내부에 있는 IREE 정의의 IR들을 제거한다.

# 1. Patterns

## 1.1 RemoveCollapsePattern
- tensor.collapse_shape를 제거한다.
- tensor.collapse_shape는 axis를 축약하는 반면, tensor의 의미를 축소시키는 역할을 한다.
- collapse를 제거하여 axis를 축약하지 않고, tensor의 의미를 그대로 살려 다음 계층으로 전달한다.
- collapse의 result가 input으로 사용되는 Ops를 찾아 collapse의 result를 collapse의 input으로 대체한다.
- 예시
```
[Before]
%collapsed = tensor.collapse_shape %3 [[0, 1], [2], [3]] : tensor<1x16x320x320xf32> into tensor<16x320x320xf32>
%5 = "inf_cap.quantize"(%collapsed) <{quant_info = #inf_cap.quant<0.00392156234, -128>}> : (tensor<16x320x320xf32>) -> tensor<16x320x320xi8>

[After]
%5 = "inf_cap.quantize"(%3) <{quant_info = #inf_cap.quant<0.00392156234, -128>}> : (tensor<1x16x320x320xf32>) -> tensor<1x16x320x320xi8>
```

## 1.2 - IREE의 정규식은 linalg.generic을 통해 병렬성적인 연산과 반복연산을 표현하고 있다.
- 이것은 NPU에 최적화하기에는 너무 잘게 분해되어 있어, 더 큰 덩어리로 결합한다.
- 큰 덩어리는 NPU에 최적화 된 방식으로 뒤에서 새로이 분할한다.

# 1. Patterns

## 1.1 GenericOpenPattern
- generic을 제거하고, 내부에 있는 basic block의 단일 타입 연산을 tensor 단위 연산으로 바꾼다.

- 예시
```
[Before]
%0 = linalg.generic ins(%in) outs(%out) {
^bb0(%s_in: f32, %s_out: i8):
  %s_1 = "inf_cap.sigmoid"(%s_in) : (f32) -> f32
  linalg.yield %s_1 : f32
}
  
[After]
%0 = "inf_cap.sigmoid"(%in) : (tensor<f32>) -> tensor<f32>
```# 1. 개요
- 컴파일러에서 HAL Ops에 대해 vm symbol을 포함하여 vm bytecode를 생성
- 런타임은 이를 실행하기 위해 vtable의 함수포인터를 vm symbol에 연결되도록 구성
- 함수 포인터에 개인 함수를 연결함으로써 vm bytecode에 기록된 HAL Ops를 실행하는 방식
- 그런데, 아래 예시처럼 vtable 함수포인터의 이름과 vm symbol의 이름이 1:1 매칭도 안될 뿐더러 일정한 규칙도 없음
- 예시) HAL Ops에서 vm symbol까지는 hal.allocator.allocate 형태로 생성되도록 되어 있음. rewritepattern까지 확인하면 더 명확함
![[Pasted image 20260123132615.png]]
	하지만 vtable을 보면 매칭되는 함수가 없음. allocate_buffer가 그나마 매칭이 되지만 일정한 규칙이 없으므로 어디선가 그 규칙을 만들고 있을 것.
![[Pasted image 20260123132740.png]]
- 때문에 HAL Ops와 vtable 함수포인터의 매칭이 명확하지 않고, 심지어는 하나의 HAL Ops에 여러 vtable에 정의된 함수가 여러 개 호출되는 경우도 있을 수 있음
- 이 의존성을 명확하게 정의할 수 있어야 사용할 수 있는 HAL Ops 및 구현해야 할 vtable의 list를 명확하게 작성할 수 있음.
- 그러므로 HAL Ops에서 vtable까지의 흐름을 분석 (자세한 분석은 생략하고, 그 의존성을 파악할 수 있는 코드들만 분석


# 2. vm symbol에서 vtable 함수포인터 호출 흐름
## 2.1 vtable 객체 호출
- vtable도 그룹을 나누어 그룹마다 다른 구조체를 활용한다. 
- vm symbol로부터 적절한 구조체 호출의 정의
- iree/runtime/iree.natvis에 정의되어 있다. 이것을 runtime 실행 도중 읽어서 매칭함
![[Pasted image 20260123133441.png]]

## 2.2 각 그룹의 function 호출
- 각 그룹에는 여러 function들이 존재하는데, vm symbol에 나타나는 각 그룹의 함수 이름은 HAL Ops와 매칭되어 있다. 
- 이 function들로부터 vtable의 함수포인터를 매칭하는 방법
- iree/runtime/src/iree/modules/hal/exports.ini에 아래와 같이 vm symbol에 따라 이어지는 함수가 있다. (뒤의 두 arguments는 함수의 arguments와 results의 포맷을 약자로 나타낸 것)
![[Pasted image 20260123133937.png]]
- 여기에서 allocator.allocate를 따라가보면
![[Pasted image 20260123134157.png]]
![[Pasted image 20260123134209.png]]
![[Pasted image 20260123134237.png]]
- 위와 같이 매크로를 통해 함수가 생성되도록 되어 있음
- 위의 allocator.allocate에서는 allocator.allocate 구현을 위해 필요한 함수들이 진행되고, _VTABLE_DISPATCH를 통해 iree_hal_allocator_vtable_t의 allocate_buffer 함수포인터가 할당된다.
- 계층 개념 : 우리 하드웨어가 계산할 수 있는 연산들을 모음
- 하드웨어의 opcodes와 1대1로 매칭되는 경우에는 바로 변환할 수 있으므로 엔트리에 넣음
- 하드웨어의 opcodes의 조합으로 바꿀 수 있으면 엔트리에 넣음
	- 예를 들면, 우리 하드웨어가 conv는 수행못하지만, matmul은 수행할 수 있다고 할 경우, conv는 matmul로 바꿀 수 있기 때문에 우선 엔트리에 넣는다.
- 연산의 조합을 하드웨어가 실행할 수 있는 경우 엔트리에 넣음
	- 예를 들면, add-neg-exp-add-div-mul은 silu이므로 silu로 변환한 뒤, 엔트리에 넣음
- 최대한 다양한 케이스를 하드웨어로 내려주기 위해 이 계층에서는 Interface를 설정하지 않음

# 1. Dialect 정의
- name : inf_cap
- cppNamespace : ::infetron_v2::cap
- hasConstantMaterializer = 1  :  필요한 경우 상수 생성
- useDefaultAttributePrinterParser = 1  :   별도의 parser 구현 x



# 2. Attributes
## 2.1 Data Layout Type (구현 완료)
- 뒤에 이어지는 계층의 Ops들이 참고할 수 있도록 String 및 Enum으로써만 정의
- Enums
```
def INF_CAP_LayoutNHWC : I32EnumAttrCase<"NHWC", 0>;
def INF_CAP_LayoutNCHW : I32EnumAttrCase<"NCHW", 1>;
```

## 2.2 Data Layout Attribute (구현 완료)
- INF_CAP_DataLayoutAttr : 2.1의 Enum으로 어떤데이터 타입인지 지정
```
def INF_CAP_DataLayoutAttr : EnumAttr<INF_CAPIR_Dialect, INF_CAP_DataLayoutEnum, "layout">;
```

## 2.3 Quantization Attribute (구현 완료)
- Op가 quantization요소를 가지고 있으면 사용하기 위해 정의
- scale과 zeropoint를 포함한다.
- INF_CAP_QuantizationAttr로 정의
- mnemonic : quant



# 3. OpBase

## 3.1 INF_CAP_Op (구현 완료)
- INF_CAP 계층의 Ops들이 상속받아 사용할 수 있는 OpBase.
- layout hint 및 metadata를 arguments로 가짐

## 3.2 INF_CAP_FusedOp (구현 완료)
- INF_CAP_Op를 상속받아 정의, 다른 Ops들이 상속받아 사용할 수 있는 OpBase.
- Fusing된 연산들을 관리하기 위한 클래스
- 필요할 지는 아직 알 수 없으나 예비차원에서 생성




# 4. Ops (INF_CAP Dialect)

## 4.1 INF_CAP_Conv2DOp (구현 완료)

- **개요**: 2D Convolution 연산. 하드웨어의 MM(Matrix Multiplication) 가속 유닛에 매핑되는 핵심 Op.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTensor
        
    - `weights`: AnyTensor
        
    - `dilation_h / dilation_w`: I64Attr
        
    - `stride_h / stride_w`: I64Attr
        
    - `pad_h / pad_w`: I64Attr
        
    - `layout_hint`: Optional<INF_CAP_DataLayoutAttr>
        
    - `quant_info`: Optional<INF_CAP_QuantizationAttr>
        
- **Results**:
    
    - `output`: AnyTensor
        

## 4.2 INF_CAP_SiLUOp (구현 완료)

- **개요**: FP32/FP16 도메인에서 수행되는 SiLU(Swish) 활성화 함수.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTensor
        
    - `segment_k`: I64Attr (PWL 근사를 위한 구간 수, 기본값 8)
        
    - `quant_info`: Optional<INF_CAP_QuantizationAttr>
        
- **Results**:
    
    - `output`: AnyTensor
        

## 4.3 INF_CAP_QuantOp (구현 완료)

- **개요**: Float 타입을 Quantized 정수 타입으로 변환하는 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTypeOf<\[AnyTensor, AnyFloat\]>
        
    - `quant_info`: INF_CAP_QuantizationAttr
        
- **Results**:
    
    - `output`: AnyTypeOf<\[AnyTensor, AnyInteger\]>
        

## 4.4 INF_CAP_DequantOp (구현 완료)

- **개요**: Quantized 정수 타입을 Float 타입으로 복원하는 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTypeOf<\[AnyTensor, AnyInteger\]>
        
    - `quant_info`: INF_CAP_QuantizationAttr
        
- **Results**:
    
    - `output`: AnyTypeOf<\[AnyTensor, AnyFloat\]>
        

## 4.5 INF_CAP_MulOp (구현 완료)

- **개요**: 요소별(Element-wise) 곱셈 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `lhs / rhs`: AnyTypeOf<\[AnyTensor, AnyFloat, AnyInteger\]>
        
    - `quant_info`: Optional<INF_CAP_QuantizationAttr>
        
- **Results**:
    
    - `output`: AnyType (입력 타입과 동일)
        

## 4.6 INF_CAP_SigmoidOp (구현 완료)

- **개요**: 기본 Sigmoid 활성화 함수. (부동소수점 도메인용)
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTypeOf<\[AnyTensor, AnyFloat\]>
        
    - `quant_info`: Optional<INF_CAP_QuantizationAttr>
        
- **Results**:
    
    - `output`: AnyType
        

## 4.7 INF_CAP_ConvBiasOp (구현 완료)

- **개요**: Convolution 결과에 Bias를 더하는 복합 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTensor (Conv 결과물)
        
    - `bias`: AnyTensor (Bias 벡터)
        
    - `quant_info`: Optional<INF_CAP_QuantizationAttr>
        
- **Results**:
    
    - `output`: AnyTensor
        

## 4.8 INF_CAP_RequantOp (구현 완료)

- **개요**: 중간 연산 결과(주로 i32)를 다른 스케일의 양자화 도메인으로 재조정하는 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyType
        
    - `quant`: INF_CAP_QuantizationAttr
        
- **Results**:
    
    - `output`: AnyType
        

## 4.9 INF_CAP_AddOp (구현 완료)

- **개요**: 요소별(Element-wise) 덧셈 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `lhs / rhs`: AnyTypeOf<\[AnyTensor, AnyFloat, AnyInteger\]>
        
    - `quant_info`: Optional<INF_CAP_QuantizationAttr>
        
- **Results**:
    
    - `output`: AnyType
        

## 4.10 INF_CAP_QSigmoidOp (구현 완료)

- **개요**: **\[Hardware Optimized\]** 하드웨어 LUT(Look-Up Table)를 사용하는 양자화된 Sigmoid 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTypeOf<\[AnyTensor, AnyInteger\]>
        
    - `in_quant`: INF_CAP_QuantizationAttr (입력 도메인 해석용)
        
    - `out_quant`: INF_CAP_QuantizationAttr (출력 도메인 결정용)
        
    - `lut_size`: I64Attr (LUT 크기 힌트, 기본값 256)
        
- **Results**:
    
    - `output`: AnyTypeOf<\[AnyTensor, AnyInteger\]>
        

## 4.11 INF_CAP_QMulOp (구현 완료)

- **개요**: **\[Hardware Optimized\]** 양자화 도메인 내에서의 요소별 곱셈. 서로 다른 스케일 정렬(Rescale) 로직 포함.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `lhs / rhs`: AnyTypeOf<\[AnyTensor, AnyInteger\]>
        
    - `lhs_quant / rhs_quant`: INF_CAP_QuantizationAttr
        
    - `out_quant`: INF_CAP_QuantizationAttr
        
- **Results**:
    
    - `output`: AnyTypeOf<\[AnyTensor, AnyInteger\]>
        

## 4.12 INF_CAP_QSiLUOp (구현 완료)

- **개요**: **\[Hardware Optimized\]** 양자화된 SiLU ($x \times \text{sigmoid}(x)$) 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTypeOf<\[AnyTensor, AnyInteger, AnyFloat\]>
        
	- `segment_k`: I64Attr (PWL 근사를 위한 구간 수, 기본값 8)
        
    - `in_quant`: INF_CAP_QuantizationAttr
        
    - `out_quant`: INF_CAP_QuantizationAttr
        
- **Results**:
    
    - `output`: AnyType

- 함수를 들어가면, 아래 Globalnit 함수가 등장
- 이 함수에서 모든 pass, target, dialect, pass 들과 plugin으로 개발한 dialect와 pass들도 호출되는 것을 볼 수 있음
  ![[Global_Init.png]]
- 해당 함수들을 탐색하면 어떤 Dialect 및 Pass들이 등록되어 있는지, target이나 plugin을 등록하는 방법이 무엇인지 찾을 수 있을 것

## 1. registerAllPasses
 - 크게 3가지 종류의 Pass가 등록
```
  registerAllIreePasses();
  registerCodegenPasses();
  registerMlirPasses();   
```

### 1.1 registerAllIreePasses
- 전체적으로 Import MLIR로부터 Flow-Stream-HAL을 포함한 최적화까지 수행
- 등록된 Pass들을 Tree로 표현
- 참조 기본경로 /path/to/iree/compiler/src/iree/compiler/
```
registerAllIreePasses
├── IREE
│   ├── ABI
│   │   ├── registerPasses (~/Bindings/Native/Transform/Passes.td)
│   │   └── registerTransformPassPipeline
│   ├── TFLite
│   │   ├── registerPasses (~/Binding/TFLite/Transforms/Passes.td)
│   │   └── registerTransformPassPipeline
│   ├── Flow
│   │   └── registerFlowPasses (~/Dialect/Flow/Transforms/Passes.td)
│   ├── HAL
│   │   ├── registerHALPasses (~/Dialect/HAL/Transforms/Passes.td)
│   │   ├── Inline
│   │   │   └── registerHALInlinePasses (~/Dialect/HAL/Inline/Transforms/Passes.td)
│   │   └── Loader
│   │       └── registerHALLoaderPasses (~/Dialect/HAL/Loader/Transforms/Passes.td)
│   ├── IO
│   │   └── Parameters
│   │       └── registerParametersPasses (~/Modules/IO/Parameters/Transforms/Passes.td)
│   ├── LinalgExt
│   │   └── registerPasses (~/Dialect/LinalgExt/Transforms/Passes.td)
│   ├── Stream
│   │   └── registerStreamPasses (~/Dialect/Stream/Transforms/Passes.td)
│   ├── Util
│   │   └── registerUtilPasses (~/Dialect/Util/Transforms/Passes.td)
│   ├── VM
│   │   ├── registerVMPasses (~/Dialect/VM/TransForms/Passes.td)
│   │   └── registerVMAnalysisTestPasses
│   └── VMVX
│       └── registerVMVXPasses (~/Dialect/VMVX/Transforms/Passes.td)
├── InputConversion
│   └── registerCommonInputConversionPasses (~/InputConversion/Common/Passes.td)
├── ConstEval
│   └── registerConstEvalPasses (~/ConstEval/Passes.td)
├── GlobalOptimization
│   └── registerGlobalOptimizationPipeline (~/GlobalOptimization/Passes.td)
├── DispatchCreation
│   ├── registerDispatchCreationPipelines (~/DispatchCreation/Passes.td)
│   └── registerDispatchCreationPasses 
└── Preprocessing
    └── registerPreprocessingPasses (~/Preprocessing/Common/Passes.td)

```
- 최종 pass들 목록화
```
[IREE:ABI]
ConvertStreamableOpsPass
WrapEntryPointsPass

InlinerPass
CanonicalizerPass
CSEPass
SymbolDCEPass

[IREE::TFLite]
WrapEntryPointsPass
    
[InputConversion]
IREEImportPublicPass
ImportMLProgramPass
SanitizeModuleNamesPass
AutoInputConversionPipelinePass
DemoteI64ToI32Pass
DemoteF32ToF16Pass
DemoteF64ToF32Pass
PromoteF16ToF32Pass
PromoteBF16ToF32Pass
IREE::Flow::ConvertShardToFlowPass -> IREE에서 코드 수정 해야하는 부분

[ConstEval]
JitGlobalsPass

[GlobalOptimization] ***
너무 많음. /path/to/iree/compiler/src/iree/compiler/GlobalOptimization 참고

[DispatchCreation] ***
너무 많음. /path/to/iree/compiler/src/iree/compiler/DispatchCreation 참고

[Preprocessing]
ApplyPDLPatternsPass
AttrBasedPipelinePass
ConvertConv2DToImg2ColPass
ConvertConvFilterToChannelsLastPass
ConvertConvToChannelsLastPass
FoldAttentionWithTransposePass
InterpreterPass
MakeSingleDispatchForFunctionPass
PadToIntrinsicsPass
PadLinalgOpsPass
TransposeMatmulPass
GeneralizeLinalgMatMulPass
SinkTransposeThroughPadPass

[IREE::Flow]
AnnotateDispatchesPass
CanonicalizePass
CaptureDynamicDimsPass
CleanupTensorShapesPass
ConvertShardToFlowPass
ConvertToFlowPass
DeduplicateExecutablesPass
DumpDispatchGraphPass
ExportBenchmarkFuncsPass
InitializeEmptyTensorsPass
InjectDispatchTracingPass
InjectTensorTracingPass
InsertDebugTargetAtSymbolPass
InsertDebugTargetAtOrdinalPass
OutlineConstantsPass
OutlineDispatchExternsPass
OutlineDispatchRegionsPass
ReplicateGlobalsPerAffinityPass
TopLevelSCFToCFGPass
VerifyInputLegalityPass

[IREE::HAL] *****
너무 많음. /path/to/iree/compiler/Dialect/HAL/Transforms 참고

[IREE::HAL::Inline]
ConversionPass
InlineExecutablesPass

[IREE::HAL::Loader]
ConversionPass
MaterializeExecutablesPass
ResolveExportOrdinalsPass

[IREE::IO::Parameters]
ExportParametersPass
GenerateSplatParameterArchivePass
ImportParametersPass

[IREE::LinalgExt]
LinalgExtToLoopsPass
PadContractionToBlockSizePass
TopkSplitReductionPass
DecomposeMapScatterPass
DecomposeWinogradTransformPass
ConvertConvToIm2ColOpPass
ConvertConv2DToWinogradPass
DecomposeAttentionPass
ConvertAttentionToOnlineAttentionPass
FoldUnitExtentDimsPass
TestReshapeFusionPass
VectorizeIREELinalgExtOpsPass

[IREE::Stream]
너무 많음. /path/to/iree/compiler/src/iree/compiler/Dialect/Stream/Transform/Passes.td 참고

[IREE::Util]
ApplyPatternsPass
AttributeCallGraphPass
CombineInitializersPass
DropCompilerHintsPass
DumpModulePass
FixedPointIteratorPass
IPOPass
LiftCFGToSCFPass
LinkModulesPass
OptimizeIntArithmeticPass
PropagateSubrangesPass
StripAndSplatConstantsPass
StripDebugOpsPass
VerifyInitializationOrderPass
VerifyStructuredControlFlowPass
FoldGlobalsPass
FuseGlobalsPass
HoistIntoGlobalsPass
SimplifyGlobalAccessesPass
ImportResourcesPass
AnnotateOpOrdinalsPass
TestConversionPass
TestFloatRangeAnalysisPass

[IREE::VM]
ConversionPass
ReifyRodataTablesPass
HoistInlinedRodataPass
DeduplicateRodataPass
ResolveRodataLoadsPass
GlobalInitializationPass
OrdinalAllocationPass
DropEmptyModuleInitializersPass
DropUnusedCallsPass
SinkDefiningOpsPass

<test function용 pass>
ValueLivenessTestPass
RegisterAllocationTestPass

[IREE::VMVX]
ConversionPass
MaterializeConstantsPass
ResolveBufferDescriptorsPass


```



### 1.2. registerCodegenPasses
- 실제로 사용될 code를 생산하기 위한 pass들을 등록하는 곳
- 등록된 Pass들을 Tree로 표현
- 참조 기본경로 /path/to/iree/compiler/src/iree/compiler/
```
registerCodegenPasses
├── registerCodegenCommonPasses (~/Codegen/Common/Passes.td)
├── registerCodegenCommonCPUPasses (~/Codegen/Common/CPU/Passes.td)
├── registerCodegenCommonGPUPasses (~/Codegen/Common/GPU/Passes.td)
├── registerCodegenLLVMCPUPasses (~/Codegen/LLVMCPU/Passes.td)
├── registerCodegenLLVMGPUPasses (~/Codegen/LLVMGPU/Passes.td)
├── registerCodegenROCDLPasses (~/Codegen/LLVMGPU/Passes.td)
├── registerCodegenSPIRVPasses (~/Codegen/SPIRV/Passes.td)
├── registerCodegenVMVXPasses (~/Codegen/VMVX/Passes.td)
├── registerCodegenWGSLPasses (~/Codegen/WGSL/Passes.td)
├── registerIREEGPUPasses (~/Codegen/Dialect/GPU/Transform/Passes.td)
└── registerIREEVectorExtPasses (~/Codegen/Dialect/VectorExt/Transforms/Passes.td)
```
- 최종 Pass들을 목록화
```
[CodegenCommonPasses] ***
너무 많음. /path/to/iree/compiler/src/iree/compiler/Codegen/Common/Passes.td 참고

[CodegenCommonCPUPasses]
CPULowerToUKernelsPass
CPUPrepareUkernelsPass
CPUPropagateDataLayoutPass

[CodegenCommonGPUPasses]
너무 많음. /path/to/iree/compiler/src/iree/compiler/codegen/Common/GPU/Passes.td 참고

[CodegenLLVMCPUPasses]
(꽤 많지만 참고할만한 부분이라 모두 작성함)
ConvertToLLVMPass
ExpandF16OpToF32Pass
LLVMCPUAssignConstantOrdinalsPass
LLVMCPUAssignImportOrdinalsPass
LLVMCPUCheckIRBeforeLLVMConversionPass
LLVMCPUEmitVectorizationRemarksPass
LLVMCPULinkExecutablesPass
LLVMCPULowerExecutableTargetPass
LLVMCPUMmt4dVectorLoweringPass
LLVMCPUPeelPass
LLVMCPUSelectLoweringStrategyPass
LLVMCPUSplitReductionPass
LLVMCPUSynchronizeSymbolVisibilityPass
LLVMCPUTilePass
LLVMCPUTileToVectorSizePass
LLVMCPUTileAndFuseProducerConsumerPass
LLVMCPUVerifyVectorSizeLegalityPass
LLVMCPU2DScalableTo1DScalablePass
LLVMCPUUnfuseFMAOpsPass
LLVMCPUVirtualVectorLoweringPass
LLVMCPUVectorTransposeLoweringPass
LLVMCPUVectorShapeCastLoweringPass
VectorContractCustomKernelsPass
VerifyLinalgTransformLegalityPass

[CodegenLLVMGPUPasses]
AMDGPUEmulateNarrowTypePass
ConvertToNVVMPass
ConvertToROCDLPass
ExtractAddressComputationGPUPass
LLVMGPUAssignConstantOrdinalsPass
LLVMGPUCastAddressSpaceFunctionPass
LLVMGPUCastTypeToFitMMAPass
LLVMGPUConfigureTensorLayoutsPass
LLVMGPULinkExecutablesPass
LLVMGPULowerExecutableTargetPass
LLVMGPUPackSharedMemoryAllocPass
ROCDLPrefetchSharedMemoryPass
LLVMGPUSelectLoweringStrategyPass
LLVMGPUTensorCoreVectorizationPass
LLVMGPUTileAndDistributePass
LLVMGPUVectorDistributePass
LLVMGPUVectorLoweringPass
LLVMGPUVectorToGPUPass
TestLLVMGPUScalarizeMathOpPass

[CodegenSPIRVPasses]
ConvertToSPIRVPass
SPIRVAnnotateWinogradLoopsPass
SPIRVBreakDownLargeVectorPass
SPIRVConvertGPUTargetPass
SPIRVEmulateI64Pass
SPIRVEraseStorageBufferStaticShapePass
SPIRVFinalVectorLoweringPass
SPIRVInitialVectorLoweringPass
SPIRVLinkExecutablesPass
SPIRVLowerExecutableTargetPass
SPIRVLowerExecutableUsingTransformDialectPass
SPIRVMapMemRefStorageClassPass
SPIRVMaterializeExecutableConditionsPass
SPIRVSelectLoweringStrategyPass
SPIRVTileAndDistributePass
SPIRVTileAndPromotePass
SPIRVTileToCooperativeOpsPass
SPIRVTrimExecutableTargetEnvPass
SPIRVVectorizeLoadStorePass
SPIRVVectorizeToCooperativeOpsPass
SPIRVVectorToGPUSubgroupMMAPass

[CodegenVMVXPasses]
VMVXAssignConstantOrdinalsPass
VMVXSelectLoweringStrategyPass
VMVXLinkExecutablesPass
VMVXLowerExecutableTargetPass
VMVXLowerLinalgMicrokernelsPass

[CodegenWGSLPasses]
- 구현안됨
  
[IREEGPUPasses]
CombineBarrierRegionsPass
DistributeInnerTiledToLanesPass
ExpandUndistributedInnerTilesPass
LowerIREEGPUOpsPass
UnrollToIntrinsicsPass
VectorizeIREEGPUOpsPass

[IREEVectorExtPasses]
VectorizeIREEVectorExtOpsPass
VectorExtFoldUnitExtentDimsPass
```


### 1.3 registerMlirPasses
- 기본적으로 llvm project의 mlir에서 제공되는 pass들을 등록
```
registerMlirPasses
├── Core Transforms
│   ├── registerCanonicalizerPass
│   ├── registerCSEPass
│   ├── registerInlinerPass
│   ├── registerLocationSnapshotPass
│   ├── registerLoopCoalescingPass
│   ├── registerLoopInvariantCodeMotionPass
│   ├── registerAffineScalarReplacementPass
│   ├── registerPrintOpStatsPass
│   ├── registerViewOpGraphPass
│   ├── registerStripDebugInfoPass
│   ├── registerSymbolDCEPass
│   ├── registerBufferizationPasses
│   └── registerConvertComplexToStandardPass
├── Generic Conversions
│   └── registerReconcileUnrealizedCastsPass
├── Affine
│   ├── registerAffinePasses
│   └── registerLowerAffinePass
├── Arm SME
│   └── registerArmSMEPasses
├── GPU
│   └── registerGPUPasses
├── Linalg
│   └── registerLinalgPasses
├── LLVM
│   └── registerConvertArmNeon2dToIntrPass
├── MemRef
│   └── registerMemRefPasses
├── SCF
│   ├── registerSCFParallelLoopFusionPass
│   ├── registerSCFParallelLoopTilingPass
│   └── registerSCFToControlFlowPass
├── Shape
│   └── registerShapePasses
├── SPIR-V
│   ├── registerSPIRVLowerABIAttributesPass
│   ├── registerConvertGPUToSPIRVPass
│   ├── registerConvertControlFlowToSPIRVPass
│   └── registerConvertFuncToSPIRVPass
└── Transform Dialect
    └── registerTransformPasses

```


## 2. registerVMTargets (더 자세히 조사 필요)

- 특정 module operation을 읽어서 ostream에 bytecode를 저장


## 3. registerAllDialects
- Dialect들을 등록
- MLIR의 Dialect와 IREE의 Dialect를 등록

### 3.1 registerMlirDialects
- MLIR에서 기본적으로 제공되는 Dialect들이 등록됨.

```
registerMlirDialects
├── Core / IR 기본
│   ├── arith::ArithDialect
│   ├── func::FuncDialect
│   ├── tensor::TensorDialect
│   ├── memref::MemRefDialect
│   ├── scf::SCFDialect
│   ├── cf::ControlFlowDialect
│   ├── complex::ComplexDialect
│   ├── math::MathDialect
│   ├── shape::ShapeDialect
│   ├── ub::UBDialect
│   └── ml_program::MLProgramDialect
├── High-level / Algorithmic
│   ├── affine::AffineDialect
│   ├── linalg::LinalgDialect
│   ├── vector::VectorDialect
│   ├── quant::QuantDialect
│   └── shard::ShardDialect
├── GPU / Accelerator
│   ├── gpu::GPUDialect
│   ├── nvgpu::NVGPUDialect
│   ├── amdgpu::AMDGPUDialect
│   ├── NVVM::NVVMDialect
│   ├── ROCDL::ROCDLDialect
│   └── spirv::SPIRVDialect
├── LLVM / Low-level
│   └── LLVM::LLVMDialect
├── ARM Architecture
│   ├── arm_neon::ArmNeonDialect
│   ├── arm_sve::ArmSVEDialect
│   └── arm_sme::ArmSMEDialect
├── Pattern / Meta
│   ├── pdl::PDLDialect
│   └── pdl_interp::PDLInterpDialect
└── Transform Dialect
    └── transform::TransformDialect

```

- MLIR에서 가져온 Dialect들과 연동하기 위해 Op Interface / Pass 연동 / Transform Dialect 확장 등 기능을 해주는 함수들 추가
```
  // clang-format on
  cf::registerBufferizableOpInterfaceExternalModels(registry);
  func::registerInlinerExtension(registry);
  LLVM::registerInlinerInterface(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);

  // Register all transform dialect extensions.
  affine::registerTransformDialectExtension(registry);
  bufferization::registerTransformDialectExtension(registry);
  func::registerTransformDialectExtension(registry);
  gpu::registerTransformDialectExtension(registry);
  linalg::registerTransformDialectExtension(registry);
  memref::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);
  tensor::registerTransformDialectExtension(registry);
  transform::registerLoopExtension(registry);
  vector::registerTransformDialectExtension(registry);
```

### 3.2 registerIreeDialects
- IREE의 Dialect들을 등록
```
registerIreeDialects
└── IREE Dialects
    ├── IREE::CPU::IREECPUDialect
    ├── IREE::Codegen::IREECodegenDialect
    ├── IREE::Flow::FlowDialect
    ├── IREE::GPU::IREEGPUDialect
    ├── IREE::HAL::HALDialect
    ├── IREE::HAL::Inline::HALInlineDialect
    ├── IREE::HAL::Loader::HALLoaderDialect
    ├── IREE::IO::Parameters::IOParametersDialect
    ├── IREE::LinalgExt::IREELinalgExtDialect
    ├── IREE::Encoding::IREEEncodingDialect
    ├── IREE::Stream::StreamDialect
    ├── IREE::TensorExt::IREETensorExtDialect
    ├── IREE::Util::UtilDialect
    ├── IREE::VM::VMDialect
    ├── IREE::VMVX::VMVXDialect
    └── IREE::VectorExt::IREEVectorExtDialect
```
- 마찬가지로 각종 extension과 interface를 등록
```
  // External models.
  registerExternalInterfaces(registry);
  registerCodegenInterfaces(registry);
  registerGlobalOptimizationInterfaces(registry);
  registerUKernelBufferizationInterface(registry);

  // Register transform dialect extensions.
  registerTransformDialectPreprocessingExtension(registry);
  IREE::Util::registerTransformDialectExtension(registry);
```


## 4. registerLLVMIRTranslations
- LLVM project에서 제공되는 translations를 등록
  ![[register_llvm_translations.png]]- IREE는 IR을 변환시키는 방법으로 2가지 방법을 모두 사용하고 있다.

## 1. PASS를 이용한 방법
- MLIR의 정석적인 방법으로, rewritePattern을 정의하고, Pattern을 Conversion 및 Pass에 등록 후 runOnOperator에서 변환을 적용하는 방법
- IREE에 있는 Canonicalize.cpp를 예시로 Pass를 이용하는 예시를 보여줌
- 1) rewritePattern 정의
  -> 아래 처럼 정의, structure 선언에 사용된 argument를 이용하여 해당 operation에 해당하는 location을 가져올 수 있음
  -> matchAndRewrite 함수에서 location을 중심으로 pattern을 찾고, 새로운 operation으로 rewrite하는 규칙을 정의
```
struct FoldFullInsertSlice : public OpRewritePattern<tensor::InsertSliceOp> {
  using Base::Base;
  
  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!insertSliceOp.hasUnitStride() || !insertSliceOp.hasZeroOffset()) {
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "non-unit stride or non-zero offset.");
    }
  
    RankedTensorType sourceType = insertSliceOp.getSourceType();
    RankedTensorType resultType = insertSliceOp.getResultType();
    if (sourceType != resultType) {
      return rewriter.notifyMatchFailure(
          insertSliceOp,
          "unimplemented: Cast-like or reshape-like insert ops.");
    }
  
    std::optional<SmallVector<OpFoldResult>> mixedSizes =
        getDefiningMixedSizes(insertSliceOp.getDest());
        
    if (!mixedSizes) {
      return rewriter.notifyMatchFailure(
          insertSliceOp, "Could not find producer with list of tensor sizes.");
    }
  
    for (auto [insertSize, destSize] :
         llvm::zip_equal(insertSliceOp.getMixedSizes(), mixedSizes.value())) {
      if (isa<Value>(insertSize) || isa<Value>(destSize)) {
        if (insertSize != destSize) {
          return rewriter.notifyMatchFailure(insertSliceOp,
                                             "dynamic size mismatch");
        }
        continue;
      }
  
      // `getMixedSizes` for different ops returns different attribute types
      // (`index` or `i64`) so we compare the values of the ints directly here.
      int64_t staticInsertSize = getConstantIntValue(insertSize).value();
      int64_t staticDestSize = getConstantIntValue(insertSize).value();
      if (staticInsertSize != staticDestSize) {
        return rewriter.notifyMatchFailure(insertSliceOp,
                                           "static size mismatch");
      }
    }
  
    rewriter.replaceOp(insertSliceOp, insertSliceOp.getSource());
    return success();
  }
};  
```

- 2) rewrite pattern을 Pass에 추가
  -> Pass Structure에서 Initialize에 Pattern을 추가
```
struct CanonicalizePass : public impl::CanonicalizePassBase<CanonicalizePass> {
  using IREE::Flow::impl::CanonicalizePassBase<
      CanonicalizePass>::CanonicalizePassBase;
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Inherit the same config defaults from the upstream canonicalizer pass.
    config.setUseTopDownTraversal().setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Normal);
  
    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);
  
    // Pull in some borderline/downstream canonicalizations for the Flow
    // compilation phase.
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(owningPatterns);
    owningPatterns.add<FoldFullInsertSlice>(context);
    owningPatterns.add<AffineApplyLowering>(context);
  
    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }
...
};

void mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(
    RewritePatternSet &patterns) {
  patterns.add<MergeConsecutiveExtractSlice,
               MergeConsecutiveInsertSlice<InsertSliceOp>,
               MergeConsecutiveInsertSlice<ParallelInsertSliceOp>>(
      patterns.getContext());
}

```

- 3) runOnOperation으로 pattern 적용
  -> Pass Structure에 runOnOperation에서 applyPatterns를 통해 Pass가 적용되는 시점에 2)에서 추가한 패턴이 적용됨
```
struct CanonicalizePass : public impl::CanonicalizePassBase<CanonicalizePass> {
	
	...
	
  void runOnOperation() override {
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    config.enableConstantCSE(cseConstants);
    LogicalResult didConverge =
        applyPatternsGreedily(getOperation(), *patterns, config);
    if (this->testConvergence && failed(didConverge)) {
      getOperation()->emitError("Canonicalizer failed to converge");
      return signalPassFailure();
    }
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};
```

- 4) 예제 : 적용 전
```
util.func public @fold_full_insert_into_extract(
    %source: tensor<8x?xf32>,
    %dest: tensor<10x?xf32>,
    %size: index) -> tensor<8x?xf32> {
  %extract = tensr.extract_slice %dest [1, 1] [8, %size] [1, 1] : tensor<10x?xf32> to tensor<8x?xf32>
  %insert = tensor.insert_slice %sorce into %extract [0, 0] [8, %size] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
  util.return %insert : tensor<8x?xf32>
}
```

- 5) 예제 : 적용 후
```
module {
  util.func public @fold_full_insert_into_extract(%arg0: tensor<8x?xf32>, %arg1: tensor<10x?xf32>, %arg2: index) -> tensor<8x?xf32> {
    util.return %arg0 : tensor<8x?xf32>
  }
	...
}
```

## 2. transform Dialect를 이용한 방법
- transform Dialect를 이용하여 mlir 내부에 변환규칙을 추가하는 방법 ([링크](https://mlir.llvm.org/docs/Dialects/Transform/#overview))
- fuse_consumer 예제를 통한 기본 원리 파악
- 아래와 같은 mlir을 보면, 변환하고자하는 source IR과 변환 규칙을 기술한 transform IR이 하나의 mlir 파일에 존재
```
// RUN: iree-opt %s --iree-transform-dialect-interpreter --transform-dialect-drop-schedule | FileCheck %s
  
#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0)>
func.func @pack_consumer_fusion(%arg0: tensor<32xf32>) -> tensor<2x16xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %1 = scf.forall (%arg1) in (2) shared_outs(%arg2 = %0) -> (tensor<32xf32>) {
    %3 = affine.apply #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[%3] [16] [1] : tensor<32xf32> to tensor<16xf32>
    %extracted_slice_0 = tensor.extract_slice %arg2[%3] [16] [1] : tensor<32xf32> to tensor<16xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice : tensor<16xf32>) outs(%extracted_slice_0 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %in : f32
      linalg.yield %5 : f32
    } -> tensor<16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %arg2[%3] [16] [1] : tensor<16xf32> into tensor<32xf32>
    }
  }
  %2 = tensor.empty() : tensor<2x16xf32>
  %pack = linalg.pack %1 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %2 : tensor<32xf32> -> tensor<2x16xf32>
  return %pack : tensor<2x16xf32>
}




// CHECK-LABEL: @pack_consumer_fusion
// CHECK:       scf.forall
// CHECK:         %[[GENERIC:.+]] = linalg.generic
// CHECK:         %[[PACK:.+]] = linalg.pack %[[GENERIC]]
// CHECK:         scf.forall.in_parallel {
// CHECK:           tensor.parallel_insert_slice %[[PACK]]
  
  
  
  
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %slice_op = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg0
    : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %arg0
    : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.iree.fuse_consumer %slice_op in (%loop)
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
     transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }
}
```
- 아래의 transform dialect에 정의된 규칙에 따라 IR을 변환, 아래는 변환결과
```
#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @pack_consumer_fusion(%arg0: tensor<32xf32>) -> tensor<2x16xf32> {
    %0 = tensor.empty() : tensor<32xf32>
    %1 = tensor.empty() : tensor<2x16xf32>
    %2 = scf.forall (%arg1) in (2) shared_outs(%arg2 = %1) -> (tensor<2x16xf32>) {
      %3 = affine.apply #map(%arg1)
      %extracted_slice = tensor.extract_slice %arg0[%3] [16] [1] : tensor<32xf32> to tensor<16xf32>
      %extracted_slice_0 = tensor.extract_slice %0[%3] [16] [1] : tensor<32xf32> to tensor<16xf32>
      %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice : tensor<16xf32>) outs(%extracted_slice_0 : tensor<16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = arith.addf %in, %in : f32
        linalg.yield %5 : f32
      } -> tensor<16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg1, 0] [1, 16] [1, 1] : tensor<2x16xf32> to tensor<1x16xf32>
      %pack = linalg.pack %4 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %extracted_slice_1 : tensor<16xf32> -> tensor<1x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %pack into %arg2[%arg1, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<2x16xf32>
      }
    }
    return %2 : tensor<2x16xf32>
  }
}
```
- IREE에서는 /path/to/iree/compiler/src/iree/compiler/Codegen/Common/의 TransformDialectInterpreterPass.cpp에서 transform dialect entry point를 만들어, transform dialect로 변환 규칙을 등록하면 해당지점에 변환규칙들이 추가되도록 해 둠.
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
    
    - 터미널 출력에 색상을 사용합니다. (default=autodetect)## 1. Importer
- Importer의 목적은 프레임워크 별로 서로 다르게 표현되어 있는 모델 표현식을 LLVM-MLIR 프로젝트에서 제공하는 기본 IR들 (특히, linalg 및 arith)을 이용하여 IREE의 정규식으로 표현하는 것.
- HLO도 동일한 역할을 하지만, IREE의 Middle-end로 이어지는 데 동일한 규칙을 가진 식들로 표현된다는 데 의의가 있음
- 그 구조를 보기 위해 Torch Input conversion을 확인
![[Pasted image 20260107172128.png]]
![[Pasted image 20260107172204.png]]
- 기본적으로 torch-mlir을 third-party로 두고 거기에 구현된 Pass들을 등록하여 IREE의 정규식이 되도록 구현
- onnx의 경우에는 onnx가 직접 IREE로 import되는 경우는 없으며, torch-mlir을 이용하여 torch의 dialect로 만든 뒤, TorchOnnxToTorch Pass를 이용하여 IREE의 정규식으로 들어오게 되어 있음.
![[Pasted image 20260107172447.png]]
- IREE의 정규식으로 편입이 필요하지만 정의가 되지 않은 경우에 위 그림처럼 RewritePattern부터 Conversion - Pass까지 구현이 되어 있음.
![[Pasted image 20260107172707.png]]
- onnx 변환을 예시로 보면 torch mlir에는 onnx를 torch mlir로 가져와 관리하기 위한 dialect와 operation들이 구현되어 있음. (여기에서 찾아봤을 때, QLinear가 포함된 Operation은 Conv와 Matmul뿐인 것을 확인)


## 2. Tools

- IREE는 라이브러리로서 지원하고자 함. 아직은 라이브러리로 활용하기는 부족하다고 생각됨
- IREE는 라이브러리화 시켜둔 source file들을 가져와 다양한 도구들을 제공하고 있고, 그 코드는 /path/to/iree/compiler/tools/에 모아 둠
- 가장 먼저 살펴봐야 할 것은 iree-compiler-main 코드, iree-run-module-main, iree-benchmark-module임
- 각각 import된 mlir을 compile하는 코드 / compile된 파일을 실행하는 런타임 코드 / benchmarking하는 코드다.

### 2.1 iree-compiler-main (컴파일러 코드 분석)
- import된 mlir을 compile하는 코드로, 실제로는 /path/to/iree/compiler/src/iree/compiler/Tools/iree_compiler_lib.cc : runIreecMain 함수에서 수행됨. 
- 자세한 분석은 [[컴파일러 코드 분석]]에서 진행

### 2.2 iree-run-module-main


### 2.3 iree-benchmark-module


## 3. vm bytecode와 hal executable을 생산하는 원리
- 동일 연산을 통해 hal executable과 vm bytecode를 모두 생산해 냄
	![[iree_lowering_to_binary.png|400]]
   
- example mlir 을 변환시키고 dump 된 파일을 확인하여 코드가 변해가는 과정을 통해 검증
```
[example.mlir]
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

- dump 결과(full_pipeline.mlir) 및 [[iree pipeline 흐름 분석]]
![[full_pipeline.mlir]]

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
![[full_pipeline.mlir]]

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
![[Pasted image 20260113115644.png]]
- vm function, function을 어떤 순서에 따라서 그리고 어떤 값을 넣을 지 계획획
![[Pasted image 20260113115726.png]]
- 결과를 export
![[Pasted image 20260113115919.png]]- **[[개발 파이프라인 정리]]**
- [[임시 기록 - Custom IR 설계를 위한 정보들 기입]]
- [[Custom IR 설계]]
## 1. 개요

 - [model -> 정적 그래프 생성 -> StableHLO -> MLIR -> IREE -> Binary] 순서로 compile 되도록 진행
- Model ~ StableHLO는 사용자가 실행, pytorch, onnx에 대해 예제만 제공
- iree-air-aie 구조를 채용, plugin으로 iree에 Dialect, Pass를 등록
![[iree-air-aie-architecture 2.png]]



- 개발 범주
![[KETI-Compiler-develop-area.png]]

- 전체 코드 구조 : 2가지 git으로 관리
	- RepresentHW git : 하드웨어에 종속된, general하게 사용할 수 있는 Dialect들과 Pass들을 정의하고 관리 + 하드웨어 구동에 필요한 device driver 등을 관리
	- PublishHW git : iree 등 다양한 middle-end 또는 front-end와 연동 시키기 위한 git 현재는 iree에 연동하는 것만 구상 중
![[AI-Compiler-code_outline.png]]
- [[컴파일러 적용과 mlir의 변화]]
## 2. 이론 및 개념

- [[정적 그래프]]
- [[StableHLO]]
- [[IREE 코드 구성]]
- [[Session과 Invocation]]
- [[IREE가 IR을 변환시키는 2가지 방법]]
- [[Plugin]]
- [[HAL Ops에서 vtable 함수포인터 호출 흐름]]
- [[컴파일러 관련 기본 이론들 정리]]
- [[IREE 옵션 설정에 대해]] 

## 3. Setup

- ubuntu20.04에서 진행, ubuntu22.04 및 ubuntu24.04 추천
- IREE는 26.01.06 기준 안정화 버전인 v3.9.0을 사용
### 3.1 IREE Build
1) torch mlir 설치
   - python3.11 설치 및 venv 가상환경 설정
   - 가상환경에서 pip 로 torch mlir 설치```
```
$ pip install --pre torch-mlir torchvision \ --extra-index-url https://download.pytorch.org/whl/nightly/cpu \ -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels onnx protobuf
```
1) IREE 빌드
    - iree를 clone하고 아래 두 줄을 수정
      <코드 수정> iree/compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:426에서 nullptr을 모두 false로 바꿈
      <코드 수정> iree/compiler/src/iree/compiler/Dialect/Util/Transforms/HoistIntoGlobals.cpp:354에서 const 제거
    ```
    $ sudo apt install cmake ninja-build clang lld
    $ git clone https://github.com/iree-org/ire.git
    $ git checkout tags/v3.9.0
    $ git submodule update --init
    $ cmake -G Ninja -B ../iree-build/
    $ cmake --build ../iree-build/
    ```

### 3.2 IREE 실행

3.2.1 Host CPU / Device CPU : standard 예제 - onnx 모델 사용
1) python3.11의 가상환경 실행
   ```
   $ source ./venv3.11/bin/activate
   ```
2) 모델을 IREE환경으로 import (front-end 실행)
```
$ cd ../iree-build/tools/
$ (save model to /path/to/iree-build/models/yolov8n.onnx)
$ python -m torch_mlir.tools.import_onnx ../models/yolov8n.onnx -o ../yolov8n.onnx.mlir
```
3) IREE 컴파일 수행
   ```
   $ ./iree-compile ../models/yolov8n.onnx.mlir --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-cpu=host -o ../models/yolov8n_cpu.vmfb
   ```
4) 런타임 수행
   - 실행 전, yolov8n.onnx.mlir을 열어 어떤 function으로 구성되어 있는지 확인 후 실행
   - input은 임의로 0으로 초기화 된 텐서를 입력, 실제 데이터를 확인하려면 .npy로 저장한 데이터를 입력으로 주어 실행
   - 결과값을 npy로 저장하려면 --output=@filename.npy 옵션을 추가
   ```
   $ ./iree-run-module --module=../models/yolov8n_cpu.vmfb --device=local-task --function=main_graph --input=1x3x640x640xf32=0
   ```


### 3.3 Compiler Example & Structure Build
- 구조를 설계하고 해당 구조에서 Dialect 등록 및 Pass 적용이 잘 이루어지는지 확인하기
- linalg.matmul -> hw1.matmul로 변경하는 실험을 진행. simple MLP를 변환하려 했으나 이를 위해서는 많은 계층의 IR들을 구현해야 할 필요가 있어, 간단하게 pass 1개가 적용되는지만 실험
- 3.1 / 3.2 를 진행하지 않고도 진행할 수 있지만, torch-mlir을 설치하고 진행하길 권장
<br>
3.3.1 Build
- 최초 1번은 LLVM, MLIR, IREE까지 모두 빌드하기 때문에 오래걸림
```
$ git clone --recursive https://github.com/sedie1234/PublishHW.git 
$ git checkout [dbde48d]
$ cd PublishHW
$ mkdir build
$ cmake -G Ninja -B ./build -S third_party/iree   \
-DIREE_CMAKE_PLUGIN_PATHS=$PWD   \
-DIREE_BUILD_PYTHON_BINDINGS=OFF   \
-DIREE_INPUT_STABLEHLO=ON   \
-DIREE_INPUT_TORCH=ON   \
-DIREE_INPUT_TOSA=OFF   \
-DIREE_HAL_DRIVER_DEFAULTS=ON   \
-DIREE_TARGET_BACKEND_DEFAULTS=OFF   \
-DIREE_TARGET_BACKEND_LLVM_CPU=ON   \
-DIREE_BUILD_TESTS=ON   \
-DIREE_BUILD_SAMPLES=OFF  \
-DTARGET_DEVICE="HW1"
```
<br>
3.3.2 실행
- 아래 코드를 복사하여 test.mlir을 만듬
```
func.func @test_matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
```
- iree-opt를 통해 linalg-to-hw1 pass를 적용 : iree-opt는 컴파일러 개발의 디버깅을 위해 기능을 하나씩 테스트 할 수 있도록 만들어진 도구
```
$ ./build/tools/iree-opt test.mlir --pass-pipeline="builtin.module(linalg-to-hw1)" > test_example.mlir 2>&1
```
- 아래와 같이 pass가 적용된 것을 확인
```
module {
  func.func @test_matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = hw1ir.matmul %arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32> -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
```

## 4. 주의사항
- [[양자화 관련]]



## 5. 컴파일러 구현에 필요한 사전 작업 및 실험들
- StableHLO 적용 실험 (TODO : 링크, 문서작성)
	- onnx와 torch 모델에 대해 StableHLO를 적용해 HLO Level의 MLIR을 생성하고 cpu모드로 컴파일 시켜보기
	- 모델은 yolo와 qdq quantized yolo로 실험
	- import 직후의 mlir 확인

- [[torch simpleMLP test]]

## 6. File naming & 구성요소 정리 (용어정리)
- File naming 
	- xxxIR.td : xxx Dialect의 operation이 구현되어 있는 tablegen file
	- xxxIROpBase.td : xxx Dialect의 operation이 사용할 수 있는 Attribute나 Interface를 정의하고 있는 tablegen file
	- xxxTransformOps.td : xxx Dialect의 변환규칙을 Transform Dialect의 operation으로 정의하고 있는 tablegen file
	- xxxDialect.h / .cpp : 정의한 xxx Dialect를 등록하기 위한 파일
	- Passes.td : Pass들을 정의한 tablegen file 
	- Passes.h / .cpp : 정의한 Pass들을 등록하기 위한 파일
	- xxxPass.h / .cpp : 특정 변환을 위한 Pass를 구성하는 Conversion들을 정의한 파일

- 구성요소 : [[MLIR 구성요소와 사용법 정리]]

# 7. 개발 단계

## 7.1 동작만 하는 수준의 구현

- IR은 후일의 최적화까지 고려하여 구성하지만 pass 자체는 최적화를 고려하지 않고 진행한다.
- [[변환해야 할 Ops 정리 - Conv-Activation 편]]

## 7.2 VLIW 최적화

- 병렬성이 있는 연산을 찾아 VLIW를 적용한다.

## 7.3 일반적인 최적화 기술 적용

- loop unrolling이나 연산 순서 바꾸는 등, 기본적인 기술들을 적용해본다.

## 7.4 실험적인 최적화 기술들을 적용

- 실험적인 최적화 기술들을 적용해본다.

## 7.5  프로파일링 기법

- 다양한 방법으로 컴파일 할 수 있도록 하고, profiling용 ops를 이용하여 프로파일링 후, 가장 최적의 방법을 채택하여 컴파일한다.- 주로 VLIW 하드웨어에 적합한 컴파일러 제작 관점에서 작성되었음
# 1. Loop Unrolling

- 루프를 해체함. basic block이 커져서 basic block 내에 더 많은 최적화기술을 적용할 수 있다. 

# 2. SSA(Static Single Assignment)

- 모든 변수는 한 번만 할당된다는 원칙으로, 동일 변수 및 참조를 가지고 있더라도 다른 이름을 부여하여 구분한다.
```
Original>>
x = 1;
x = x + 1;

SSA>>
x1= 1;
x2 = x1 + 1;
```
- 이 과정을 통해 동일한 변수에 대해 다른 상태를 부여하여 WAW, WAR hazard를 회피할 수 있다.
- 과거의 register renaming 기법의 완벽한 상위호환 기법
- MLIR은 기본적으로 SSA로 mlir을 관리한다.

# 3. Speculative execution

- 조건 분기에서 결과를 예측하여 미리 계산을 진행하는 기법
- 예측에 성공하면 성능이 비약적으로 상승, 실패하면 진행 중이던 연산을 모두 버리고, 버블정도의 시간 손실이 발생함.
- 예측에 확률을 접목하면 성능을 더욱 끌어올릴 수 있다.

# 4. Memory disambiguation

- 두 연산의 순서를 바꿀 때, 접근하는 메모리 주소가 겹치는지 분석하여 3가지 상태로 분류
- 1) No-Alias : 절대 겹치지 않음. 컴파일러가 연산의 순서를 마음대로 바꿀 수 있음
- 2) May-Alias : 겹칠 가능성이 있음. 순서도 바꿀 수 없고, 두 메모리를 연결할 수도 없는 상태로, 가장 성능이 떨어지는 방향의 상태
- 3) Must-Alias : 반드시 같은 주소임. 데이터를 메모리에 쓰지 않고 바로 전달하는 최적화가 가능

# 5. Bubble

- 컴파일러 수준에서 bubble을 수행할 수 있음. 명령어 사이에 기다려야하는 시간에 nop를 넣음으로써 해결
- 조금 더 최적화하기 위해 nop구간에 다른 독립적인 연산을 실행시키는 것이 가장 좋다.
- bubble이 필요한 두 구간을 합치는 것이 가장 이상적

# 6. Dead Code Elimination (DCE)

- 결과값을 아무도 사용하지 않는다면, 해당 연산을 통째로 삭제
- 단, side effect가 없어야 함. Ops 선언시에 side effect로써 Pure를 준 경우, 부수효과가 없다는 것을 의미하며, 결과값이 아무런 영향도 주지 않는다면 해당 연산을 통째로 삭제한다.
- MLIR에서 기본적으로 제공하는 최적화

# 7. Common Subexpression Elimination (CSE)

- 동일한 입력으로 같은 연산을 두 번 한다면, 한 번만 계산하고 결과를 재사용
- 마찬가지로 side effect가 없는 경우 MLIR에서 기본적으로 제공하는 최적화에 의해 적용됨


# 8. Hoisting/Reordering

- 연산의 순서를 바꾸거나 루프 밖으로 꺼내는 행위
- 마찬가지로 side effect가 없는 경우 MLIR에서 기본적으로 제공하는 최적화에 의해 적용됨
- /path/to/iree/compiler/src/iree/compiler/Tools/iree_compiler_lib.cc : runIreecMain 함수 분석
- 해당 함수에 인용되는 핵심 함수들은 링크를 걸어 분석해 둘 예정

# 1. 코드 구조
- 코드 구조는 크게 4단계로 나누어져 있음.
- 컴파일 옵션 / 초기화 / 컴파일 람다함수 / 조건분기 컴파일 실행(현재는 multi-input 유무 정도로 분기됨)

# 2. 컴파일 옵션
- 컴파일 옵션은 llvm::cl::opt를 이용
	![[compile option 예시.png]]
	사용 가능한 옵션과 초기 설정 값, 각 옵션에 대한 설명이 기재되어 있음

# 3. 초기화
- session의 state를 관리하는 객체 초기화 : [[Session과 Invocation]]의 개념 참고
- RAII 패턴으로 관리

# 4. 컴파일 람다함수
- 실제 컴파일을 수행하는 함수로, compile의 source를 입력받아 output을 session state에 저장하고 성공여부를 bool 타입으로 반환
- [&]를 사용하여 지역변수를 모두 참조할 수 있도록 구성
- 1) 아래와 같이 초기 부분에서는 session에서 invocation으로 필요한 정보들을 불러옴
  [[ireeCompilerGlobalInitialize 분석]] -> 많은 Dialect, Target Device, Pass 들이 여기에서 등록됨
  ![[compile_lamda_invocation_init.png]]
- 2) compile mode에 따라 Invocation Pipeline을 수행 -> 내부함수 [[runPipeline 분석]]
  ![[compile_lamda_invocation_pipeline.png]]
- 3) output format에 따라 session state에 output을 저장
  ![[compile_lamda_save_output.png]]

# 5. 컴파일 실행
- 4번에서 작성한 람다함수(processBuffer)로 컴파일 수행, output에 저장 (output 포인터를 파일포인터를 주었으므로 파일에 저장됨)
  ![[Pasted image 20260108112005.png]]

# 6. 번외
- C API로 제공되는 것을 활용하여 컴파일 작성하는 [예제](https://iree.dev/reference/bindings/c-api/#plugins) 
- IREE 라이브러리를 읽어와서, session과 invocation을 열고 compile을 수행하는 예제 : 컴파일러를 라이브러리화 하고 사용하고자 하는 방향성이 보임
```
#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>

int main(int argc, char** argv) {
  // Load the compiler library then initialize it.
  ireeCompilerLoadLibrary("libIREECompiler.so");
  ireeCompilerGlobalInitialize();

  // Create a session to track compiler state and set flags.
  iree_compiler_session_t *session = ireeCompilerSessionCreate();
  ireeCompilerSessionSetFlags(session, argc, argv);

  // Open a file as an input source to the compiler.
  iree_compiler_source_t *source = NULL;
  ireeCompilerSourceOpenFile(session, "input.mlir", &source);

  // Use an invocation to compile from the input source to one or more outputs.
  iree_compiler_invocation_t *inv = ireeCompilerInvocationCreate(session);
  ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD);

  // Output the compiled artifact to a file.
  iree_compiler_output_t *output = NULL;
  ireeCompilerOutputOpenFile("output.vmfb", &output);
  ireeCompilerInvocationOutputVMBytecode(inv, output);

  // Cleanup state.
  ireeCompilerInvocationDestroy(inv);
  ireeCompilerOutputDestroy(output);
  ireeCompilerSourceDestroy(source);
  ireeCompilerSessionDestroy(session);
  ireeCompilerGlobalShutdown();
}
```
- 양자화는 Convolution이나 Matmul 연산에서 연산 가속 및 메모리 절약에 이점이 있음
- 프레임워크 및 런타임에만 의존하는 형태의 인공지능 모델 실행에서는 Add, Mul, Sigmoid 등 함수를 양자화 하지 않을 경우, 양자화 된 Conv/Matmul 연산 사이에서 값을 dequantize/quantize를 새로 해야 하기 때문에 이 부분에서 병목이 발생했음
- 컴파일러를 사용할 때는 이 사이의 병목이 크게 영향이 없을 정도로 최적화가 가능하다는 것이 일반적인 여론으로 보이기 때문에 해당 연산들의 quantized operation이 Front-end에서 Middle-end로 연결되는 기능을 대부분 정의해두지 않음.
- 컴파일러를 사용하게 되면 Add, Mul, Sigmoid와 같은 연산들은 어짜피 최적화가 되기 때문에 이것을 양자화 했을 때 얻는 시간 절약에 대한 이점보다 정확도면에서 손해보는 것이 더 크다고 판단하는 것 같음
- 때문에 이를 구현하기 위해서는 지원하는 프레임워크 -> HLO로 변환되는 Front-end의 변환규칙을 모두 정의해주어야 함.
- ==이 부분은 HLO에서 [dequantize - sigmoid - quantize]를 고정하고 Middle-end의 Pass에 [dequantize - sigmoid - quantize] = [QLinearSigmoid]로 변환하는 규칙들의 Pass를 적용하는 식으로 해결할 계획==
- 260205 기록 -> IR outline 설계 완료
- 260205 기록 -> model들은 int8 quantize model으로 진행
# 0. 업무 진행 순서


## 0.0 variant 1개로 만들기 (1차 완료)

- 무조건 1개의 variant로 만들기 성공.
- 추후 cpu와 디바이스의 정보를 입력하여 N개의 variant로 만드는 로직을 짤 필요가 있음.
- third_party/iree/compiler/src/iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.cpp를 수정하여 해결, iree를 새로 받아오는 경우에 파일을 복사해 넣을 수 있도록 ./RegionOpUtils_one_variant.cpp로 저장해 둠. 나중에 iree 빌드 전에 복사해 넣어서 사용.

## 0.1 Conv-Sigmoid-Mul 1 layer 모델 : top3 node model (진행 중)
- **내용 :** yolo의 시작지점에 위치한 세 노드만으로 구성된 작은 모델을 구성하여 컴파일 / 런타임을 적용해 봄. 
- **기간 :** c-model이 나오기 전까지 완성
- **목적 :** yolo 모델 시작 전 준비 모델 + c-model 연동용

### 0.1.1 상위 Ops 리스트 (1차 완료)
- --iree-global-opt-experimental-disable-conv-generalization를 사용했을 때를 기준으로 함
- [[변환해야 할 Ops 정리 - Conv-Activation 편]]
- 




### 0.1.2 사용 가능한 HAL Ops 리스트 (진행 중)
- 하드웨어 담당분들께 검토요청 부탁함. 진행 중

### 0.1.3 사용할 IR들 과 Pass들 설계 (1차 완료)
- 설계를 위한 정보 수집 중
- [[Custom IR 설계]]


### 0.1.4 layout 정의를 위한 사용가능한 하드웨어정보들 (진행 중)
- 하드웨어 담당분들께 요청드림. 진행 중

### 0.1.5 각 계층별 layout 정의 (진행 중)

### 0.1.6 IR 구현 (진행 중)

### 0.1.7 Pass 구현 (진행 중)


### 0.1.8 검증 (진행 예정)

### 0.1.9 컴파일러 완성 (진행 예정)
- 0.1.6 ~ 0.1.8을 반복하며 컴파일러 완성

### 0.1.10 런타임 완성 (진행 예정)
- 런타임을 완성하여 최종확인




--------------------------------------------------------------------------

## 0.2 yolov10n 모델 (진행 예정)
- **내용 :** 비전용 모델
- **기간 :** ~ 4월
- **목적 :** 비전용 모델로 보여주기 좋은 첫 타겟모델 + 필요한 경우 데모까지 보여줄 수 있는 점

## 0.3 Face 모델 (진행 예정)
- **내용 :** yolo에 attention이 추가된 모델
- **기간 :** ~ 5월
- **목적 :** 내부/대외적으로 진행상황을 공유

## 0.4 모델 미정 - 




# 1. IREE에 배포하기 위한 전체 구성



# 2. IR (Dialect)
- [[Custom IR 설계]]
- 하드웨어를 IREE에 등록하기 위해 개발할 IR들 
- 항목 표기 방법 : \[Dialect이름 + "IR"\] / \[git위치 (RepH or PubH)\] / 구현 수준(Lv0 ~ LV3)
- 하단에 간단한 IR의 의미 설명
- 링크를 걸어 내부에 자세한 구현사항 기입
	- 반드시 포함해야 할 내용 : Ops, data layouts, attributes, IR의 역할 설명

# 3. PassPipelines
- [[Custom IR 설계]]
- MLIR에서는 Pass가 변환에서 가장 상위 개념이지만, 이 페이지에는 그 상위 개념인 PassPipeline을 항목으로 기입
- 항목 표기 방법 : \[PassPipeline 이름 + "PassPipeline"\] / \[git위치 (RepH or PubH)\] / 구현 수준(Lv0 ~ LV3)
- 하단에 간단한 PassPipeline의 목적 설명
- 링크를 걸어 내부에 자세한 구현사항 기입
	- 반드시 포함해야 할 내용 : 구성 Pass와 그 발현순서, PassPipeline의 역할 설명

# 4. 구현수준

## 4.1 Lv0

- 최소 동작을 위해 반드시 구현해야 함

## 4.2 Lv1

- 최소한의 최적화를 제공

## 4.3 Lv2

- 연산에 대해 오류가 생길 가능성이 있는 수준의 최적화 제공

## 4.4 Lv3

- 실험적인 최적화


# 5. 컴파일러 개발 partial work flow

## 5.1 plugin에 들어오는 연산 확인하기

### 5.1.1 onnx 모델 준비
- onnx모델을 준비한다. 
- 양자화가 필요한 경우,  qdq로 양자화 한다. 
- dummy calibration으로 qdq 양자화 하는 python code
```
import onnx
from onnx import version_converter
import numpy as np
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader

input_model_path = "yolov10_top3.onnx"
intermediate_path = "dummy.onnx" # 중간 변환 파일
output_model_path = "yolov10_top3_qdq.onnx"
target_opset = 17
input_tensor_name = "images" # 실제 모델의 Input 이름 확인 필요
input_shape = (1, 3, 640, 640)

print(f"Step 1: Converting model to Opset {target_opset}...")
model = onnx.load(input_model_path)

try:
    model = version_converter.convert_version(model, target_opset)
    onnx.save(model, intermediate_path)
    print(" -> Conversion successful using onnx.version_converter")
except Exception as e:
    print(f" -> Standard conversion failed: {e}")
    print(" -> Fallback: Forcing Opset version in header (Use with caution)")
    model.opset_import[0].version = target_opset
    onnx.save(model, intermediate_path)

class QuickFakeReader(CalibrationDataReader):
    def __init__(self):
        self.done = False
    def get_next(self):
        if self.done: return None
        self.done = True
        return {input_tensor_name: np.random.rand(*input_shape).astype(np.float32)}

print("Step 2: Quantizing to QDQ format...")
quantize_static(
    model_input=intermediate_path,  # Opset 17로 변환된 모델을 입력으로 사용
    model_output=output_model_path,
    calibration_data_reader=QuickFakeReader(),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8
)
print(f"Done! Saved to {output_model_path} (Opset {target_opset})")
```

### 5.1.2 torch mlir로 input mlir 만들기

```
$ iree-import-onnx yolov10_top3_qdq.onnx --opset-version 17 -o yolov10_top3_qdq.onnx.mlir
```

### 5.1.3 keti_hw1으로 컴파일하기

```
./iree-compile ~/workspace/testStage/work1/yolov10_top3_qdq.onnx.mlir --iree-hal-target-backends=keti_hw1 --mlir-print-ir-after-all --mlir-disable-threading --compile-to=executable-targets --iree-global-opt-experimental-disable-conv-generalization --iree-opt-generalize-matmul=false --mlir-elide-elementsattrs-if-larger=32 -o dummy.vmfb > ~/workspace/testStage/work1/yolov10_top3_qdq_pipeline.mlir 2>&1
```

```
./build_inf/tools/iree-compile ~/workspace/testStage/work1/yolov10_middle_3_5_qdq.onnx.mlir --iree-hal-target-backends=infetron_v2 --mlir-print-ir-after-all --mlir-disable-threading --compile-to=executable-targets --iree-global-opt-experimental-disable-conv-generalization --iree-opt-generalize-matmul=false --mlir-elide-elementsattrs-if-larger=32 -o dummy.vmfb > ~/workspace/testStage/work1/yolov10_middle_3_5_qdq_pipeline.mlir 2>&1
```
### 5.1.4 변환할 module 확인하기
\"===== \[Entry Module\]  ===== \" 을 검색하여 변환할 module을 확인한다.

## 5.2 Dialect 구현
- 추후 작성


## 5.3 Pass 구현
- 1) Passes.td에 구현할 Pass 정의
- 2) yourPass.h 작성 : create가 포함되어야 함. 
- 3) yourPass.cpp 작성 : 실제 변환이 정의되어야 함. [[#5.3.1 Pass 구조체 정의]]로 이동
- 4) Passes.h에 yourPass.h include
- 5) cmake 수정 : 현재 cmake는 모두 pub HW에서 관리하고 있음. -> Pass 타겟 생성 부분 수정
- 6) ConversionPasses.cpp 수정 <--- 여기까지 수정하면 iree-opt 사용가능
- 7) target.cpp의 buildTranslationPassPipeline에서 정의한 pass를 추가. 여기서부터는 pass가 순차적으로 적용되므로 주의 필요
### 5.3.1 Pass 구조체 정의
- 컴파일러에 등록할 Pass 구조제 정의
- 아래는 예시
```
struct OpFusionPass
    : public impl::OpFusionBase<OpFusionPass> {
    OpFusionPass() = default;
    OpFusionPass(const OpFusionPass &pass) : OpFusionBase(pass) {}
    void runOnOperation() override;
};
```

### 5.3.2 Pattern 정의
- Pass에서 실제 변환을 수행하는 MatchAndRewrite Pattern 정의
- 아래 문구를 시작으로 작성
```
class IREEToUnsymetricQConvPattern : public OpRewritePattern<linalg::GenericOp> {
public:
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
```

### 5.3.3 runOnOperation 작성
- 실제 변환 적용을 정의
- 아래는 예시
- 예시 이외에도 패턴 분석을 위한 attributes 추가 및 제거, 엄격한 규칙을 위한 illegal, legal 선언 등 작성 가능
```
void OpFusionPass::runOnOperation() {
    MLIRContext *context = &getContext();
    Operation *op = getOperation(); // ModuleOp 혹은 FunctionOp
    RewritePatternSet patterns(context);
    patterns.add<IREEToUnsymetricQConvPattern>(context);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
        signalPassFailure();
    }
}
```

### 5.3.4 create your Pass 작성
- 등록할 Pass 객체를 생성
```
std::unique_ptr<mlir::Pass> createOpFusionPass() {
  return std::make_unique<OpFusionPass>();
}
```



## 5.4 IREE Plugin으로 등록

## 5.5 iree-opt 로 Pass 및 Dialect 확인하기
```
./build_inf/tools/iree-opt --qdq-fusion ./test/QDQFusionPass/input.mlir -o ./test/QDQFusionPass/output.mlir
```
- Shape이 정해진 그래프, 컴파일에서 가장 문제가 되는 두 가지 중 하나로, Shape이 정적이어야 컴파일이 가능.
- IREE 페이지에서는 동적 그래프에 대한 컴파일러도 개발이 진행 중인 것으로 보이지만, pytorch에 대해서만 지원할 것으로 보이고 된다고 하더라도 동적 그래프는 하드웨어에 최적화시키기가 어려움.
- 현재는 배제하고 진행할 예정, 추후 연구주제로 채택 가능 (ex : 동적그래프 컴파일 방법에 대한 연구 등)- torch model이 llvm-cpu를 target으로 IREE가 적용되는 과정을 보여주면서, 새롭게 제작하는 컴파일러가 적용될 위치를 보여줌
# 1. 실험용 모델
- 단순한 MLP 1 Layer로, 아래와 같이 구성
```
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.fc(x))
```
- IREE에 넣기 위해서 IREE가 지원하는 포맷 중 하나인 StableHLO 포맷으로 컴파일
```
model = SimpleMLP().eval()
example_input = torch.randn(1, 128)

hlo_module = torch_mlir.compile(model, [example_input], output_type="stablehlo")
with open("mlp_hlo.mlir", "w") as f:
    f.write(str(hlo_module))
```
- IREE의 input으로 들어갈 mlir
![[Pasted image 20260116094333.png]]
# 2. IREE 정규식 표현으로 변경
- IREE의 Importer부분이 적용되면 mlir은 아래와 같이 변화 
- 공식 mlir이 제공하는 dialect들로 변경된 것을 확인할 수 있음.
- iree의 정규식으로 변경 됨
![[Pasted image 20260116094434.png]]

# 3. Stream / Flow / HAL 적용
- 연산의 순서를 정비하고, 최적화하는 패스들(Stream, Flow 계층)을 적용하고, HAL 패스를 적용하면 아래와 같이 변화
![[Pasted image 20260116094859.png]]
- hal.executalbe.variant가 생성되고, 그 아래 Region으로 builtin.module이 생성.
- builtin.module 안에는 아직 IREE의 정규식으로 표현된 IR
- **==해당 위치가 Plugin이 적용될 위치==**
- ==**builtin.module의 내부의 IR을 hal 표현식으로 바꿔가는 것이 컴파일러 개발의 최종 목표**==

# 4. Plugin 변환 적용
- Plugin이 모두 적용되면 아래와 같이 변화
- llvm-cpu의 경우 llvmIR로 변화시키고, 시스템에 저장된 바이너리를 호출하는 형식이지만, custom HW에 대한 Plugin의 기본 전략은 hal interface로 변경시키는 것
- **==Plugin의 결과물로 원하는 것은 본래의 연산을 hal interface로 표현하고, 메타데이터를 저장하는 것을 목표로 함==**
![[Pasted image 20260116100201.png]]

# 5. VM 생성
- builtin에 저장된 결과를 바탕으로 2가지를 생성.
- 그 중 하나는 vm의 생성
- builtin에 저장된 결과를 통해 runtime이 hal을 호출할 스캐줄을 구성해 줌.
![[Pasted image 20260116101013.png]]

# 6. IREE의 최종 형태
- IREE의 최종 형태는 모두 vm dialect로 표시되며, 3가지 부분으로 나뉨
## 6.1 메타데이터
- 어떤 메타데이터가 어디에 있는지, 또는 어떤 값인지를 기억
![[Pasted image 20260116101517.png]]

## 6.2 HAL Interface 정의
- 사용할 HAL Interface를 선언 및 정의함. 이는 runtime 구현의 **vtable**로 이어짐
![[Pasted image 20260116102611.png]]

## 6.3 실제 연산을 정의
- 위에 선언한 HAL Interface와 rodata, metadata를 조합하여 연산의 실행을 구현
![[Pasted image 20260116102729.png]]
- vm.export \<function name\>을 통해 function을 수행

## 6.4 기타 내용들
- 최종형태 코드를 보면 init이 뒤에 실행되는 것처럼 보이는데, runtime이 코드에서 init을 먼저 찾아서 수행해주는 것으로 보임
- init은 export된 이후에 정의가 나오는데, 이것이 표준이라고 함. 
![[Pasted image 20260116102925.png]]
# 7. Runtime으로 연결
- 최종 코드는 모두 VM으로 되어 있어, Runtime은 VM code를 읽고 거기에 맞는 행동을 수행한다. 
- 일반적인 기능들(vm.rodat, vm.const, ... 등)은 Runtime이 자원과 메모리를 관리하며 적절하게 호출하도록 IREE가 구축해 뒀다.
- vm.call의 경우에는 사용자가 정의하는 함수들이 호출이 되는데, 기본적으로 HAL command들을 호출하게 된다. 

## 7.1 vm.call HAL
- Runtime이 vm.call 명령어를 실행할 순간이 되면, HAL Interface에 따라runtime의 vtable로 이동하고, vtable이 정의하는 함수가 실행되는 방식
- 기본적으로 사용할 수 있는 Interface를 IREE가 제공해주기 때문에 이 안에서 조합하는 것이 좋음
- 예시 (iree-amd-aie의 xrt 연결방식)

1) VM이 hal.device.queue.alloca 라는 인터페이스를 사용
![[Pasted image 20260116134733.png]]

2) hal.device.queue.alloca 선언
![[Pasted image 20260116134851.png]]

3) xrt의 vtable에서 인터페이스와 백엔드 함수 연결
![[Pasted image 20260116134948.png]]

4) 연결된 함수 실행 : 해당함수에 queue alloca에 대한 행동이 구현되어 있음. **==device drvier나 시뮬레이션의 alloca에 해당하는 함수의 호출도 이 함수로 연결됨==**
![[Pasted image 20260116135018.png]]# 1. Dialect 하위 구성요소

## 1.1 operation
- Dialect를 구성하는 연산들, 일반 기계어의 opcode와 동일한 역할을 수행
- Dialect를 정의하는 것은 Dialect를 구성하는 operation들을 정의하는 것과 같다고 할 수 있음
### 1.1.1 구성요소
- Mnemonic : Dialect 내에서 Op를 식별하는 고유 이름 (ex : myIR.dma_copy)
- Operands : 일반 기계어의 operand와 같은 개념. 연산의 입력값. (ex : %input, %weight)
- Results : 연산의 결과값. (ex : %output)
- Regions : Op 내부에 포함된 하위 코드 블록 (ex : 반복문의 경우, scf.for 내부의 연산들)
- [Attribute](#Attribute) : 속성, 컴파일타임에 결정되는 상수 값으로 각 연산의 설정값과 같은 역할
- [Interface](#Interface) : Operation이 할 수 있는 행동을 나타냄
- Builders : rewriter.create\<MyOp\>(...)를 호출할 때, arguments와 attributes 등을 어떻게 초기화할 지 정의. 
  - 구현 예시
```
let builders = [
  OpBuilder<(ins "Value":$src, "Value":$dst), [{
    build($_builder, $_state, src, dst, $_builder.getI64IntegerAttr(0));
  }]>
];
```
- Verifiers : 연산이 올바른지 검사하는 규칙 유무, \[let hasVerifiler = 1;\]로 설정하고, 검증하는 함수 LogicalResult verify()를 구현해야 함.

### 1.1.2 Operation 정의 예시
- tablegen file에서 구현하는 것이 편리
```
def HW1IR_DmaCopyOp : HW1IR_Op<"dma_copy", [
  Pure,                        // Trait: 부수효과 없음
  hw1ir_MemcpyInterface        // Interface: 복사 인터페이스 구현
]> {
  let summary = "메모리 간 데이터 복사 연산";

  let arguments = (ins
    AnyMemRef:$src,            // 입력 메모리
    AnyMemRef:$dst,            // 출력 메모리
    I64Attr:$size              // 복사할 크기 (상수)
  );

  let results = (outs
    HW1IR_AsyncToken:$token    // 비동기 실행을 위한 토큰
  );

  // 인터페이스 메서드 구현
  let extraClassDeclaration = [{
    Value getSrcMemref() { return getSrc(); }
    Value getDstMemref() { return getDst(); }
  }];
}

** Pure : 해당 연산이 메모리에 대해 Read, Write, Alloc, Free 중 어떤 효과도 발생시키지 않음을 나타내는 속성
```
## 1.2 Attribute
- 연산에 부착되는 메타데이터, 런타임 중에 값이 변하지 않음
- 메모리 정렬정보나 디바이스정보, 타일링크기 등 다양한 상수를 저장 가능
### 1.2.1 속성 정의
- 기본적으로 정의된 속성 : I32Attr, I64Attr 등
- Enum 속성 정의
```
// 1. 케이스 정의
def HW1IR_DMA_READ  : I32EnumAttrCase<"Read", 0>;
def HW1IR_DMA_WRITE : I32EnumAttrCase<"Write", 1>;

// 2. Enum 속성 본체 정의
def HW1IR_DmaModeAttr : I32EnumAttr<"DmaMode", "DMA 전송 모드",
  [HW1IR_DMA_READ, HW1IR_DMA_WRITE]> {
  let cppNamespace = "::keti::hw1ir";
}
```
- 커스텀 속성 정의
```
// 1. 다이얼렉트 내에 AttrDef 정의
def HW1IR_MemoryConfigAttr : AttrDef<HW1IR_Dialect, "MemoryConfig"> {
  // 2. IR에서 불릴 이름 (예: #hw1ir.mem_config<...>)
  let mnemonic = "mem_config";

  // 3. 속성이 가질 파라미터(구성 요소) 정의
  let parameters = (ins
    "int64_t":$tile_size,
    "bool":$is_cached
  );

  // 4. IR에서 어떻게 보일지 형식 지정
  let assemblyFormat = "`<` $tile_size `,` $is_cached `>`";
}
```

### 1.2.2 활용
- 주로 operation 정의에서 데이터의 속성으로 사용함
```
def HW1IR_LoadOp : HW1IR_Op<"load"> {
  let arguments = (ins
    AnyMemRef:$input,
    HW1IR_MemoryConfigAttr:$config // 위에서 정의한 커스텀 속성
  );
}
```

## 1.3 Interface
- operation의 정체를 몰라도 특정 행동을 수행할 수 있도록 약속한 규격
- 특정 행동을 수행하는 operation들을 분별하여 변환을 정의하거나 공통된 메소드를 호출하는 형식으로 활용 가능

### 1.3.1 Interface 정의
```
def MyIR_MemcpyInterface : OpInterface<"MemcpyInterface"> {
  let description = "메모리 복사 기능을 가진 연산들을 위한 인터페이스";
  let cppNamespace = "::publisher::myir";

  let methods = [
    // 메서드 정의
    InterfaceMethod<
      "설명", 
      "반환타입", "메서드이름", 
      (ins "인자타입":$인자이름),
      [{}], // Interface 클래스 내 본문 (보통 비움)
      [{}]  // 기본 구현 (Default Implementation, 선택사항)
    >
  ];
}
```
- 해당 Interface를 Operation에 인용하면, Interface가 제공하는 함수를 반드시 제공해야 함. 기본적으로 Interface에 정의가 되어 있으나, 없는 경우에는 Operation 정의 또는 Builder에 함수들을 정의해 둬야 함
```
def MyIR_PowerAnalysisInterface : OpInterface<"PowerAnalysisInterface"> {
  let description = [{
    가속기 내의 각 연산이 소비하는 예상 전력량을 
    계산하기 위한 인터페이스입니다.
  }];

  let cppNamespace = "::publisher::myir";

  let methods = [
    // 1. 기본 구현이 없는 메서드 (Op에서 반드시 로직을 짜야 함)
    InterfaceMethod<"연산의 기본 소비 전력을 반환 (Watt 단위)",
      "float", "getBasePower">,

    // 2. 기본 구현이 있는 메서드 (따로 안 짜면 이 로직을 사용)
    InterfaceMethod<"데이터 크기에 따른 가중 전력 계산",
      "float", "getTotalPower", (ins "float":$dataScale), [{}], [{
        // $_op를 통해 현재 연산의 메서드에 접근 가능
        return $_op.getBasePower() * dataScale;
      }]
    >
  ];
}
```

### 1.3.2 활용
- operation에 인용
```
def MyIR_DmaOp : MyIR_Op<"dma", [MyIR_MemcpyInterface]> {
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst, I64Attr:$size);

  // 인터페이스 메서드 실제 로직 연결
  let extraClassDeclaration = [{
    ::mlir::Value getSrc() { return getOperand(0); }
  }];
}
```

- 변환에 활용
```
void myOptimizationPass(Operation *op) {
    // 1. 해당 연산이 인터페이스를 가지고 있는지 확인 (Cast)
    if (auto memcpyOp = llvm::dyn_cast<publisher::myir::MemcpyInterface>(op)) {
        
        // 2. Op의 실제 종류(DmaOp인지 다른 것인지) 몰라도 메서드 호출 가능
        mlir::Value source = memcpyOp.getSrc();
        
        // 3. 공통 로직 수행
        optimizeTransfer(source);
    }
}
```

# 2. Pass 하위 구성요소

## 2.1 구성요소
- RewritePattern : 특정 Op를 찾아 새로운 Op로 교체하는 최소 실행단위 (일반적으로 MatchAndRewrite로 진행)
- RewritePatternSet : Pass에서 사용할 여러 Pattern들을 담는 객체
- PatternRewriter : IR을 안전하게 수정해주는 도구 (Op 생성, 삭제, 교체 등을 담당)
- GreedyPatternRwriteDriver : 등록된 Pattern들을 반복적으로 적용하여 해당 Op가 없을때까지 변환을 수행하는 엔진

## 2.2 Pass 구현

### 2.2.1 RewritePattern 정의
- 특정 Op를 중심으로 mlir코드 내에 location을 기점으로 pattern을 분석 및 match한 뒤, operation을 생성하거나 삭제 또는 교체하는 식으로 IR을 변화시킴
- class로써 정의 (structure도 괜찮은 것으로 아는데, 실험해보지는 않음)
- 예시
```
class LinalgMatmulToHW1Pattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                  PatternRewriter &rewriter) const override {

        Value inputA = matmulOp.getInputs()[0];
        Value inputB = matmulOp.getInputs()[1];
        Value outputC = matmulOp.getOutputs()[0];
        
        rewriter.replaceOpWithNewOp<keti::hw1ir::MatmulOp>(
            matmulOp,
            matmulOp.getResult(0).getType(),
            inputA,
            inputB
        );
  
        return success();
    }
}
```

### 2.2.2 Pass 생성 및 Pattern 등록
- Pass 생성 - tablegen  : 어떤 Pass들을 생성할 지 tablegen으로 기록하면 컴파일러 빌드 시, pass creator와 pass register를 생성해줌
```
[Passes.td]

#ifndef HW1IR_CONVERSION_PASSES
#define HW1IR_CONVERSION_PASSES
  
include "mlir/Pass/PassBase.td"
  
def LinalgToHW1 : Pass<"linalg-to-hw1", "ModuleOp">{
    let summary = "Convert Linalg operations to HW1IR dialect";
    let constructor = "keti::hw1ir::createLinalgToHW1Pass()";
    let description = [{
        This pass converts Linalg operations to HW1IR dialect operations.
    }];
}
```
- Pass 생성 - Pass 헤더 : create를 포함시켜 줌. 이름 포맷은 Passes.td에 def로 정의한 이름과 맞춰야 함.
```
[ConvertToHW1IRPass.h]

#ifndef CONVERT_TO_HW1IR_H
#define CONVERT_TO_HW1IR_H

#include "HW1/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>
  
namespace keti {
namespace hw1ir {

std::unique_ptr<mlir::Pass> createLinalgToHW1Pass();
    
} // namespace hw1ir
} // namespace keti
  
#endif // CONVERT_TO_HW1IR_H
```
- Pass 생성 - RewritePattern : class로써 Pattern을 정의, class를 만들 때 사용한 argument Op를 통해 argument Op에 해당하는 모든 location을 가져옴. matchAndRewrite에서 패턴과 변환규칙을 정의
```
[ConvertToHW1IRPass.cpp]

class LinalgMatmulToHW1Pattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  
    LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                  PatternRewriter &rewriter) const override {
  
        Value inputA = matmulOp.getInputs()[0];
        Value inputB = matmulOp.getInputs()[1];
        Value outputC = matmulOp.getOutputs()[0];
        rewriter.replaceOpWithNewOp<keti::hw1ir::MatmulOp>(
            matmulOp,
            matmulOp.getResult(0).getType(),
            inputA,
            inputB
        );

        return success();
    }
}
```
- Pass 생성 - Pass 구조체 : 이름은 Passes.td에 정의한 이름과 맞춰서 작성, runOnOperation에서 패턴들을 등록, 이외에도 dialect나 operations에 대해 legal, illegal을 선언하거나 메소드 선언 등을 수행할 수도 있음.
```
[ConvertToHW1IRPass.cpp]

struct LinalgToHW1Pass
    : public keti::hw1ir::impl::LinalgToHW1Base<LinalgToHW1Pass> {
    LinalgToHW1Pass() = default;
    LinalgToHW1Pass(const LinalgToHW1Pass &pass) {}

    void runOnOperation() override;
}

...

void LinalgToHW1Pass::runOnOperation() {
    auto module = getOperation();
    auto context = module.getContext();
  
    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    RewritePatternSet patterns(context);
    // Add patterns to convert Linalg ops to HW1IR ops here
    patterns.add<LinalgMatmulToHW1Pattern>(context);
  
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
    }
  
    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));
}

```
- 이렇게 생성된 Pass들은 Passes.h와 Passes.cpp에서 등록
```
[Passes.h]

#ifndef HW1IR_CONVERSION_PASSES_H
#define HW1IR_CONVERSION_PASSES_H

#include "HW1/Conversion/ConvertToHW1IRPass.h"

namespace keti {
namespace hw1ir {
  
void registerConversionPasses();
  
} // namespace hw1ir
} // namespace keti
  
#endif // HW1IR_CONVERSION_PASSES_H

----------------------------------------------------------
[Passes.cpp]

#include "HW1/Conversion/Passes.h"
  
namespace {
#define GEN_PASS_REGISTRATION
#include "HW1/Conversion/Passes.h.inc"
}
void keti::hw1ir::registerConversionPasses() { ::registerPasses(); }

```- OpFusionPass의 뒷처리를 한다. : 앞선 규칙을 모두 적용하고 남은 1:1 op를 변환한다. 예시로, 일반 mul이나 add가 있다. 

# 1. Patterns


## 1.1 IREEAddConversionPattern
- 남아있는 Add를 CAP Layer의 Add로 변환한다.
- 가장 마지막에 남아있는 줄을 변환하므로 benefit은 1을 준다.
- benefit=1
- 예시
```
[before]
	%0 = arith.addi %arg0, %arg1 : i32
	%1 = arith.addf %arg2, %arg3 : f32
	
[after]
	%0 = "inf_cap.add"(%arg0, %arg1) : (i32, i32) -> i32 
	%1 = "inf_cap.add"(%arg2, %arg3) : (f32, f32) -> f32
```

## 1.2 IREEMulConversionPattern
- 남아있는 Mul를 CAP Layer의 Mul로 변환한다.
- 가장 마지막에 남아있는 줄을 변환하므로 benefit은 1을 준다.
- benefit=1
- 예시
```
[before]
	%0 = arith.muli %arg0, %arg1 : i32
	%1 = arith.mulf %arg2, %arg3 : f32
	
[after]
	%0 = "inf_cap.mul"(%arg0, %arg1) : (i32, i32) -> i32 
	%1 = "inf_cap.mul"(%arg2, %arg3) : (f32, f32) -> f32
```
- 각종 연산은 arith와 math, linalg등의 Dialect를 이용한 IREE의 정규식으로 식이 펼쳐져 있다. 이를 다시 묶어낸다.

# 1. Patterns

같은 정규식이어도, 최적화 수준에 따라 다르게 표현되어 있으므로 각각의 모든 경우에 대해 변환패턴을 등록한다.

## 1.1 Conv

### 1.1.1 IREEToUnsymetricQConvPattern
- generic을 잡고, yield부터 back tracking
- IREEToSymetricQConvPattern과 결합 = IREEToQConvPattern
- benefit = 10

```
[변환 전]
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 * 2 + d4, d2 * 2 + d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%inserted_slice, %cst_0 : tensor<3x642x642xi8>, tensor<16x3x3x3xi8>) outs(%10 : tensor<16x320x320xi32>) {
    ^bb0(%in: i8, %in_10: i8, %out: i32):
      %14 = arith.extsi %in : i8 to i32
      %15 = arith.subi %14, %c-128_i32 : i32
      %16 = arith.extsi %in_10 : i8 to i32
      %17 = arith.muli %15, %16 : i32
      %18 = arith.addi %out, %17 : i32
      linalg.yield %18 : i32
    } -> tensor<16x320x320xi32>
```
```
[변환 후]
%12 = inf_cap.conv2d %inserted_slice, %cst_0 { dilation_h = 1 : i64, dilation_w = 1 : i64, stride_h = 2 : i64, // indexing_map의 'd1 * 2'에서 추출 stride_w = 2 : i64, // indexing_map의 'd2 * 2'에서 추출 pad_h = 0 : i64, pad_w = 0 : i64, quant_info = #inf_cap.quant<scale = 1.0, zero_point = -128>, // subi에서 추출 layout_hint = #inf_cap.layout<NCHW> // 필요시 추가 } : (tensor<3x642x642xi8>, tensor<16x3x3x3xi8>) -> tensor<16x320x320xi32>
```

### 1.1.2 IREEToSymetricQConvPattern
- generic을 잡고, yield부터 back tracking
- IREEToUnsymetricQConvPattern과 결합 = IREEToQConvPattern
- benefit = 10
```
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 * 2 + d4, d2 * 2 + d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%inserted_slice, %cst_0 : tensor<3x642x642xi8>, tensor<16x3x3x3xi8>) outs(%10 : tensor<16x320x320xi32>) {
    ^bb0(%in: i8, %in_10: i8, %out: i32):
      %14 = arith.extsi %in : i8 to i32
      %15 = arith.extsi %in_10 : i8 to i32
      %16 = arith.muli %14, %15 : i32
      %17 = arith.addi %out, %16 : i32
      linalg.yield %17 : i32
    } -> tensor<16x320x320xi32>
```

## 1.2 Sigmoid

### 1.2.1 IREEToSigmoidPattern
- benefit = 10
- 아래 flow를 inf_cap.sigmoid로 고친다.
```
[변환 전]
      %27 = arith.negf %26 : f32
      %28 = math.exp %27 : f32
      %29 = arith.addf %28, %cst_6 : f32
      %30 = arith.divf %cst_6, %29 : f32
```
```
[변환 후]
      %30 = inf_cap.sigmoid %29 : f32
```



## 1.3 SiLU
- SiLU는 여기서 구현하지 않는다. IREEToCapPass에서 Sigmoid와 Mul을 찾아 변환한다.


## 1.4 IREEToMulPattern# 1. 모델 개요

- yolo를 타겟하기 전, yolo를 가장 많이 구축하고 있는 Conv-Activation을 타겟으로 변환을 시도해 봄.
- onnx에서 Conv-Activation은 Conv-Sigmoid-Mul로 나타남

# 2. 추가로 입력하는 옵션

- **--iree-global-opt-experimental-disable-conv-generalization** : convolution이 루프로 풀리는 것을 방지

# 3. Ops List - 1차

## 3.1 **arith.constant** 
- c++객체 : arith::ConstantOp
- ins : -
- outs : -
- 설명 : 상수 메타데이터
## 3.2 linalg.fill
- c++객체 : linalg::FillOp
- ins
	- value : 채울 값
	- buffer : 초기화 할 버퍼
- outs : 
- 설명 : 버퍼를 특정 값으로 초기화 (c언어 memset과 동일)

## 3.3 linalg.conv_2d_nchw_fchw
- c++객체 : linalg::Conv2DNchwFchwOp
- ins
	- input : NCHW Layout
	- Kernel : FCHW Layout
- outs : 
	- resutl_tensors : tensor
- attributes
	- strides - ::mlir::DenseIntElementsAttr  :  64bit unsigned int
	- dilations - ::mlir::DenseIntElementsAttr  :  64bit unsigned int
- 설명 : input은 NCHW, Kernel은 FCHW인 conv2d연산

## 3.4 linalg.generic
- c++객체 : linalg::GenericOp
- ins : load할 메모리
- outs : store할 메모리
- attributes
	- indexing_maps : load / store 할 메모리 접근순서 및 방식
	- iterator_types : 연산 방식 (parallel 등)
- 설명 : linalg.generic 하위의 데이터 접근 방법을 표시하여 접근한 데이터로 basic block에 정의된 연산을 수행

## 3.5 arith.addf
- c++객체 : arith::AddFOp
- ins
	- lhs : float 64, 32, bf16, vector, tensor
	- rhs : float 64, 32, bf16, vector, tensor
- outs
	- result : float
- 설명 : 두 float의 덧셈

## 3.6 arith.negf
- c++객체 : arith::NegFOp
- ins
	- in : f64, 32, 8, vector, tensor
- outs
	- result : float
- 설명 : 입력값의 negation을 반환 (동일 타입으로)


## 3.7 math.exp
- c++객체 : math::ExpOp
- ins
	- in : float like
- outs
	- result : float like
- attributes
	- fastmath  :  ::mlir::arith::FastMathFlagsAttr, fast math 플래그
- 설명 : base e의 exponential을 계산


## 3.8 arith.divf
- c++객체 : arith::DivFOp
- ins
	- lhs : float
	- rhs : float
- outs
	- result : float
- attributes
	- fastmath  :  ::mlir::arith::FastMathFlagsAttr, fast math 플래그
- 설명 : float인 lhs를 rhs로 나눈 값

## 3.9 arith.mulf
- c++객체 : arith::MulFOp
- ins
	- lhs : float
	- rhs : float
- outs
	- result : float
- 설명 : float인 lhs와 rhs의 곱

## 3.10 linalg.yield
- c++객체 : linalg::YieldOp
- ins : any value
- 설명 : basic block과 같은 지역의 반환값을 설정







## 3.10 arith.addf
- c++객체 : 
- ins
- outs
- 설명 : 


- IREE는 모듈화된 구조를 최대한 활용하게 하기 위해 plugin 기능을 제공
- 해당 기능을 활용함으로써 IREE는 그대로 유지하면서 새로운 backend를 추가하거나 새로운 최적화 pass를 추가해볼 수 있다.

# 1. 사용방법

- AMD AIR-AIE 사용 예를 참고
- IREE의 PluginSession class를 상속받고, 그 안에 Pass, Dialect, Device 정보(HAL 등) 등을 입력하고, compiler_plugin 함수를 mangling하여 내보냄
- Plugin에 등록하는 Pass, Dialect, Device 관련 라이브러리들은 IREE와는 독립적으로 개발
```
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
#include "aie/AIEDialect.h"
#include "aie/AIEXDialect.h"
#include "aie/Passes.h"
#include "aievec/AIEVecDialect.h"
#include "aievec/Passes.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Passes.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Target/AIETarget.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
  
namespace mlir::iree_compiler {
namespace {
  
namespace {
#define GEN_PASS_REGISTRATION
#include "aie/Passes.h.inc"
}  // namespace
  
struct AMDAIESession
    : public PluginSession<AMDAIESession, AMDAIE::AMDAIEOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    AMDAIE::registerAMDAIEPasses();
    registerAMDAIEAssignBufferAddresses();
    AMDAIE::registerAMDAIEAssignBufferDescriptorIDs();
    registerAMDAIECoreToStandard();
    AMDAIE::registerAMDAIELocalizeLocks();
    AMDAIE::registerAMDAIENormalizeAddressSpaces();
    registerAMDAIERouteFlowsWithPathfinder();
    AMDAIE::registerAMDAIEDmaToNpu();
    AMDAIE::registerAMDAIEIncrementRepeatCount();
    AMDAIE::registerAIRConversionPasses();
    AMDAIE::registerAIRTransformPasses();
    aievec::registerConvertAIEVecToLLVMPass();
    aievec::registerAlignTransferReadsPass();
    aievec::registerCanonicalizeVectorForAIEVecPass();
    aievec::registerLowerVectorToAIEVecPass();
  }
  
  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<AMDAIE::AMDAIEDialect, xilinx::AIE::AIEDialect,
                    aievec::AIEVecDialect, xilinx::AIEX::AIEXDialect,
                    xilinx::air::airDialect>();
  }
  
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    // #hal.device.target<"xrt", ...
    targets.add("xrt", [=] {
      options.deviceHal = AMDAIE::AMDAIEOptions::DeviceHAL::XRT;
      return AMDAIE::createTarget(options);
    });
    // #hal.device.target<"xrt-lite", ...
    targets.add("xrt-lite", [=] {
      options.deviceHal = AMDAIE::AMDAIEOptions::DeviceHAL::XRT_LITE;
      return AMDAIE::createTarget(options);
    });
  }
  
  void populateHALTargetBackends(
      IREE::HAL::TargetBackendList &targets) override {
    targets.add("amd-aie", [=]() { return AMDAIE::createBackend(options); });
  }
};
  
}  // namespace
}  // namespace mlir::iree_compiler
  
IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::AMDAIE::AMDAIEOptions);
  
extern "C" bool iree_register_compiler_plugin_amd_aie(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::AMDAIESession>("amd_aie");
  return true;
}
```

- iree-amd-aie 프로젝트 구조 참고
       ![[iree-air-aie-architecture 2.png|600]]


- plugin 적용 위치
![[Apply_Plugin.png]]

# 2. Plugin 동작의 이해

- [[컴파일러 적용과 mlir의 변화]]에 Plugin의 동작을 포함하여 컴파일러가 적용되는 흐름을 정리- OpFusionPass의 뒷처리를 한다. : conv를 묶고 남은 bias나, requantize가 있다. 

# 1. Patterns



## 1.1 Post - Conv

- Conv와 Bias, Requantize는 다른 generic으로 분리되어 있음. 
- Conv와 Bias, Requantize는 따로 관리하기 위해 별도의 operation으로 관리한다.
### 1.1.1 FindConvBiasPattern
- addi를 만나면 Bias로 바꾼다. addi의 경우 Bias가 아닌 add일 수도 있으므로 엄격한 규칙을 적용한다.
- Bias가 아닌 addi 보다 우선순위를 높히기 위해 benefit을 활용한다.
- 규칙 1 : generic의 첫 operation으로 존재하는 addi 에 대해서만 규칙을 적용한다.
- 규칙 2 : addi가 첫번째로 존재하는 generic의 definingOp를 찾아 Conv인 경우만 규칙을 적용한다.
- 규칙 3 : addi의 basic block 및 generic의 affine map을 조합하여 그 shape이 Conv의 result와 일치하는 경우에만 규칙을 적용한다.
- Benefit=5
- 예시
```
=============================================================//
FindConvBias Pattern
=============================================================//
This pattern fuses the 'arith.addi' operation following a convolution
into a single hardware-specific bias operation.
It validates the bias broadcasting pattern via Affine Maps to ensure
the addition occurs across the channel dimension.

[before]
%0 = "inf_cap.conv2d"(%input, %weight) : ... -> tensor<16x160x160xi32
%1 = linalg.generic 
  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, // conv result
    affine_map<(d0, d1, d2) -> (d0)>,         // bias (1D)
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>] // output
      } ins(%0, %bias : ...) {
        ^bb0(%in: i32, %in_bias: i32, %out: i8):
        %2 = arith.addi %in, %in_bias : i32  <-- Target for Fusion
        ...
}
  
[after]
        ^bb0(%in: i32, %in_bias: i32, %out: i8):
        %2 = "inf_cap.bias"(%in, %in_bias) : (i32, i32) -> i32
        ...
```

### 1.1.2 FindRequantPattern
- conv - bias 이후에 오는 requant를 변환한다. 이 requant를 찾아 inf_cap.requant로 바꾼다. 
- 이 패턴은 zeropoint가 0인 패턴으로, 최적화에 의해 사라진 버전이다.
- 규칙 1 : bias 이후에 오는 값이어야 한다.
- 규칙 2 : arith.sitofp -> arith.mulf로 이어지는 패턴을 찾아 requant로 변환한다.
- benefit=5
- 예시
```
[before]
          ^bb0(%in: i32, %in_bias: i32, %out: f32):
            %0 = "inf_cap.conv_bias"(%in, %in_bias) : (i32, i32) -> i32
            %1 = arith.sitofp %0 : i32 to f32
            %2 = arith.mulf %1, %scale : f32

[after]
          ^bb0(%in: i32, %in_bias: i32, %out: f32):
            %0 = "inf_cap.conv_bias"(%in, %in_bias) : (i32, i32) -> i32
            %1 = "inf_cap.requant"(%0, %scale) : (i32, f32) -> f32
```

### 1.1.3 FindRequant2Pattern
- conv-bias 이후에 오는 requant를 변환한다.
- 이 패턴은 zeropoint가 0이 아닌 패턴이다. (TODO : 아직 실제로 requant의 zeropoint가 0이 아닌 버전을 보지 못해서 검증을 하지 못했다. 해당 패턴을 발견하고 실험하여 적용되는지 검증할 필요가 있다.)
- 규칙 1 : bias 이후에 오는 값이어야 한다.
- 규칙 2 : 아래 패턴을 매칭한다.
- benefit=4
- 예시
```
[before]

          %0 = "inf_cap.conv_bias"(%in, %in_bias) : (i32, i32) -> i32
          %1 = arith.sitofp %0 : i32 to f32
          %2 = arith.mulf %1, %scale : f32
          %3 = arith.addf %2, %zp_f32 : f32 

[after]
          %4 = "inf_cap.requant"(%0, %scale, %zp_i32) : (i32, f32, i32) -> i8
```


- Quantize와 Dequantize는 arith와 math Dialect를 이용한 IREE의 정규식으로 식이 펼쳐져 있다. 이를 QuantOp와 DequantOp로 다시 묶어낸다.

# 1. Patterns

같은 정규식이어도, 최적화 수준에 따라 다르게 표현되어 있으므로 각각의 모든 경우에 대해 변환패턴을 등록한다.

## 1.1 Quant1Pattern
- back tracking 방식으로 찾는다.
- 안전장치로, 각 Op마다 user가 1개인지 검사한다.
- 아래 식을 QuantOp로 변환한다.
```
[변경 전]
      %1 = arith.divf %in, %scale : f32
      %2 = math.roundeven %1 : f32
      %3 = arith.addf %2, %zeropoint : f32
      %4 = arith.maximumf %3, %min : f32
      %5 = arith.minimumf %4, %max : f32
      %6 = arith.fptoui %5 : f32 to i8
```
```
[변경 후]
      %6 = inf_cap.quantize %in : f32 to i8
```

## 1.2 Dequant1Pattern
- back tracking 방식으로 찾는다.
- 안전장치로, 각 Op마다 user가 1개인지 검사한다.
- 아래 식을 DequantOp로 변환한다.
```
[변경 전]
      %1 = arith.extui %in : i8 to i32
      %2 = arith.subi %1, %zeropoint : i32
      %3 = arith.sitofp %2 : i32 to f32
      %4 = arith.mulf %3, %scale : f32
```
```
[변경 후]
      %4 = inf_cap.dequantize %in : f32 to i8
```
## 1.3 Dequant2Pattern
- back tracking 방식으로 찾는다.
- 안전장치로, 각 Op마다 user가 1개인지 검사한다.
- zeropoint가 0인 경우, 최적화가 적용되어 아래와 같이 나타날 수 있다.
- 아래 식을 DequantOp로 변환한다.
```
[변경 전]
      %1 = arith.extui %in : i8 to i32
      %2 = arith.sitofp %1 : i32 to f32
      %3 = arith.mulf %2, %scale : f32
```
```
[변경 후]
      %3 = inf_cap.dequantize %in : f32 to i8
```

## 1.4 AddZeropointFusionPattern
- quantize 다음에 add가 오는 경우, zeropoint에 반영하여 하나의 quantize로 바꾼다.
```
[before]
      %12 = "inf_cap.quantize"(%in) <{quant_info = #inf_cap.quant<0.0039215646699999997, 0>}> : (f32) -> i8
      %13 = arith.addi %12, %c-128_i8 : i8

[after]
%12 = "inf_cap.quantize"(%in) <{quant_info = #inf_cap.quant<0.0039215646699999997, %c-128_i8>}> : (f32) -> i8
```
## 1.5 SubZeropointFusionPattern
- quantize 다음에 sub가 온느 경우, zeropoint에 반영하여 하나의 quantize로 바꾼다. - quantize 모델을 사용하는 경우, QLinear로 시작하는 quantized operation을 IREE는 지원하지 않는다. 
- quantize 모델을 사용하기 위해 qdq로 quantize를 실행하고, 컴파일러가 q - dq와 operation을 합성해, quantized operation으로 만들어 낸다. 
- 1. 먼저 dequantize를 찾고 backward tracking으로 연결된 operation을 찾아낸다. 
- 2. 그 앞에 붙은 quantize들을 모아서 새로운 arguments들을 만든다. 
- 3. 이 arguments를 넣는 quantized operation을 만들어 낸다.
- 요약 : dequant + operation = quantized operation // quant + arguments = quantized arguments

# 1. Patterns

## 1.1 QdqToQopPattern
- **개요** : Q-Dq 사이의 Op를 QuantOp로 변경한다. (Op이름은 그대로이고, Attribute를 추가하는 것이 대부분)
- **Trigger : QuantOp**
- 1. Quant에서부터 back tracking방식으로 Op를 발견하고 QuantOp를 생성, Op의 설정들을 QuantOp로 옮긴다.
- 2. Quant의 output을 QuantOp의 output으로 연결
- 3. Back tracking으로 찾은 Dequant의 input을 QuantOp로 연결
- 4. Op 삭제
- 5. 이렇게 하면 Dequant와 Quant가 그래프에서 분리되기 때문에 Canonicalization에서 제거가 될 것이라고 보지만(DCE), 확인 결과 지워지지 않으면 지우는 패턴을 만들어야 함.
  (여기서 지우는 것은 안된다. Dequant의 output이 다른 노드에서도 활용될 수 있기 때문에, 다른 패턴을 만들어 지워야 한다.)


![[QdqToQop.png]]

### 1.1.1 일반 Ops
- 일반적인 Ops들은 Attributes로 quantization 변수를 넣을 수 있게 해뒀기 때문에 새로운 QuantOp를 만들지 않고, Attributes를 추가하고, input과 output을 변경해주면 된다. 
- 이 경우에 해당하는 Ops들에 대해 하나의 Pattern으로 구현할 수 있다. 

#### 1.1.1.1 FindQSigmoidPattern
- dequant - sigmoid - quant를 찾아, sigmoid를 qsigmoid로 바꾼다. 
- 이 때, dequant와 quant는 다른 곳에서 사용할 수도 있으므로 제거하지 않고, input과 output만 옮겨준다.
- mlir의 canonicalizer를 통해 결과값이 사용되지 않는 ops는 자동으로 삭제될 것.
- 1. backward tracking으로 dequant - sigmoid - quant 패턴을 찾는다.
- 2. sigmoid를 qsigmoid로 바꾼다
- 3. qsigmoid의 input을 dequant의 input으로 바꾸고, output을 quant의 output으로 바꾼다.
- 예시
```
[Before]
%i8_in = ... : i8
%f32_1 = "inf_cap.dequantize"(%i8_in) <{quant_info = #in_q}> : (i8) -> f32
%f32_2 = "inf_cap.sigmoid"(%f32_1) : (f32) -> f32
%i8_out = "inf_cap.quantize"(%f32_2) <{quant_info = #out_q}> : (f32) -> i8

[After]
%i8_in = ... : i8
%i8_out = "inf_cap.qsigmoid"(%i8_in) <{in_quant = #in_q, out_quant = #out_q}> : (i8) -> i8
```


#### 1.1.1.2 FindQMulPattern
- dequant - mul - quant를 찾아 mul을 qmul로 바꾼다.
- sigmoid와 비슷한 규칙적용
- 예시
```
[before]
 %lhs_f = "inf_cap.dequantize"(%lhs_i8) <{quant_info = #lhs_q}> : (i8) -> f32
 %rhs_f = "inf_cap.dequantize"(%rhs_i8) <{quant_info = #rhs_q}> : (i8) -> f32
 %mul_f = "inf_cap.mul"(%lhs_f, %rhs_f) : (f32, f32) -> f32
 %q_out = "inf_cap.quantize"(%mul_f) <{quant_info = #out_q}> : (f32) -> i8

[after]
 %q_out = "inf_cap.qmul"(%lhs_i8, %rhs_i8) <{
	 lhs_quant = #lhs_q,
	 rhs_quant = #rhs_q,
	 out_quant = #out_q
 }> : (i8, i8) -> i8
```


#### 1.1.1.3 FindQSiLUPattern
- conv에 연결된 qsigmoid와 qmul을 찾으면 qsilu로 변환한다.
- 현재는 conv가 conv2d - bias - requant - quantize로 나타나도록 했으므로 
- 예시
```
[Before]
%x = ... // Any source (quantize, requant, etc.)
%qsig = "inf_cap.qsigmoid"(%x) <{in_q, sig_out_q}> : (...) -> f32
%qsilu = "inf_cap.qmul"(%x, %qsig) <{in_q, sig_out_q, final_out_q}> : (..., f32) -> i8

[After]
%x = ...
%qsilu = "inf_cap.qsilu"(%x) <{in_q = #in_q, out_q = #final_out_q}> : (...) -> i8
```
### 1.1.2 특별한 Ops
- 특별한 Ops의 경우에는 본래의 취지대로, 새로운 Ops를 만들고 input, output, attributes를 연결한 후, 기존 Ops를 삭제한다.
- 이 방식을 채택할 경우, Ops 1개당 1개의 Pattern을 추가해야 한다.
- 아래에는 해당하는 Ops에 대해 남긴다.
- List
	- TODO : Ops List- compile 수행을 session과 Invocation으로 나누어, session은 실행 환경을 관리 invocation은 해당 환경에서 실제 작업을 수행
- session과 Invocation 사이에서 wrap과 unwrap 동작을 통해 값이나 포인터를 동기화
# 1. Session

- 컴파일러에서
	- 컴파일 작업을 수행하기 위한 설정 및 환경을 관리
	- 컴파일 옵션을 포함하여 input / output 등을 관리하고 있음
- 런타임에서
	- 컴파일된 바이너리를 실행하는 시점에서, 메모리와 리소스 환경을 관리

# 2. Invocation
- 컴파일러에서
	- 컴파일 파이프라인을 실제로 수행, Invocation 한 번은 파이프라인 한 번 수행하는 것을 의미
	- 컴파일 옵션에 따라 실제로 IR 변환을 적용함
- 런타임에서
	- predict나 run_module / run_function 등 하나의 함수를 수행
- Google의 XLA(Accelerated Linear Algebra)에서 시작
- HLO(High-Level Optimizer/Operation)은 XLA의 컴파일러가 사용하는 IR(Intermediate Representation)
- 컴파일러가 모델을 이해하고 최적화하기 위한 중간단계
- 추후 PyTorch/XLA, TensorFlow/XLA 등 다양한 프레임워크별로 HLO를 개발하여 사용
- 다양한 프레임워크에 공통된 표현으로 사용할 수 있도록 정리하여 새로 나온 것이 StableHLO이며, 프레임워크와 컴파일러 사이를 표현하는 중간언어의 표준이 됨.
- StableHLO는 MLIR을 output으로 가짐
- 수학적으로 잘 정의된 텐서 연산 언어
- 반복문, 포인터 등이 제외된 텐서 연산으로만 구성되어 컴파일러가 이해하기 쉬운 구조로 되어 있음- Linear(Matmul)와 ReLU로 구성된 간단한 1 layer를 만들어 컴파일 해본다.
- torch mlir과 stableHLO로 변환시키고, 이를 iree 정규식으로 가져온 뒤 customIR까지 변환시켜본다. (torch mlir + stable HLO -> iree : 완료  |  iree -> customIR : X)
- 최신 버전의 torch mlir을 사용하는 경우, torch_mlir 내부의 함수를 제대로 가져오지 못하는 현상 -> 20231229 버전을 받아서 사용
```
pip install torch-mlir==20231229.1067 -f https://llvm.github.io/torch-mlir/package-index
```

- torch mlir의 경우에는 iree로 import 되었을 때 이미 hal dialect를 호출하여 메모리의 계획이 잡혀있지만, StabelHLO는 iree로 import 되었을 때 아직 실행계획이 잡혀있는 상태. -> StableHLO를 통하는 방식을 선호

## 1. 모델 정의

- Linear와 ReLU로 표현한 SimpleMLP 1layer
```
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.fc(x))
  
model = SimpleMLP().eval()
example_input = torch.randn(1, 128)
```

## 2. torch mlir로 컴파일
### 2.1 컴파일 방법
```
# 컴파일
module = torch_mlir.compile(
	model,
	[example_input],
	output_type=torch_mlir.OutputType.TORCH
)
# 파일 저장
with open("mlp_torch.mlir", "w") as f:
    f.write(str(module))
```

### 2.2 컴파일 결과
```
module attributes {torch.debug_module_name = "SimpleMLP"} {
  func.func @forward(%arg0: !torch.vtensor<[1,128],f32>) -> !torch.vtensor<[1,64],f32> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %0 = torch.vtensor.literal(dense<[0.0504985414, ... , -0.0699762106]> : tensor<64xf32>) : !torch.vtensor<[64],f32>
    %1 = torch.vtensor.literal(dense<"0xF85...13D"> : tensor<64x128xf32>) : !torch.vtensor<[64,128],f32>
    %2 = torch.aten.transpose.int %1, %int0, %int1 : !torch.vtensor<[64,128],f32>, !torch.int, !torch.int -> !torch.vtensor<[128,64],f32>
    %3 = torch.aten.mm %arg0, %2 : !torch.vtensor<[1,128],f32>, !torch.vtensor<[128,64],f32> -> !torch.vtensor<[1,64],f32>
    %4 = torch.aten.add.Tensor %3, %0, %float1.000000e00 : !torch.vtensor<[1,64],f32>, !torch.vtensor<[64],f32>, !torch.float -> !torch.vtensor<[1,64],f32>
    %5 = torch.aten.relu %4 : !torch.vtensor<[1,64],f32> -> !torch.vtensor<[1,64],f32>
    return %5 : !torch.vtensor<[1,64],f32>
      }
}
```


## 3. StableHLO로 컴파일
### 3.1 컴파일 방법
```
# 컴파일
hlo_module = torch_mlir.compile(model, [example_input], output_type="stablehlo")
# 파일 저장
with open("mlp_hlo.mlir", "w") as f:
    f.write(str(hlo_module))
```

### 3.2 컴파일 결과
```
module attributes {torch.debug_module_name = "SimpleMLP"} {
  func.func @forward(%arg0: tensor<1x128xf32>) -> tensor<1x64xf32> {
    %0 = stablehlo.constant dense<[0.0504985414, ... , -0.0699762106]> : tensor<64xf32>
	%1 = stablehlo.constant dense<"0xF85...13D"> : tensor<64x128xf32>
	%2 = chlo.constant dense<0.000000e+00> : tensor<1x64xf32>
    %3 = stablehlo.transpose %1, dims = [1, 0] : (tensor<64x128xf32>) -> tensor<128x64xf32>
    %4 = stablehlo.dot %arg0, %3 : (tensor<1x128xf32>, tensor<128x64xf32>) -> tensor<1x64xf32>
    %5 = chlo.broadcast_add %4, %0 : (tensor<1x64xf32>, tensor<64xf32>) -> tensor<1x64xf32>
    %6 = stablehlo.maximum %5, %2 : tensor<1x64xf32>
    return %6 : tensor<1x64xf32>
  }
}
    
```

## 4. torch to iree
```
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "SimpleMLP"} {
  util.func public @forward$async(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %cst = arith.constant dense<[0.0504985414, ... , -0.0699762106]> : tensor<64xf32>
    %cst_0 = arith.constant dense<"0xF85...13D">: tensor<64x128xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<1x128xf32>
    %1 = tensor.empty() : tensor<128x64xf32>
    %transposed = linalg.transpose ins(%cst_0 : tensor<64x128xf32>) outs(%1 : tensor<128x64xf32>) permutation = [1, 0]
    %2 = tensor.empty() : tensor<1x64xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %4 = linalg.matmul ins(%0, %transposed : tensor<1x128xf32>, tensor<128x64xf32>) outs(%3 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst : tensor<1x64xf32>, tensor<64xf32>) outs(%2 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.addf %in, %in_2 : f32
      linalg.yield %9 : f32
    } -> tensor<1x64xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x64xf32>) outs(%2 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.cmpf ugt, %in, %cst_1 : f32
      %10 = arith.select %9, %in, %cst_1 : f32
      linalg.yield %10 : f32
    } -> tensor<1x64xf32>
    %7 = hal.tensor.barrier join(%6 : tensor<1x64xf32>) => %arg2 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<1x64xf32> -> !hal.buffer_view
    util.return %8 : !hal.buffer_view
  }
  util.func public @forward(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1 = util.call @forward$async(%arg0, %0, %fence) : (!hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.return %1 : !hal.buffer_view
  }
}    
```

## 5. StableHLO to iree
```
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "SimpleMLP"} {
  func.func @forward(%arg0: tensor<1x128xf32>) -> tensor<1x64xf32> {
    %0 = shape.const_witness true
    %cst = arith.constant 0.000000e+00 : f32
	%cst_0 = arith.constant dense<[0.0504985414, ... , -0.0699762106]> : tensor<64xf32>
    %cst_1 = arith.constant dense<"0xF85...13D">: tensor<64x128xf32>
	%1 = tensor.empty() : tensor<128x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<64x128xf32>) outs(%1 : tensor<128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x64xf32>
    %3 = tensor.empty() : tensor<1x64xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %5 = linalg.matmul ins(%arg0, %2 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%4 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %6 = shape.assuming %0 -> (tensor<1x64xf32>) {
	%9 = tensor.empty() : tensor<1x64xf32>
	%10 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<64xf32>) outs(%9 : tensor<1x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x64xf32>
	%11 = tensor.empty() : tensor<1x64xf32>
	%12 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%5, %10 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%11 : tensor<1x64xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
	%13 = arith.addf %in, %in_2 : f32
        linalg.yield %13 : f32
      } -> tensor<1x64xf32>
      shape.assuming_yield %12 : tensor<1x64xf32>
    }
    %7 = tensor.empty() : tensor<1x64xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<1x64xf32>) outs(%7 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.maximumf %in, %cst : f32
      linalg.yield %9 : f32
    } -> tensor<1x64xf32>
    return %8 : tensor<1x64xf32>
  }
}
```- IREE의 정규식에서 Conv의 Input은 Pad를 포함하고 있다. 이것을 원래 Input으로 바꾸고, Pad는 Attributes로써 Conv에 포함시킨다.

# 1. Patterns

## 1.1 VailPadPattern
- 아래 식에서 %8의 값을 %7에 넣음으로써 padding을 가진 input을 만들고 있다. 
```
    %7 = linalg.fill ins(%c-128_i8 : i8) outs(%6 : tensor<3x642x642xi8>) -> tensor<3x642x642xi8>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<3x640x640xf32>) outs(%5 : tensor<3x640x640xi8>) {
    
    ...
    
    } -> tensor<3x640x640xi8>
    %inserted_slice = tensor.insert_slice %8 into %7[0, 1, 1] [3, 640, 640] [1, 1, 1] : tensor<3x640x640xi8> into tensor<3x642x642xi8>
```

- trigger : tensor.insert_slice
- 1. inf_cap.conv2d를 찾음
- 2. input이 insert_slice인지 확인.
- 3. pad가 되었는지 확인하고, pad의 수치를 conv2d의 attributes로 넣음. %6, %7, %inserted_slice는 삭제하고, conv의 input은 %8로 한다. (quantize는 유지)

- 예시
```
[before]
 %padded_empty = tensor.empty() : tensor<3x642x642xi8>
 %fill_const = linalg.fill ins(%cst) outs(%padded_empty)
 %quant_out = linalg.generic { ... } // input quantize result (3x640x640)
 %padded_input = tensor.insert_slice %quant_out into %fill_const [0, 1, 1] ...

 %0 = "inf_cap.conv2d"(%padded_input, %weight) <{pad_h = 0, pad_w = 0, ...}>

[after]
 %0 = "inf_cap.conv2d"(%quant_out, %weight) <{pad_h = 1, pad_w = 1, ...}>
```

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
![[Apply_Plugin.png|500]]

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
