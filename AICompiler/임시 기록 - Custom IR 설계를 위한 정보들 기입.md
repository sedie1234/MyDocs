
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
![Pasted image 20260129101809](https://imgs.hwan123.synology.me:19999/AICompiler/Pasted image 20260129101809.png)
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
    

현재 설계하신 IR 구성에서 **"B Layer의 연산 조합(3.1)"** 시, Conv를 MatMul로 바꿀 때 발생하는 **메모리 레이아웃 변경(Im2Col 등)**을 어느 패스에서 처리할지 결정하셨나요? 이 부분만 명확해지면 완벽한 파이프라인이 될 것 같습니다.