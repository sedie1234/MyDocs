keti

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
- constructor = infetron_v2::createQDQFusionPass()

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


