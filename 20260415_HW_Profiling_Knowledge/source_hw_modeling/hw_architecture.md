# 하드웨어 구성요소

## Shared memory
- cpu와 모든 core들이 접근할 수 있는 메모리이다.
- size는 hw_spec에 따른다.
- NPU가 동작모드 (program counter register가 1개라도 0이 아닌 상태)라면 cpu는 shared memory의 input과 output 구간에만 접근하는 것을 원칙으로 한다.

## Unified Buffer
- core 내부의 연산기와 직접 연결된 Buffer다.
- 연산기가 연산에 필요한 데이터는 이 곳에서 가져간다.
- 연산기가 연산이 완료되면 이 곳에 저장한다.

## Queue
- 각 core마다 존재하며 program counter register와 함께 동작한다.

## Acore
- depthwise convolution만 연산할 수 있다. depthwise convolution과 이어진 Activation을 연산할 수 있다.
- standard convolution, Gemm 등은 pCore에 위임한다.

## Pcore
- 모델을 돌리기 위한 모든 연산을 수행할 수 있다.
- 구체적인 수행할 수 있는 연산은 ISA 문서에서 정의한다.


## registers
- 각 core별로 같은 크기의 bit를 가진 register들은 1개의 통합 register로써 관리될 수 있다.

### start register
- 각 core별로 1개씩 존재한다.
- start register는 기본적으로 0이고, 1이 되는 순간 core의 program counter register가 0이 되는 것을 기다린다.
- start register가 1이고, core의 program counter register가 0이면 core의 program counter register를 1 증가시킨다.
- core의 program counter register를 1 증가시키는 임무를 마쳤다면 start register를 0으로 만든다.

### program counter register
- 각 core별로 1개씩 존재한다.
- program counter register는 각 queue의 instrunction layer의 번호를 의미한다.
- 실제 주소는 ISA record size 만큼 이동한다.
- 결과적으로 실제로 가리키고 있는 주소는 pc x ISA record size 가 현재 가리키고 있는 주소이다.
- program counter register의 최대값은 queue size / ISA record size이다. 

### loop counter register
- 각 core별로 1개씩 존재한다.
- cmd header가 loop jump인 instruction layer를 만나면 1증가시킨다.
- cmd header가 loop jump인 instruction layer의 loop_count 값과 비교하여 register의 값이 더 작다면 program counter를 cmd header가 loop start인 명령어 줄로 이동시킨다. 현재 ISA구조상으로는 program counter를 고정적으로 1로 만든다. 
- cmd header가 loop jump인 instruction layer의 loop_count 값이 0이라면 무조건 program counter를 cmd header가 loop start인 명령어 줄로 이동시킨다. 현재 ISA구조상으로는 program counter를 고정적으로 1로 만든다.
- cmd header가 loop jump인 instruction layer의 loop_count 값과 비교하여 register의 값이 더 작지 않다면 program counter를 cmd header가 loop end인 명령어 줄로 이동시킨다.
- cmd header가 stop인 instruction layer를 만나면 0이 된다. 현재 ISA 구조상으로는 program counter이 0일때 0이 된다.

### output buffer enable register
- 각 core별로 1개씩 존재한다.
- cmd header가 loop jump인 instruction layer에 도착하면 각자의 register를 1로 만든다. 
- 모든 값이 1이 되면 output enable interrupt를 발생시킨다.
- 런타임의 interrupt service routine이 종료되면 모두 0으로 초기화 한다. 

### read reference count register
- 어떤 buffer의 참조 count를 정의한다. DMA 동작에서 read reference count에 따라 값을 설정한다.
- 하드웨어를 만드는 입장에서는 bank 단위로 복잡하게 관리해야 하지만, 본 시뮬레이터에서는 소프트웨어적인 방법으로 정의한다. 다만, 하드웨어 구현에는 관여하지 않음을 명시하고, 실제 하드웨어 구현에 도움이 될 수 있도록 이 레지스터의 전체 크기에 대한 프로파일링을 제공받을 수 있도록 한다. 
- 저장되는 값은 buffer(SM인지, \#번째 core의 UB인지) - address - size - reference count 형식의 구조체를 array로써 저장할 수 있도록 한다. 
- 시뮬레이터에서는 위 구조체의 array를 동적으로 관리하여, 사용될 경우 구조체 element를 늘리고 사용이 끝나면 element를 제거한다. 시뮬레이터는 array의 최대 element 수와 최대 size에 대해 프로파일링 정보를 제공한다.
- dma에서 dst addr, data size, transfer type의 dst buffer, reference count의 값을 저장한다.
- dma가 동작할 때, src addr, data size에 따라 array에서 적절한 element의 reference count를 1감소시키고, 0이되면 해당 element를 array에서 제거한다.
- dma를 동작시킬 때, dst addr고 data size를 기반으로 array의 element를 조회하여 reference count가 남아있다며 해당 dma 동작 및 instruction layer의 동작을 일시정지한다. reference count가 0이 되고 elemnet에서 사라지면 동작을 재개한다. => 각 buffer 별로 write queue를 운영할 것도 고려.

### read only buffer register
- weight가 쓰여져 있는 Shared Memory 구간은 reference count로 동작하지 않지만 read only로 동작한다.
- 오직 runtime의 write에 의해서만 데이터가 바뀔 수 있으며, 하드웨어 동작 중에는 read만 가능하다.
- 이 구역에 대한 정보를 저장하는 register다.
- 시작 주소와 끝 주소를 1개의 pair로써 저장하고 있으며, pair의 array로 구성한다.
- dma에 dst addr 및 data size를 조사하여 이 공간에 접근한다면 오류를 출력하며 시뮬레이션을 종료한다. 시뮬레이션을 종료할 때는 디버깅을 할 수 있도록 적절한 정보를 출력한다.

# component connection info

## core pairing
- acore와 pcore는 두 core가 하나의 쌍으로 존재한다.

## cluster n / m
- acore와 pcore를 한 개로 봤을 때, n개의 core를 1개의 cluster로 관리한다.
- 같은 모양의 bus를 사용하더라도 cluster 내부의 데이터 이동속도가 더 빠르다.
- m개의 cluster로 존재하여 최종 hw spec은 n x m x 2 core라고 부른다.

## core의 연결
- core는 2줄로 연결되어 있다. : 한 줄은 p core 줄이고, 다른 한 줄은 a core 줄이다.
- 각 p core와 a core는 서로 짝이 되는 core끼리만 직접 연결되어 있다. 
- 각 줄은 링 형식으로 연결되어 있다. 
- 연결이라 함은 각각의 Unified Buffer 사이의 연결을 의미한다.
- 총 6개의 dma 버스가 공유되고 있다. 

## 메모리 사이의 연결
- Shared memory는 어떤 Unified Buffer라도 연결될 수 있다.


# core 동작에 대한 특이사항

## Convolution
- Convolution은 Convolution + Activation 을 일괄적으로 수행하는 것을 원칙으로 한다.
- Weight Stationary로 동작한다.
- Weight를 Unified Buffer에 모두 올리고, Input을 tiling하여 계산한다.
- Input의 tiling의 우선순위는 Channel을 가장 먼저 자르고, H를 그 다음으로 자른다.
- Core 연계 계산 : Core를 병렬로 연산하는 것이 아니라, Core의 output이 다음 Core의 input이 되는 형식으로 Convolution을 수행한다.
- 예시로, 1번 Core에서는 5xn tiling된 input을 계산하여 3xn output을 계산한다. -> 2번 Core에서는 이 output을 input으로 받아 3xn input을 계산하여 1xn output을 계산한다.


## Double Buffering 구조
- Unified Buffer를 2개의 구역으로 나누어 PingPong하는 구조를 기본으로 한다. 
