# 문서 정의
- 이 문서는 실제 하드웨어를 동작시키는 것에 대해 정의되어 있다.
- 이 문서를 기반으로 시뮬레이터와 하드웨어 모델링 제작에 참고가 되도록 한다.


# phase 0 : 동작 준비 
- 컴파일러는 하드웨어에 올릴 바이너리 파일들과 매핑정보를 file로 만든다. 
- .ihnn은 하드웨어에 올릴 바이너리가 있다.
- .json에는 런타임이 활용할 주소들이 정의되어 있다.

## .ihnn
- .ihnn은 각 queue들과 shared memory, acore의 메모리에 올라갈 바이너리들이 모두 들어있다.
- header에는 .ihnn의 몇 번 주소에서부터 얼마만큼의 크기가 어떤 데이터인지 정보가 들어있다.
- 이 규칙은 별도 문서로 안내한다.

## .json
- 런타임이 .ihnn의 바이너리를 하드웨어의 어느 영역에 복사할 지 정보가 있다.
- 런타임이 pre-processing된 데이터를 어디에 저장할 지 정보가 있다.
- 런타임이 하드웨어 계산 결과 값을 받는 주소가 어디인지 정보가 있다.


# phase 1 : 런타임 model initialize

## 1. data parsing
- 런타임이 .ihnn과 .json을 parsing하여 정보들을 가지고 있는다. 

## 2. binary fetch
- .ihnn에 있는 데이터들을 하드웨어의 각 부분에 입력한다. (shared memory, queue, acore)

### 2.1 data and binary fetch
- .json에는 .ihnn에서의 주소와 사이즈, 컴포넌트 번호와 주소, 사이즈가 하나의 쌍으로 존재한다. : 총 5가지 항목이 하나의 쌍으로 존재한다. (이외에 layer 이름이나 특별한 명칭이 더해질 수 있다.)
- 5가지 정보를 이용하여 .ihnn의 특정 주소에서 특정 사이즈의 데이터를 메모리에 로드하여 지정된 위치에 복사한다.
- 전달 경로는 acore eflash 또는 shared memory 또는 queue이다.


# phase 2 : input-processing
- 런타임이 input image를 pre-processing한다. 
- 런타임이 pre-processing된 image를 Shared Memory의 정해진 구역에 쓴다.
- input image를 쓸 때, .json의 input reference count 값을 read reference count register에 쓴다. 이것은 input 영역에 대한 최초의 reference count 설정이다.
- 이후 하드웨어 동작 중에는 instruction layer의 DMA pack에 적힌 read reference count 값에 따라 register가 동작한다. (런타임이 추가로 개입하지 않음)
- 런타임이 start 함수를 호출하면 phase 3으로 넘어간다.


# phase 3 : HW trigging
- phase 2에서 start 함수가 사용되면 모든 core의 start register를 1로 만든다. 
- start register가 1이 되면 각 core에 있는 program counter register를 1 증가시킨다.


# phase 4 : HW execution
- 각 core는 ISA의 동작 정의에 따라 동작을 실행한다.
- 1개의 instruction layer가 끝나면 program counter register를 1 증가시킨다.
- loop jump를 만나면 해당 core의 output buffer enable register를 1로 만든다. 
- 모든 core의 output buffer enable register가 1이라면 phase 5로 넘어간다.

## 4.1 layer start-run-end 동작
- conv나 matmul를 기준으로 하드웨어는 weight stationary로 동작하므로, layer start에서 weight를 써둔다. (bias도 layer start에서 쓰는 것으로 정의하지만, run으로 옮기는 실험을 진행할 수도 있다. 현재 정의는 layer start에서 하는 것으로 한다.)
- layer run에서는 input만 옮겨와 계산을 하고, output을 옮긴다.
- layer end에서는 마저 옮겨야하는 데이터들을 옮긴다.

# phase 5 : output capture
- 모든 core의 output buffer enable register가 1이 되는 순간 output enable 인터럽트가 발생한다.
- output enable 인터럽트가 발생하면 런타임은 정해진 주소에서 output을 가져가고 output buffer enable register를 모두 0으로 만든다.

# phase 6 : inference loop
- 런타임은 읽은 output을 post-processing하여 적절한 출력으로 만든다.
- inference loop가 종료되는 조건이 만족되지 않았다면 phase2로 넘어간다. 
- inference loop가 종료되는 조건이 만족되었다면 phase7로 넘어간다.

# phase 7 : exit
- 모든 버퍼를 해제하고 런타임을 종료한다. (하기에 따라서 profiling 결과를 출력할 수도 있다.)