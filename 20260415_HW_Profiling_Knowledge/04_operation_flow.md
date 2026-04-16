# 04. Operation Flow — Runtime Phase 0~7

원본: `source_hw_modeling/hw_operating_flow.md`

## 전체 phase 개요

```
Phase 0: 컴파일러 산출물 준비 (.ihnn + .json)
   │
Phase 1: 런타임 초기화 — parsing + binary fetch
   │
Phase 2: input 전처리 + SM input 영역에 write + ref_count 세팅
   │
Phase 3: HW trigger — start register = 1
   │
Phase 4: HW execution — core 들이 자기 queue 를 PC 기반 실행
   │
Phase 5: output capture — interrupt → 런타임이 output 읽음
   │
Phase 6: inference loop — 종료 조건 미만족 시 phase 2 로 return
   │
Phase 7: exit — 버퍼 해제, profiling 결과 출력
```

## Phase 0 — 컴파일러 산출물

### `.ihnn` (바이너리)

구성:
- **header** (392 bytes): 파일 내 각 섹션의 위치 navigator
  - Pcore queue × 12 (address + size)
  - Acore queue × 12
  - Acore eflash tensor × 12 (initial weight)
  - Shared memory tensor (metadata)
- **pcore queue binary × 12**: 각 core 의 ISA record 스트림
- **acore queue binary × 12**
- **acore eflash tensor × 12**: acore 별 depthwise weight
- **shared memory tensor**: SM 에 올릴 정적 데이터 (Conv/Gemm weight 등)

모든 tensor binary 는 **i8 (1 byte/element)** 로만 기록. scale/zp 는 ISA record 안에 전달되므로 별도 섹션 차지 없음.

### `.json` (런타임용 메타)

구성:
- **memory_configuration**:
  - `shared_memory.metadata` — weight 등 readonly 영역의 address/size
  - `shared_memory.input` — 입력 텐서 위치
  - `shared_memory.output` — 출력 텐서 위치
  - `shared_memory.forbidden` — 예비
  - `eflash.acore_0 ~ 11` — 각 acore 의 weight 위치
- **queue_configuration**: pcore/acore 별 queue 의 address/size
- **input reference count**: 최초 input 이 몇 번 참조될지

## Phase 1 — 런타임 초기화

1. **Parsing**: .ihnn / .json 을 읽고 내부 구조체로 보관
2. **Binary fetch**: .json 의 매핑 정보에 따라 .ihnn 의 각 섹션을 하드웨어 영역에 전송
   - shared memory 의 metadata 영역에 weight blob 복사
   - 각 pcore queue 에 해당 바이너리 복사
   - 각 acore queue 에 해당 바이너리 복사
   - 각 acore eflash 에 해당 tensor 복사 (acore 용 weight)

이 단계에서 HW 는 아직 실행 안 함. 메모리만 준비됨.

## Phase 2 — Input 전처리

- 런타임이 input image 를 전처리 (resize, normalize, quantize 등)
- 전처리 결과를 `SM.input` 영역에 write
- `.json.input_reference_count` 값을 RefCountTable 에 등록:
  - buffer = SM, addr = input_addr, size = input_size, count = N (consumer 수)
- 이후 런타임은 **더 이상 개입하지 않음** — HW 동작 중 ref_count 는 ISA record 의 DMA pack 이 자동 관리
- 런타임이 `start()` 호출 → phase 3

## Phase 3 — HW Trigger

- 모든 core 의 `start register` 를 1 로 쓴다
- start register 가 1 이 되고 core 의 PC 가 0 이면 PC 를 1 로 증가 (첫 record 실행 시작)
- start register 는 PC 를 증가시킨 뒤 자동으로 0 으로 리셋

## Phase 4 — HW Execution

각 core 가 독립적으로 자기 queue 를 PC 로 순차 실행:

### instruction layer 실행 루프

```
while PC != 0:
    record = queue[PC]
    
    switch record.cmd_header:
        case LAYER_START:
            - opcode 해석
            - weight / bias DMA 수행 (Conv/Gemm 등)
            - shape / scale / zp 를 layer context 에 보관
        
        case LAYER_RUN:
            - input DMA pack 수행 (SM→UB)
            - compute 수행 (HW 연산기)
            - output DMA pack 수행 (UB→SM or UB→다른 core UB)
            - ref_count 갱신
        
        case LAYER_END:
            - 후처리 DMA 최대 6 개 수행
        
        case LOOP_START:
            - loop counter 리셋 (아래에서 사용)
        
        case LOOP_JUMP:
            - loop counter < loop_count 면 PC = 1 (loop 처음으로)
            - 아니면 PC = LOOP_END 위치
            - 또한 output_buffer_enable register = 1
        
        case LOOP_END:
            - PC 증가
        
        case STOP:
            - PC = 0 (종료)
    
    PC += 1
```

### ref_count 동기화

- DMA read 시: src 주소의 RefCountTable entry 를 찾아 count -= 1. 0 이 되면 entry 제거
- DMA write 시: dst 주소에 대한 entry 가 있고 count > 0 이면 **write 대기** (해당 consumer 가 읽을 때까지)
- 이 메커니즘이 **cross-core hazard** 를 자동 처리 — 컴파일러는 core 간 실행 순서를 가정할 필요 없음

### Weight Stationary 패턴

Conv 레이어 예시:
```
LAYER_START Conv:
  weight_dma = SM → self.UB[0..W]       # weight 고정
  bias_dma = SM → self.UB[W..W+B]       # bias 고정
  (C, H, W, K, scales 등 context 보관)

LAYER_RUN (tile 0):
  input_dma_0 = SM → self.UB[W+B..]     # 첫 tile 입력
  compute 0
  output_dma_0 = self.UB → SM (or next core UB)

LAYER_RUN (tile 1):
  input_dma_1 = SM → self.UB[W+B..]     # 두 번째 tile 입력
  compute 1
  output_dma_1 = ...

...

LAYER_END Conv
```

Weight 는 LAYER_START 에서 단 한 번만 UB 에 올라가고, 모든 LAYER_RUN 이 공유. input 만 tile 단위로 갱신.

### Cross-core chain (Conv pipeline)

```
Pcore 0 (Conv A):
  LAYER_RUN (per tile):
    output_dma = SELF.UB → Pcore1.UB[A's weight_total]

Pcore 1 (Conv B):
  LAYER_RUN (per tile):
    input_dma = SELF.UB (no-op DMA, data already there from Pcore 0)
    compute
    output_dma = SELF.UB → Pcore2.UB[B's weight_total]

Pcore 2 (Conv C):
  ...
```

- core A 의 output 이 core B 의 UB 로 직접 흘러들어감 — SM 경유 불필요
- ref_count 로 동기화 (core A 가 write 를 끝낸 뒤에야 core B 가 read 할 수 있음)
- **SM footprint 절감** 이 주 목적

## Phase 5 — Output Capture

- 모든 core 가 LOOP_JUMP 에 도달하면 각자 `output_buffer_enable` register 를 1 로 세팅
- 모든 core 가 1 이 되는 순간 `output enable interrupt` 발생
- 런타임의 ISR 이 `SM.output` 영역에서 결과 읽기
- ISR 종료 시 모든 `output_buffer_enable` = 0

## Phase 6 — Inference Loop

- 런타임이 output 을 post-processing (host-side decode, NMS 등)
- inference loop 종료 조건 미만족 → phase 2 로 return (다음 입력)
- 종료 조건 만족 → phase 7

## Phase 7 — Exit

- 모든 버퍼 해제
- 프로파일링 정보 출력:
  - 총 wall time
  - per-core cycles
  - DMA bus 이용률
  - RefCountTable peak entries
  - SM/UB/Queue 점유 peak

## 주요 실행 특성

### 병렬성
- 모든 core 는 **독립 queue 를 병렬 실행**
- 동기화는 ref_count 로만 (core index 순서 가정 없음)
- DMA bus 는 6 개 공유 → contention 발생 가능 (시뮬레이터가 이를 모델링)

### Weight-Only Read-Only
- SM 의 metadata 영역은 **read-only**
- 이 영역에 쓰려는 DMA 는 에러로 시뮬 종료
- 런타임만 쓸 수 있고 HW 실행 중에는 read 만

### Loop 구조
- 각 core 의 queue 는 `LOOP_START ... records ... LOOP_JUMP LOOP_END` 블록으로 감싸짐
- LOOP_JUMP 의 `loop_count` 가 0 이면 무조건 반복 (무한 루프로 phase 6 의 inference loop 용)
- 런타임이 phase 5 의 interrupt 를 받은 뒤 다음 inference 를 시작하면 LOOP_START 로 돌아가 다시 실행

## 실행 특이사항

- **Conv 는 weight stationary**, bias 는 현재 LAYER_START 에서 UB 로 로드 (실험적으로 LAYER_RUN 으로 옮기는 것도 고려 중)
- **Double buffering**: UB 절반을 A/B 로 나눠 compute 와 DMA overlap
- **Core chain**: Conv → Conv UB 직접 전달로 SM 경유 제거
- **Interrupt 기반 phase 전환**: HW → 런타임 sync 는 interrupt 1 개 (output enable)
