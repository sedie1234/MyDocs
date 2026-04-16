# 02. Hardware Architecture

원본: `source_hw_modeling/hw_architecture.md`, `source_hw_modeling/hw_spec.md`

## HW Spec 요약

| 항목 | 값 |
|---|---|
| **Shared Memory** | 4 MB |
| **Unified Buffer** (per core) | 256 KB |
| **Queue** (per core) | 128 KB |
| **Clock** | 800 MHz (1.25 ns/cycle) |
| **MAC array** (추정) | 16~256 parallel MAC/clock |
| **Vector processor** | 16 lanes |
| **AXI bus width** | 128 bit = 16 B/beat |
| **Cluster 구성** | Pcore + Acore pair 2 개 × 6 cluster = 12 + 12 core |
| **DMA bus** | 6 개 공유 |

## 메모리 계층

```
          ┌─────────────────┐
          │  Host CPU       │
          └────────┬────────┘
                   │ (phase 0~1, 2, 5)
                   ▼
          ┌─────────────────┐
          │  Shared Memory  │  4 MB
          │  ─ metadata/wt  │  (read-only region)
          │  ─ input        │
          │  ─ activation   │  (ref_count-managed)
          │  ─ output       │
          └─┬───────────────┘
            │ DMA (6 shared buses)
            │
  ┌─────────┴─────────┬───────┬────────┐
  ▼                   ▼       ▼        ▼
┌──────┐  ┌──────┐  ... ┌──────┐  ┌──────┐
│Pcore0│  │Pcore1│      │Acore0│  │Acore1│
│ UB   │  │ UB   │      │ UB   │  │ UB   │
│256KB │  │256KB │      │256KB │  │256KB │
└──────┘  └──────┘      └──────┘  └──────┘
```

### Shared Memory (SM)

- **전체 4 MB**
- **모든 core 와 CPU 가 접근 가능**
- NPU 동작 중 (어떤 core 의 PC != 0) CPU 는 **input / output 영역만 접근**
- 영역 구성:
  - **metadata** (readonly): weights, bias, scale 등 컴파일 시 결정된 정적 데이터
  - **input**: 모델 입력 텐서가 올라갈 위치
  - **output**: 모델 출력이 쓰일 위치
  - **activation pool**: 중간 활성값 (ref_count 로 관리, 재활용 가능)
  - **forbidden area**: 특수 용도 예약

### Unified Buffer (UB)

- **core 당 256 KB**
- **연산기와 직접 연결** — weight, input tile, output tile 이 여기 올라감
- **Double buffering**: A/B 2 영역으로 분할, ping-pong 으로 compute ↔ DMA overlap
- Weight stationary: Conv 는 weight 를 UB 에 한 번 올리고 입력을 tile 단위로 흘려 재사용

### Queue

- **core 당 128 KB** (= 512 개 record, 각 2048 bit)
- **ISA record 를 순차 저장**
- 각 core 의 **Program Counter (PC)** 가 다음 실행할 record 를 가리킴
- PC 는 record 번호 (실제 주소 = PC × 2048 bit)

## 코어

### Pcore

- **주 연산 코어**. ISA 가 정의한 모든 연산 가능
  - Conv, Gemm, Attention, Concat, Split, Head
  - Elementwise 계열 전부 (Add, Mul, Sigmoid, MaxPool, Resize, TopK, Transpose, Gather, Copy, Softmax)
- 모델 레이어의 대부분은 Pcore 에서 실행

### Acore

- **Depthwise convolution 전용** 보조 코어
- depthwise conv + 이어진 activation 만 수행 가능
- standard Conv, Gemm 등은 Pcore 에 위임
- 각 Pcore 와 **pair** 로 동작 — 같은 cluster 내 pair 간 DMA 가 가장 빠른 경로

## Core 연결 구조

```
          Cluster 0              Cluster 1         ...     Cluster 5
  ┌─────────────────────┐ ┌─────────────────────┐       ┌─────────────────────┐
  │ Pcore0 ─ Acore0     │ │ Pcore2 ─ Acore2     │       │ Pcore10 ─ Acore10   │
  │   │       │          │ │   │       │          │       │   │         │       │
  │ Pcore1 ─ Acore1     │ │ Pcore3 ─ Acore3     │       │ Pcore11 ─ Acore11   │
  └─────────────────────┘ └─────────────────────┘       └─────────────────────┘
       │                       │                              │
       └─── P ring (12 core 환형) ─────────────────────────────┘
       └─── A ring (12 core 환형) ─────────────────────────────┘
```

- **2 줄 ring**: P core 줄, A core 줄이 각각 원형으로 연결
- **Pair**: 같은 cluster 의 Pcore ↔ Acore 는 직접 연결 (Pair Connect, PC 경유)
- **in-cluster**: 같은 cluster 내 Pcore ↔ Pcore 또는 Acore ↔ Acore 는 short ring hop (NC, Near Connect)
- **out-cluster**: cluster 간 이동은 multi-hop ring — hop 수에 비례해 latency 증가
- **DMA bus 6 개 공유**: 동시 6 개 전송, arbitration 발생 시 stall

### DMA 경로별 성능 (hw_spec.md 기준)

| 경로 | latency (clock/beat) | throughput | 비고 |
|---|---:|---:|---|
| SM ↔ UB | 4~8 | 2~4 GB/s | NoC 경유, bank 충돌 시 +2~4 clock |
| Pcore ↔ Acore pair | 2~4 | 4~8 GB/s | 동일 cluster 내 직접 연결 (빠름) |
| P ↔ P in-cluster | 3~6 | 2.7~5.3 GB/s | 1-hop ring |
| A ↔ A in-cluster | 3~6 | 2.7~5.3 GB/s | 1-hop ring |
| P ↔ P out-cluster | 6~15 | 1~2.7 GB/s | multi-hop, hop 수 비례 |

DMA 총 시간:
```
dma_cycles = initial_latency + ceil(data_size / 16) × clock_per_beat
```

## Register 세트

각 core 마다 존재:

| Register | 용도 |
|---|---|
| **start** | 런타임이 1 로 쓰면 core 가 실행 시작 (PC=1 로 진입) |
| **program counter (PC)** | 다음 실행할 record 번호. 각 instruction layer 종료 후 1 증가 |
| **loop counter** | LOOP_JUMP record 의 loop_count 와 비교하여 loop 판단 |
| **output buffer enable** | LOOP_JUMP 에 도달 시 1, 모든 core 가 1 이면 output interrupt 발생 |
| **read reference count (RefCountTable)** | SM/UB 영역에 "이 데이터는 N 번 읽힐 것" 을 기록하는 잠금 테이블. DMA 가 생산/소비할 때마다 감소. 0 이 되면 영역 재사용 가능. buffer index + addr + size + count 의 구조체 array |
| **read only buffer** | weight 영역의 주소 범위 pair. 이 영역에 쓰려는 DMA 는 오류 처리 |

### RefCountTable 의 역할

- **Core 간 동기화** 의 핵심: producer 가 쓴 영역에 `ref_count = N` 을 걸면, N 명의 consumer 가 읽기 전까진 **재사용 불가**
- Consumer 가 DMA read 할 때 해당 영역의 ref_count 를 1 감소
- 0 이 되면 entry 제거 → allocator 가 그 영역을 재사용 가능
- **Write lock 도 겸함**: 어떤 영역에 쓰려는 DMA 가 있을 때, 그 영역에 ref_count > 0 의 entry 가 있으면 해당 DMA 가 대기
- HW 팀은 bank 단위로 구현 예정. 시뮬은 소프트웨어적으로 관리하되 **max entries / max size 프로파일링 제공**

## 연산 특이사항

### Convolution
- **Conv + Activation 일괄 수행** 이 원칙 — 별도 activation op 불필요 (SiLU 등 활성값 fusion 포함)
- **Weight stationary**: weight 를 UB 에 미리 올리고 input 만 tile 단위로 바꿔가며 재사용
- **Input tiling 우선순위**: Channel 먼저, H 그 다음
- **Core 연계 chain**: core 를 병렬로 돌리는 것이 아니라, core A 의 output 이 core B 의 input 이 되는 **cross-core pipeline** 을 구성할 수 있음 (예: 5×n 입력 → 3×n 출력 → 1×n 출력)

### Double Buffering
- UB 를 절반으로 나눠 compute 와 DMA 를 overlap
- compute-bound 인 큰 Conv 에서 DMA cost 가 숨겨짐
- 작은 op 에선 ping-pong overhead 가 오히려 손해일 수 있음

## 프로파일링 포인트

시뮬레이터가 측정해야 할 주요 값:
1. **Per-core wall time (cycles)** — critical path 진단
2. **SM peak usage** — 4 MB 한도 초과 여부
3. **UB peak per core** — 256 KB 초과 여부
4. **Queue usage per core** — 128 KB / 512 records 한도 여부
5. **DMA bus contention** — 6 개 bus 에서 몇 번 stall 이 발생했는지
6. **RefCountTable peak entries** — sync 테이블 크기
7. **Cross-core hop distribution** — pair / in-cluster / out-cluster DMA 비율

이 값들을 컴파일러 정책 (fusion 수준, chain 적용, 할당기 전략) 을 바꿔가며 관찰하는 것이 실험의 목적.
