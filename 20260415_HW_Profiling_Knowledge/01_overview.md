# 01. Overview — Infetron-V2 NPU 와 프로파일링 체계

## 30 초 요약

Infetron-V2 는 **int8 추론 전용 NPU** 로, 다음 특성을 가진다:

- **12 Pcore + 12 Acore**, 2 개씩 pair 를 이루고 6 cluster 로 묶임
- **Weight stationary** Conv 중심 설계 — weight 를 UB 에 상주시키고 input 을 tile 단위로 흘려 보냄
- **i8 end-to-end** — 모든 data tensor 는 1 byte/element, scale/zp 는 ISA record 의 고정 필드로만 전달
- **Queue + Program Counter 기반 실행** — 각 core 는 자기 queue 의 2048-bit record 를 PC 로 순차 실행
- **ref_count 기반 데이터 동기화** — DMA 마다 read reference count 를 설정·감소시켜 core 간 hazard 회피

## NPU 동작 개요

```
Host CPU
  │
  │ phase 0~1: .ihnn / .json 로드
  ▼
┌──────────────────────────────────────────┐
│ Shared Memory (4 MB)                     │
│  - weights (read-only)                   │
│  - activation pool (ref_count managed)   │
│  - input / output 영역                    │
└──────────────────────────────────────────┘
  │ DMA
  ▼
┌──────────────────────────────────────────┐
│ 12 Pcore + 12 Acore (6 cluster)          │
│  각 core:                                 │
│    ┌─────────────────────────┐          │
│    │ Queue (128 KB)          │ ← PC      │
│    │  ISA records (2048 bit) │          │
│    └─────────────────────────┘          │
│    ┌─────────────────────────┐          │
│    │ Unified Buffer (256 KB) │ ← 연산기 │
│    │  weight + input + output│          │
│    │  double buffer (A/B)    │          │
│    └─────────────────────────┘          │
└──────────────────────────────────────────┘
```

- 모든 core 는 자기 Queue 의 record 를 **독립적으로 병렬 실행**
- Queue 는 `cmd_header` (LAYER_START / LAYER_RUN / LAYER_END / LOOP_* / STOP) 와 `opcode` 로 구성된 2048-bit record 스트림
- 데이터는 DMA 경로를 따라 SM ↔ UB 또는 UB ↔ UB (core 간) 로 이동
- core 간 동기화는 **ref_count** 레지스터로 처리 (consumer 가 read 시 감소, producer 가 write 시 lock)

## 소프트웨어 체계

```
ONNX 모델
    │
    ▼
┌─────────────────────────────────────────────┐
│ 컴파일러 (두 가지 병행)                        │
│  (1) 독립 컴파일러 (실제 제작 중, pubH/independent) │
│  (2) 가상 컴파일러 (hw-profiling/scripts/onnx_to_ihnn.py) │
│       — 시뮬레이터 검증용 hand-made 생성기     │
└─────────────────────────────────────────────┘
    │
    ▼
 .ihnn + .json + (metadata)
    │
    ▼
┌─────────────────────────────────────────────┐
│ Runtime + Simulator (hw-profiling/src)       │
│  phase 0~1: 바이너리 로드                      │
│  phase 2: input 전처리, ref_count 세팅         │
│  phase 3: HW trigger (start register = 1)    │
│  phase 4: HW execution (core 들이 queue 실행) │
│  phase 5: output capture                     │
│  phase 6: loop                                │
│  phase 7: exit + profiling report             │
└─────────────────────────────────────────────┘
    │
    ▼
  프로파일링 결과 (cycles, memory footprint, queue usage, ERRORS)
```

## 두 컴파일러의 역할 분담

| 구분 | 독립 컴파일러 | 가상 컴파일러 |
|---|---|---|
| 위치 | `pubH/independent/` | `hw-profiling/scripts/onnx_to_ihnn.py` |
| 성격 | 실제 제품 컴파일러 | hand-made Python, 시뮬 용 |
| 파이프라인 | MLIR 기반 StableHLO → inf_hw → ISA | ONNX → hw_ops → records |
| 목적 | 실제 하드웨어에 올릴 binary 생성 | 시뮬레이터 검증용 다양한 시나리오 생성 |
| 미션 | 정확한 compile flow | ISA 변경 유연성 + 다양한 정책 테스트 |

**중요**: 이 세션의 가상 컴파일러 산출물은 **실제 deployment 대상이 아님**. 시뮬레이터에 다양한 "이런 컴파일러라면 이런 binary 가 나올 것" 시나리오를 주입하기 위한 것.

## 이 지식 패키지의 독자

- **HW 엔지니어**: ISA 확장 시 영향 범위 확인
- **컴파일러 엔지니어**: 생성하는 binary 가 어떤 제약을 만족해야 하는지 확인
- **검증 엔지니어**: 시뮬레이터로 어떤 property 를 측정할 수 있는지 확인
- **시스템 엔지니어**: 전체 phase 동작 이해

다음 문서부터는 각 주제별로 상세:
- `02_hw_architecture.md` — 코어/메모리/DMA 구조
- `03_isa_structure.md` — 명령어 형식
- `04_operation_flow.md` — 실행 phase
- `05_experiment_methodology.md` — 실험·분석 방법론
