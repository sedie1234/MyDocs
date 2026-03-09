# Infetron-V2 Compiler Structure Summary

> 기준 문서: `Custom IR 설계.md` (260205 4차수정본, v0.1)

---

## 1. 전체 파이프라인 흐름

```
IREE (linalg / flow / hal)
        │
        ▼
[IREE to CAP PassPipeline]
        │
        ▼
  INF_CAP Layer  (infetron-capability layer)
        │
        ▼
[CAP to ISA PassPipeline]
        │
        ▼
[PCore Packing PassPipeline]
        │
        ▼
[PCore ISA Optimization PassPipeline]
        │
        ▼
  INF_ISA Layer  (infetron-ISA layer)
        │
        ▼
[ISA to EXP PassPipeline]  ← (INF_Q Layer 폐기, INF_EXP Layer로 대체 예정)
        │
        ▼
  INF_EXP Layer  (infetron-export layer)
        │
        ▼
[EXP to HAL PassPipeline]
        │
        ▼
[Serialization PassPipeline]
        │
        ▼
    vmfb (binary)
```

---

## 2. Abstract Layers (Dialects)

### 2.1 INF_CAP Layer — `infetron-capability layer`

- **목적**: 하드웨어가 실행 가능한 연산들의 집합
- **엔트리 기준**:
  - 하드웨어 opcode와 1:1 대응되는 연산
  - 하드웨어 opcode 조합으로 구현 가능한 연산 (예: conv → matmul 변환)
  - 연산 조합이 하드웨어로 실행 가능한 경우 (예: add-neg-exp-add-div-mul → silu)
- **네임스페이스**: `infetron_v2::cap`

---

### 2.2 INF_ISA Layer — `infetron-ISA layer`

- **목적**: 하드웨어 opcode 레벨 표현 + 스케줄링 정보 보유
- **Ops 구성**:
  - opcode와 1:1 대응
  - PCore / ACore 분리 (ACore는 PCore의 subset; 기본 PCore, 가능한 연산은 ACore attribute 부여)
  - 각 Op에 `latency`, `start_cycle`, `end_cycle`, 코어 할당 정보 포함
- **스케줄링 전략**:
  1. latency 기록 → critical path 탐색
  2. critical path를 우선 1개 코어에 배치 (stall 삽입으로 시각화)
  3. 나머지 연산은 가장 긴 latency부터 코어에 분산 배치
  4. data locality 기반으로 코어 재배치
  5. 연산 복사 최적화 (결과 복사 vs 연산 복사 비교)

---

### 2.3 INF_EXP Layer — `infetron-export layer`

- **목적**: custom runtime이 하드웨어를 동작시키기 위한 정보 생성 (metadata, memory plan, instruction queue)
- **Dialect 구성** (2개):
  1. **Memory Dialect**: Shared Memory / Unified Buffer 상태 관리, 시간 흐름에 따른 메모리 사용 추적
  2. **Instruction Queue Dialect**: ISA Layer의 region 분할을 기반으로 명령어 큐 관리 (설계 검토 중)
- **비고**: INF_Q Layer를 대체하는 계층 (INF_Q Layer는 폐기됨)

---

## 3. Pass Pipelines

### 3.1 IREE to CAP PassPipeline

| 순서 | Pass | 역할 |
|---|---|---|
| 1 | `QDQFusionPass` | quantize / dequantize 쌍 결합 |
| 2 | `OpFusionPass` | QDQ 외 분리된 연산들 합성 |
| 3 | `PostOpFusionPass` | OpFusion 후처리 (conv 잔여 bias, requantize 등) |
| 4 | `MonoOpFusionPass` | 앞선 규칙 미적용 1:1 op 변환 (mul, add 등) |
| 5 | `VailPadPass` | IREE conv padding 분리 → conv attributes로 이동 |
| 6 | `QOpFusionPass` | quantize - op - dequantize → quantized op 합성 |
| 7 | `GenericOpenPass` | linalg.generic 제거 |
| 8 | `ExitIREEPass` | IREE 관련 IR 제거 |
| 9 | `CAP_CanonicalizationPass` | 불필요 연산 제거 및 정리 (필요 시 구현) |

---

### 3.2 CAP to ISA PassPipeline

| 순서 | Pass | 역할 |
|---|---|---|
| 1 | `ToNCorePass` | CAP ops를 N Core(가상 코어)로 변환, opcode 일치 목표 |
| └ | `Conv Division Pad Pass` | conv에서 pad 분리 (ToNCorePass 하위) |
| 2 | `ToAPCorePass` | N Core ops → ACore / PCore 할당, tiling 등 데이터 포맷 맞춤 |

---

### 3.3 PCore Packing PassPipeline

| 순서 | Pass | 역할 |
|---|---|---|
| 1 | `PCoreVLIWPackingPass` | VLIW packing 가능한 연산 순서 조정 및 vliw_pack/unpack 삽입 |
| 2 | `PCoreOrderFreePackingPass` | 순서 자유 연산 묶음 처리 (현재: 전부 분할 연산) |

---

### 3.4 PCore ISA Optimization PassPipeline

| 순서 | Pass | 역할 |
|---|---|---|
| 1 | `PCoreOrderingAndMarkingPass` | 단일 코어 기준 의존성 순서 정렬 + latency 속성 설정 |
| 2 | `PCoreSpreadingPass` | 무한 코어 가정으로 start_cycle / end_cycle 계산 (backward tracking) |
| 3 | `PCoreFoldingPass` | 실제 코어 수로 제한하여 연산 압축, latency hiding 수행 |
| 4 | `PCoreISACanonicalizationPass` | 불필요 연산 제거, 가상 코어 삭제 |

---

### 3.5 ISA to EXP PassPipeline

> **주의**: 현재 `Custom IR 설계.md`의 2.5/2.6절은 폐기된 INF_Q Layer 기반으로 기술되어 있음.
> INF_EXP Layer로의 전환에 따라 해당 PassPipeline 재정의 필요.

---

### 3.6 Serialization PassPipeline

- IREE의 정규 직렬화 방식으로 vmfb 생성

---

## 4. 구현 상태 요약

| 계층 / Pipeline | 상태 |
|---|---|
| INF_CAP Layer | 구현 완료 (v0.1) |
| INF_ISA Layer | 설계 완료, 구현 진행 중 |
| INF_EXP Layer | 설계 중 (INF_Q Layer 대체) |
| IREE to CAP Pipeline | 1차 완성 (yolov10_top3_qdq 기준) |
| CAP to ISA Pipeline | 구현 중 |
| PCore Packing Pipeline | 설계 완료 |
| PCore ISA Optimization | 설계 완료 |
| ISA to EXP / EXP to HAL | 재설계 필요 (Q Layer 폐기에 따라) |
| Serialization | IREE 기본 활용 |
