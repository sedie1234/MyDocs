# 05. Experiment Methodology

이 문서는 **가상 컴파일러 + 하드웨어 시뮬레이터** 를 이용해 NPU 의 동작·성능 특성을 조사하는 실험 방법론이다. 2026-04 시점의 관행을 정리한 것이며, 시간이 지나면 업데이트.

## 실험 체계

```
┌──────────────────────┐
│ ONNX 모델 (입력)       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ 가상 컴파일러                               │
│ (hw-profiling/scripts/onnx_to_ihnn.py)    │
│                                          │
│ ─ ONNX 파싱 → hw_ops list                 │
│ ─ shape-only canonicalization             │
│ ─ virtual concat / split alias            │
│ ─ chain scheduling                        │
│ ─ 12-core 분배                             │
│ ─ SM 할당 (multi-variant FFD + search)    │
│ ─ ALAP op reordering (자동 revert)        │
│ ─ cut point 기반 variant 선택              │
│ ─ record 인코딩                            │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ .ihnn (binary) + .json (메타)              │
│ (실험 샘플 — 실제 NPU 용은 아님)             │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ 하드웨어 시뮬레이터 (별도 세션)              │
│ (hw-profiling/src/simulator/)              │
│                                          │
│ ─ .ihnn/.json 로드                         │
│ ─ phase 0~7 실행                           │
│ ─ cycle-by-cycle 모델링                    │
│ ─ per-core timeline + DMA bus             │
│ ─ ref_count sync                          │
│ ─ profiling event log                     │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ Profiling 결과                             │
│                                          │
│ ─ Wall time / cycles                      │
│ ─ Per-core 활용도                          │
│ ─ SM / UB / Queue peak                    │
│ ─ DMA bus contention                      │
│ ─ ERRORS (한도 초과 DMA)                   │
│ ─ RefCountTable 동작                       │
│ ─ Timeline HTML / CSV                     │
└──────────────────────────────────────────┘
```

## 가상 컴파일러 사용법

### 기본 명령
```bash
source /home/keti/miniconda3/bin/activate xla29
cd /home/keti/workspace/AICompiler/hw-profiling

python3 scripts/onnx_to_ihnn.py \
  --onnx <입력 onnx 경로> \
  --output-ihnn test/data/<name>.ihnn \
  --output-json test/data/<name>.json
```

### 주요 플래그

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--variant={common,decoded,full}` | `full` | 출력 경계 선택 (아래 표 참조) |
| `--nms-mode={onchip,offload}` | `offload` | NMS 처리 방식. `full` variant 에서만 유효 |
| `--head-kind` | `yolov10n-detect` | Head sub_opcode 이름 |
| `--op-fusion` | OFF | 추가 op fusion 활성화 (Conv+Add 등) — 현재는 사용 안 함 |
| `--no-shape-fold` | OFF | Reshape/Flatten/Squeeze 등 shape-only 제거 해제 (실험용) |
| `--no-conv-consumer-chain` | OFF | Conv → 1in consumer chain 해제 |
| `--no-virtual-concat` | OFF | Concat 가상화 해제 |

### Variant 의미 (YOLOv10n 기준)

| variant | cut 지점 | 최종 출력 | 사용처 |
|---|---|---|---|
| `common` | `/model.23/Concat_2` 직후 | 3 pyramid tensor (1,144,H,W) × {80,40,20} | head 교체 실험, backbone-only 벤치마크 |
| `decoded` | `/model.23/Concat_5` 직후 | (1,84,8400) 등가물 (decoded box + cls) | "NMS 만 host" 표준 deployment |
| `full` | 전체 (Head opcode 번들) | 최종 detection (1,300,6) | NMS on-chip 검증 |

## 컴파일러 정책 (어떤 것을 항상 적용하는가)

- **SiLU fusion (Conv+Sigmoid+Mul)** — QDQ 패턴 기반 자동 감지
- **Shape-only canonicalization** — Reshape/Flatten/Unsqueeze/Squeeze/Identity/Cast 제거 (alias 로 처리)
- **Virtual concat** — concat input 을 output region 에 공배치해 중복 저장 제거
- **Split metadata-only** — Split 은 레코드 0 개, 출력을 입력의 alias 로만 등록
- **12-core 프로그램 순서 분배** — op index 기반 block 할당 (ref_count 로 동기화)
- **ALAP op reordering** — dry-run interference 계산 후 이득 있을 때만 적용 (현재 모델은 자동 revert)
- **Multi-variant FFD + local search** — 10 개 heuristic + 2000 random restart + O(n²) pair swap

## 주요 측정 지표

### 컴파일러 시점 (sim 실행 전에도 산출)
- **HW ops 수** — fusion/cut 이후 남은 논리 연산 수
- **SM peak** — 할당기가 계산한 주소 범위 최대값
- **Coalesced physical peak** — 실제 물리 footprint (중복 alias 제거)
- **Queue max per core** — 12 core 중 가장 많은 레코드 수
- **Weights 크기** — weight + bias 총합
- **Concat foldable 수** — virtual concat 으로 묶인 concat 개수
- **Split alias 수** — metadata-only Split 개수

### 시뮬 시점 (별도 세션)
- **Wall time (ms)** — critical path
- **Per-core end cycle** — 12 core 각각의 종료 시점
- **DMA bus 이용률** — 6 bus 의 점유 비율
- **ERRORS** — DMA out-of-bound 등 구조적 오류
- **WAIT_DMA_DONE events** — consumer core 가 producer 를 기다린 횟수
- **RefCountTable peak entries** — sync 테이블 최대 크기

## 자주 쓰는 분석 패턴

### 1. SM overflow 디버깅

1. `SM peak` 이 4 MB 를 넘으면 어느 op 에서 peak 가 형성되는지 확인
2. Coalesced peak 와 할당기 peak 의 차이 = fragmentation
3. Peak op 의 live tensor 리스트를 추출 (어떤 텐서가 동시 생존중인가)
4. 각 텐서의 birth/death 구간을 보고 해결책 선택:
   - 짧은 lifetime → chain 확장으로 SM 경유 제거
   - 긴 lifetime → live range splitting (LRS) 으로 migrate
   - 구조적 → head offload 또는 모델 수정

### 2. Queue overflow 디버깅

1. `Max pcore records` 가 512 (128 KB / 2048 bit) 에 근접하면 overflow 위험
2. 해당 core 에 할당된 op 수와 record 수 점검
3. 해결: 12-core 분배 재조정, chain 확장으로 record 수 감소

### 3. DMA bottleneck 분석

1. 시뮬 결과의 DMA bus contention 수치
2. 특정 bus 가 포화 → 해당 경로 DMA 를 줄이거나 다른 bus 로 분산
3. 해결: 데이터 재사용 (weight stationary), UB↔UB direct (chain)

### 4. 컴파일 정책 비교

같은 ONNX 에 대해 다른 플래그 조합으로 binary 를 생성하고 측정값 차이를 관찰:
- fusion ON vs OFF → 어떤 op fusion 이 critical path 를 줄이는가
- chain ON vs OFF → cross-core pipeline 의 실질 이득
- NMS 처리 방식 (host vs on-chip Head) → 전체 지연 비교

## 아티팩트 구성 규약

실험 결과는 `docs/agent_results/virtual_compiler_experiment/artifacts/` 하위에 variant 별 디렉토리로 저장:

```
artifacts/
├── hw_profiling_<variant>/
│   ├── README.md             # 시나리오 + 컴파일러 수치 요약
│   ├── yolov10n_qdq.ihnn     # 바이너리
│   └── yolov10n_qdq.json     # 메타
```

시뮬레이션 파생물 (`dump.txt`, `sim.log`, `timeline.html`) 은 **별도 세션** 에서 생성해 덮어쓴다. 컴파일러 세션과 시뮬 세션을 분리해 관리.

## 실험 시 주의사항

### ISA 변경이 따라와야 하는 경우
- 새 opcode 추가 → `ISA.md` 갱신 → 시뮬레이터의 decode 테이블 확장 → 가상 컴파일러의 emit 함수 추가 → 측정 재실행
- LAYER_START operand 포맷 변경 → record 인코딩 호환성 검토 (기존 binary 재사용 불가)
- DMA pack 포맷 변경 → 모든 바이너리 재생성

### 시뮬 수치 해석 주의
- **HW 모델이 없는 opcode 는 compute cycle 이 default 1 cycle 또는 elementwise 근사**
- Head opcode 의 실제 cost 는 HW 구현 후에 측정 가능
- Softmax 의 exp/reduce/divide cost 는 HW 가 구체화된 후 정확해짐
- **컴파일러 시점 수치 (SM peak, queue, op 수) 는 항상 valid**
- 시뮬 수치 (cycles, wall time) 는 **HW 모델 상태에 따라 valid 범위가 다름** — 문서에 항상 명시

### 사용자 정책 (2026-04 기준)
- **Weight streaming 금지** — SM 의 weight 영역 rotation / 재적재 불가
- **Simulator 수정 금지** (이 세션에선) — 컴파일러 쪽으로만 조정
- **Host offload 가 기본** — NMS/decode 는 host 처리가 default
- **NPU 만 동작 실험** — CPU fallback 은 별도 실험 범주

## 관련 코드 경로

| 역할 | 경로 |
|---|---|
| 가상 컴파일러 | `hw-profiling/scripts/onnx_to_ihnn.py` |
| 시뮬레이터 | `hw-profiling/src/simulator/` |
| HW 모델 | `hw-profiling/src/hw_model/` |
| ISA 인코더 | `hw-profiling/src/isa/` |
| 런타임 | `hw-profiling/src/runtime/` |
| HW spec JSON | `hw-profiling/config/hw_spec.json` |
| 테스트 바이너리 | `hw-profiling/test/data/` |
| 덤프 유틸 | `hw-profiling/scripts/ihnn_dump.py` |
| 타임라인 시각화 | `hw-profiling/scripts/visualize.py` |
| 실험 아티팩트 | `docs/agent_results/virtual_compiler_experiment/artifacts/` |
| 독립 컴파일러 (참조) | `pubH/independent/` |
