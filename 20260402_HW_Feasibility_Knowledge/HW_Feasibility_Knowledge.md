# Infetron-V2 NPU — HW Feasibility 분석 지식 문서

> **용도**: Claude Team Project 지식 문서  
> **대상 독자**: 하드웨어 설계자, ISA 설계자, 런타임 개발자  
> **목적**: 새로운 ISA 또는 하드웨어 스펙 변경 시 **Queue / UB / SM 동작 가능 여부**를 즉시 검증할 수 있도록 계산 방법과 출력 포맷을 정의한다.  
> **작성일**: 2026-04-02  
> **최종 갱신**: 2026-04-02

---

## 목차

1. [이 문서가 필요한 이유](#1-이-문서가-필요한-이유)
2. [분석 대상 자원 3종](#2-분석-대상-자원-3종)
3. [입력 파라미터 정의](#3-입력-파라미터-정의)
4. [계산식 상세](#4-계산식-상세)
5. [출력 포맷 명세](#5-출력-포맷-명세)
6. [설정 변경 시나리오 가이드](#6-설정-변경-시나리오-가이드)
7. [기준 결과 (Baseline)](#7-기준-결과-baseline)
8. [분석 요청 프롬프트 템플릿](#8-분석-요청-프롬프트-템플릿)

---

## 1. 이 문서가 필요한 이유

컴파일러는 모델의 각 layer를 NPU 하드웨어 자원(Queue, UB, SM)에 매핑한다. 하드웨어 스펙이 바뀌면 기존에 동작하던 모델이 자원 초과로 실행 불가능해질 수 있고, 반대로 자원을 늘리면 더 큰 모델을 지원할 수 있다.

이 문서는 **"이 하드웨어 스펙으로 이 모델이 돌아가는가?"** 라는 질문에 대해:
- 어떤 값이 사용되는지 (입력 파라미터)
- 어떤 계산식으로 판정하는지 (수식)
- 결과를 어떤 형식으로 보여주는지 (출력 포맷)

를 정의한다.

---

## 2. 분석 대상 자원 3종

### 2.1 Queue (Q) — 명령어 저장소

| 항목 | 설명 |
|:---|:---|
| 물리적 위치 | Core 내부, pCore당 1개 |
| 용량 | `Q_SIZE` bytes = `Q_MAX_RECORDS` × `Q_RECORD_SIZE` bytes |
| 역할 | 추론에 필요한 모든 명령어(tile 연산 + layer 제어 + 시스템)를 저장 |
| 제약 | 1회 적재, 추론 중 재로딩 없음 (기본 정책) |
| 초과 시 | 추론 시작 불가 (모든 명령어를 Q에 담을 수 없음) |
| 적재 주체 | **런타임**이 컴파일러 바이너리를 파싱하여 각 pCore Q에 records를 쓴다 |
| 적재 시점 | 추론 시작 전, 1회 |

**Q가 담는 것**: 각 타일 연산 1개 = 1 record. + layer 시작/종료 record + epoch/stop record.

**Q의 실행 흐름**:
1. 컴파일러가 Q records를 생성 (오프라인)
2. 런타임이 바이너리에서 Q records를 파싱하여 각 pCore Q에 적재
3. NPU 시작 후, 각 pCore가 Q records를 순차 실행 (런타임 개입 없음)

### 2.2 Unified Buffer (UB) — 연산 작업 공간

| 항목 | 설명 |
|:---|:---|
| 물리적 위치 | Core 내부, pCore당 1개 |
| 용량 | `UB_SIZE` bytes (ping-pong 사용 시 실효 = `UB_SIZE / 2`) |
| 역할 | **하나의 타일 연산에 필요한 모든 데이터**를 동시에 적재 |
| 적재 내용 | weight + input_tile + output_tile + metadata |
| 제약 | 한 타일 연산 시 위 4종이 동시에 UB에 존재해야 함 |
| 초과 시 | 해당 layer를 현재 타일 크기로 실행 불가 |
| 적재 주체 | **NPU 내부 DMA**가 Q 명령어에 따라 SM에서 UB로 데이터 이동 |
| 적재 시점 | 추론 중, 각 타일 연산 직전 (prefetch) |

**UB 실패의 의미**: 특정 layer에서 weight가 너무 크거나 tile이 너무 크면, 하나의 타일 연산에 필요한 데이터가 UB에 들어가지 않는다.

**UB의 실행 흐름**: 런타임이 UB를 직접 제어하지 않는다. NPU 시작 후 Q 명령어가 DMA를 통해 SM↔UB 데이터 이동을 자율 수행한다.

### 2.3 Shared Memory (SM) — 공유 데이터 저장소

| 항목 | 설명 |
|:---|:---|
| 물리적 위치 | NPU 칩 내부, 12코어 전체 공유 |
| 용량 | `SM_SIZE` bytes |
| 역할 | weight 상주 + 입력/출력 이미지 + layer 전이 시 feature map 보관 |
| 제약 (정적) | `total_weight + input_image ≤ SM_SIZE` |
| 제약 (피크) | layer 전이 시 `total_weight + input_FM + output_FM ≤ SM_SIZE` |
| 초과 시 | weight를 전량 적재할 수 없거나, layer 전이 시 feature map이 넘침 |
| 적재 주체 | **런타임**이 weight와 입력 이미지를 SM의 지정 주소에 쓴다 |
| 적재 시점 | weight: 추론 전 1회. 입력: 추론 요청마다 1회. |

**SM의 실행 흐름**:
1. 런타임이 바이너리에서 weight를 파싱 → SM의 컴파일러 지정 주소에 쓰기
2. 런타임이 입력 이미지 → SM의 컴파일러 지정 주소에 쓰기
3. NPU 시작 → 추론 중 SM은 NPU가 자율 관리 (feature map read/write)
4. 추론 완료 → 런타임이 SM의 컴파일러 지정 출력 주소에서 결과 읽기

### 2.4 aCore — 주소/제어 프로세서

| 항목 | 설명 |
|:---|:---|
| 물리적 위치 | Core 내부, pCore와 1:1 쌍 |
| 역할 | 스칼라 연산, 주소 계산, DMA 제어, 코어 간 통신 관리 |
| 적재 내용 | aCore 프로그램 (컴파일러가 생성) |
| 적재 주체 | **런타임**이 바이너리에서 aCore 프로그램을 파싱하여 각 aCore에 쓴다 |
| 적재 시점 | 추론 전, 1회 |

### 2.5 런타임 실행 흐름 요약

```
런타임 (Host CPU)                              NPU (자율 실행)
─────────────────                              ─────────────────
① 바이너리 파싱
   (weight, Q records, aCore 프로그램 분리)

② 데이터 적재 (추론 전 1회)
   Weight → SM 지정 주소
   Q records → 각 pCore Q
   aCore 프로그램 → 각 aCore

③ 입력 적재 (추론마다)
   입력 이미지 → SM 지정 주소

④ NPU 시작 신호 전송 ──────────────────→  Q 명령어 순차 실행 시작
                                            SM↔UB DMA (자율)
   (대기 중)                                Layer 실행 (자율)
                                            코어 간 동기화 (자율)
⑤ NPU 완료 신호 수신 ←──────────────────  전체 추론 완료

⑥ SM 출력 주소에서 결과 읽기
```

---

## 3. 입력 파라미터 정의

### 3.1 하드웨어 파라미터

```
┌─────────────────────────────────────────────────────┐
│  하드웨어 파라미터 (HW 팀이 제공/변경하는 값)          │
├──────────────────────┬──────────┬───────────────────┤
│ 파라미터             │ 기호     │ 단위              │
├──────────────────────┼──────────┼───────────────────┤
│ 코어 수              │ N_CORES  │ 개                │
│ Q 크기 (per core)    │ Q_SIZE   │ bytes             │
│ Q record 크기        │ Q_REC    │ bytes/record      │
│ Q 최대 records       │ Q_MAX    │ = Q_SIZE / Q_REC  │
│ UB 크기 (per core)   │ UB_SIZE  │ bytes             │
│ UB ping-pong 사용    │ PP       │ true/false        │
│ UB 실효 크기         │ UB_EFF   │ = PP ? UB/2 : UB  │
│ SM 크기 (공유)       │ SM_SIZE  │ bytes             │
│ 데이터 타입          │ DTYPE    │ i8/i16/fp16/fp32  │
│ DTYPE 크기           │ D_BYTES  │ bytes/element     │
└──────────────────────┴──────────┴───────────────────┘
```

### 3.2 모델 파라미터 (per layer)

```
┌─────────────────────────────────────────────────────┐
│  모델 파라미터 (컴파일러가 모델에서 추출하는 값)       │
├──────────────────────┬──────────┬───────────────────┤
│ 파라미터             │ 기호     │ 설명              │
├──────────────────────┼──────────┼───────────────────┤
│ Layer 인덱스         │ i        │ 0-based           │
│ Layer 이름           │ name     │ 예: model.0.conv  │
│ Layer 타입           │ type     │ Conv/DW/1x1       │
│ Kernel 크기          │ K_h, K_w │ 3×3, 1×1 등       │
│ Stride               │ S        │ 1, 2 등           │
│ 입력 형상            │ H_in, W_in, C_in  │         │
│ 출력 형상            │ H_out, W_out, C_out │        │
│ Weight 크기          │ W_bytes  │ bytes             │
│ 입력 FM 크기         │ IN_bytes │ = H_in×W_in×C_in×D_BYTES │
│ 출력 FM 크기         │ OUT_bytes│ = H_out×W_out×C_out×D_BYTES │
└──────────────────────┴──────────┴───────────────────┘
```

### 3.3 운용 정책 파라미터

```
┌─────────────────────────────────────────────────────┐
│  운용 정책 (컴파일러/런타임 설계 결정)                │
├──────────────────────┬──────────┬───────────────────┤
│ 파라미터             │ 기호     │ 설명              │
├──────────────────────┼──────────┼───────────────────┤
│ 타일 크기            │ T        │ 출력 공간 T×T     │
│ 타일링 전략          │ -        │ 고정/Adaptive     │
│ 여유 계수            │ M        │ 코어 불균형 대비  │
│ Metadata overhead    │ META     │ bytes/tile        │
│ Weight 로딩 정책     │ -        │ sm_once/streaming │
│ SM I/O overlap       │ OVERLAP  │ true/false        │
│ 실행 모델            │ -        │ parallel/sequential│
└──────────────────────┴──────────┴───────────────────┘
```

---

## 4. 계산식 상세

### 4.1 타일 관련 기본 계산

#### 출력 타일 수 (per layer)

```
N_tiles(i) = ceil(H_out / T) × ceil(W_out / T)
```

- `T`: 출력 공간의 타일 한 변 크기
- 예: H_out=160, W_out=160, T=32 → ceil(160/32)² = 5² = 25 tiles

#### 입력 타일 크기 (receptive field 역산)

Conv 3×3, stride S인 경우 출력 T×T를 만들려면 입력이 필요한 크기:

```
Conv/DW:
  in_tile_h = S × T + K_h - S
  in_tile_w = S × T + K_w - S

1×1 Conv:
  stride=1: in_tile_h = T,       in_tile_w = T
  stride>1: in_tile_h = S × T,   in_tile_w = S × T
```

**예시** (Conv 3×3, S=1, T=32):
```
in_tile_h = 1 × 32 + 3 - 1 = 34
in_tile_w = 1 × 32 + 3 - 1 = 34
```

#### 타일당 데이터 크기

```
input_tile_bytes  = in_tile_h × in_tile_w × C_in × D_BYTES
output_tile_bytes = T × T × C_out × D_BYTES
```

---

### 4.2 Queue (Q) 계산

#### Record 수 계산

```
# 모든 layer의 tile-op 합산
total_tile_ops = Σ N_tiles(i)   (i = 0 ~ N_layers-1)

# Non-conv ops (concat, add, resize 등)도 tile 기반 명령어 발생
non_conv_tile_ops = Σ (해상도별 non-conv op 수 × 해당 해상도의 tile 수)

# 명령어 record 총 수
instruction_records = total_tile_ops
                    + non_conv_tile_ops
                    + N_total_layers × 2      ← LAYER_START + LAYER_END per layer
                    + 1                        ← EPOCH_COMMIT
                    + 1                        ← STOP
```

#### Sequential 모델 (코어별 분배)

```
# Layer를 N_CORES개 코어에 순차 배정
core_records[c] = (해당 코어의 tile_ops 합)
                + (해당 코어의 layer 수 × 2)
                + 1                            ← NOP/EPOCH

worst_core_records = max(core_records[c])   (c = 0 ~ N_CORES-1)
```

#### 판정

```
verdict_Q = worst_core_records ≤ Q_MAX  →  PASS
            worst_core_records > Q_MAX  →  FAIL
```

**표시할 값**:

| 표시 항목 | 계산식 |
|:---|:---|
| Core당 worst records | `worst_core_records` |
| Q 사용률 | `worst_core_records / Q_MAX × 100%` |
| 여유 records | `Q_MAX - worst_core_records` |

---

### 4.3 Unified Buffer (UB) 계산

#### 한 타일 연산에 필요한 UB 공간

```
UB_need(i, T) = W_bytes(i)                  ← weight 전량
              + input_tile_bytes(i, T)       ← 입력 타일
              + output_tile_bytes(i, T)      ← 출력 타일
              + META                         ← metadata overhead
```

#### 고정 타일 판정

```
verdict_UB(i) = UB_need(i, T_fixed) ≤ UB_EFF  →  PASS
                UB_need(i, T_fixed) > UB_EFF  →  FAIL
```

#### Adaptive 타일 (binary search)

```
T_max = max T where UB_need(i, T) ≤ UB_EFF

if UB_need(i, 1) > UB_EFF:
    T_max = 1, verdict = FAIL    ← tile=1×1로도 불가 (weight 자체가 초과)
else:
    binary search [1, max(H_out, W_out)] for largest T fitting UB_EFF
    verdict = PASS
```

#### 실패 원인 분류

| 원인 | 조건 | 의미 |
|:---|:---|:---|
| `weight_exceeds_ub` | `W_bytes > UB_EFF - META` | weight 단독으로 UB 초과. 어떤 타일 크기도 불가. |
| `tile_too_large` | 위 아닌데 `UB_need > UB_EFF` | 타일을 줄이면 해결 가능. Adaptive에서 자동 처리. |

**표시할 값 (per layer)**:

| 표시 항목 | 계산식 |
|:---|:---|
| Weight | `W_bytes` |
| Input tile | `in_tile_h × in_tile_w` → `input_tile_bytes` |
| Output tile | `T × T` → `output_tile_bytes` |
| Metadata | `META` |
| Total UB need | `UB_need` |
| Budget | `UB_EFF` |
| Margin | `UB_EFF - UB_need` |
| Verdict | PASS / FAIL |
| Fail reason | `weight_exceeds_ub` / `tile_too_large` / (없음) |

---

### 4.4 Shared Memory (SM) 계산

#### 정적 검사 (추론 시작 전)

```
static_total = total_weight + input_image_bytes

verdict_SM_static = static_total ≤ SM_SIZE  →  PASS
                    static_total > SM_SIZE  →  FAIL
```

- `total_weight = Σ W_bytes(i)` (모든 layer weight 합)
- `input_image_bytes` (예: 640×640×3×1 = 1,228,800 bytes)

#### 피크 검사 (layer 전이 중)

**Overlap 모드 (SM에서 input/output이 같은 공간 재사용)**:
```
peak(i) = total_weight + max(IN_bytes(i), OUT_bytes(i))
```

**Non-overlap 모드 (input, output 별도 공간)**:
```
peak(i) = total_weight + IN_bytes(i) + OUT_bytes(i)
```

```
verdict_SM_peak(i) = peak(i) ≤ SM_SIZE  →  PASS
                     peak(i) > SM_SIZE  →  FAIL
```

**표시할 값**:

| 표시 항목 | 계산식 |
|:---|:---|
| Total weight | `Σ W_bytes` |
| Input image | `input_image_bytes` |
| 정적 합산 | `total_weight + input_image_bytes` |
| SM 용량 | `SM_SIZE` |
| 정적 판정 | PASS / FAIL |
| Layer별 peak | `peak(i)` |
| Layer별 초과량 | `peak(i) - SM_SIZE` (음수면 여유) |

---

### 4.5 Safe Boundary 계산 (Q+UB 동시 만족 한계)

특정 하드웨어 스펙에서 **각 layer가 허용할 수 있는 최대 weight 크기**를 역산한다.

```
# Step 1: 코어의 Q 타일 예산
q_tile_budget = Q_MAX - (코어 내 layer 수 × 2) - 1

# Step 2: 각 layer의 타일 예산 (출력 면적 비례 배분)
layer_tile_share(i) = q_tile_budget × (H_out(i) × W_out(i)) / Σ(코어 내 모든 layer 출력 면적)

# Step 3: Q 제약을 만족하는 최소 타일 크기
min_T(i) = min T where N_tiles(i, T) ≤ layer_tile_share(i)

# Step 4: 해당 T에서 UB에 들어갈 수 있는 최대 weight
max_weight(i) = UB_EFF
              - input_tile_bytes(i, min_T)
              - output_tile_bytes(i, min_T)
              - META

# Step 5: 판정
weight_ok(i) = W_bytes(i) ≤ max_weight(i)
```

**이 결과의 의미**: HW 설계자가 "UB를 X KiB로 바꾸면 layer Y의 weight 한계가 Z KB가 된다"를 바로 확인 가능.

---

### 4.6 종합 판정

```
Overall = PASS   if Q=PASS AND UB=PASS AND SM=PASS
          FAIL   otherwise
```

---

## 5. 출력 포맷 명세

분석 결과는 반드시 아래 포맷을 따른다. 모든 값에 **사용된 수치와 계산 과정**이 포함되어야 한다.

### 5.1 분석 조건 요약 (반드시 첫 섹션)

```markdown
## 분석 조건

| 항목 | 값 | 비고 |
|:---|:---|:---|
| 대상 모델 | {모델명} ({layer수} layers, weight {총weight} MB) | |
| 코어 | {N_CORES} pCore | |
| Q | {Q_SIZE/1024} KiB/core = {Q_MAX} records ({Q_REC}B/record) | |
| UB | {UB_SIZE/1024} KiB/core, ping-pong = {UB_EFF/1024} KiB | |
| SM | {SM_SIZE/1024/1024} MiB 공유 | |
| 타일링 | {타일링 전략 목록} | |
| 여유 계수 | {M}× | |
| Weight 정책 | {sm_once / streaming} | |
| 입력 | {H}×{W}×{C} {dtype} = {input_bytes/1024/1024:.2f} MB | |
| SM overlap | {true/false} | |
```

### 5.2 종합 판정표

```markdown
## 종합 판정표

| 모델 | 타일링 | Q (worst/max) | UB (pass/total) | SM | **종합** |
|:---|:---|:---|:---|:---|:---|
| {model} | {tiling} | {verdict} ({worst}/{Q_MAX}, {pct}%) | {verdict} ({pass}/{total}) | {verdict} | **{overall}** |
```

**각 셀에 반드시 포함**: verdict + 핵심 수치 + 사용률(%)

### 5.3 자원별 상세 분석

#### Queue 상세

```markdown
### Queue 분석

**계산 과정**:
- Total tile-ops: {값} (conv {값} + non-conv {값})
- Instruction records: {tile_ops} + {layer_count}×2 + 2 = {총records}
- Core당 worst: {worst} records (Core {id}, layers {range})
- Q 용량: {Q_MAX} records
- **사용률: {worst/Q_MAX×100:.1f}% → {verdict}**
- 여유: {Q_MAX - worst} records ({여유%}%)

| Core | Layers | Conv | Non-conv | Tile-ops | Records | 사용률 | Verdict |
|:---|:---|---:|---:|---:|---:|---:|:---|
| Core 0 | 0-6 | 7 | 1 | {값} | {값} | {%} | {verdict} |
| ... | | | | | | | |
```

#### UB 상세

```markdown
### UB 분석

**UB 적재 모델**: weight 전량 → 남은 공간에 input tile + output tile + metadata
**계산식**: UB_need = weight + in_tile_h×in_tile_w×C_in + T×T×C_out + {META}

| Idx | Layer | Type | Kernel | Stride | Weight | In tile (HxW) | In tile (B) | Out tile (B) | Meta | Total UB | Budget | Margin | Verdict |
|---:|:---|:---|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|:---|
| 0 | model.0.conv | Conv | 3×3 | 2 | 432 | 34×34 | {값} | {값} | 512 | {값} | {budget} | {값} | PASS |

**FAIL layers 상세**:

Layer [{idx}] {name}:
  - Weight: {W_bytes} bytes
  - Input tile: {in_h}×{in_w}×{C_in} = {in_bytes} bytes
    (계산: S×T + K - S = {S}×{T} + {K} - {S} = {in_h})
  - Output tile: {T}×{T}×{C_out} = {out_bytes} bytes
  - Metadata: {META} bytes
  - Total: {total} bytes > Budget {budget} bytes
  - 초과량: {total - budget} bytes ({초과%}%)
  - 원인: {weight_exceeds_ub / tile_too_large}
```

#### SM 상세

```markdown
### SM 분석

**정적 검사**:
- Total weight: {total_weight} bytes = {MB} MB
- Input image: {input_bytes} bytes = {MB} MB
- 합산: {static_total} bytes = {MB} MB
- SM 용량: {SM_SIZE} bytes = {MB} MB
- **여유: {SM_SIZE - static_total} bytes → {verdict}**

**피크 검사** (overlap={true/false}):

| Layer | Input FM | Output FM | FM need | Peak (wt+FM) | SM | Margin | Verdict |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 0 | {in_bytes} | {out_bytes} | {fm_need} | {peak} | {SM_SIZE} | {margin} | {verdict} |
```

### 5.4 Safe Boundary 분석

```markdown
### Safe Boundary (Q+UB 동시 만족 한계)

**의미**: 각 layer에서 Q와 UB를 동시에 만족하면서 허용 가능한 최대 weight.
**HW 설계자 활용**: 이 값보다 weight가 크면 해당 layer 실행 불가.

| Idx | Core | Layer | Type | Out HW | 현재 Weight | 최대 Weight | Margin | Min T | Tiles | OK |
|---:|---:|:---|:---|:---|---:|---:|---:|---:|---:|:---|
| 0 | 0 | model.0.conv | Conv | 320×320 | 432 | {값} | {값} | {값} | {값} | OK |

**가장 빡빡한 layer**: [{idx}] {name}
  - 현재 weight: {cur} bytes
  - 최대 허용: {max} bytes
  - 여유: {margin} bytes ({margin/max×100:.1f}%)
```

### 5.5 권고사항

```markdown
### 권고사항

**현재 스펙으로 실행 가능한 모델**: {목록}
**현재 스펙으로 실행 불가능한 모델**: {목록}

**병목 자원 순위**:
1. {자원명} — {이유} (FAIL {n}건)
2. {자원명} — {이유}

**해결 방안**:
| 방안 | 변경 내용 | 예상 효과 | 우선순위 |
|:---|:---|:---|:---|
| {방안1} | {변경} | {효과} | {높음/중간/낮음} |
```

---

## 6. 설정 변경 시나리오 가이드

하드웨어 스펙 변경 시 아래 파라미터를 조정하고 재분석한다.

### 6.1 변경 가능 파라미터와 영향 범위

| 파라미터 변경 | 영향받는 자원 | 기대 효과 |
|:---|:---|:---|
| Q_SIZE 증가 | Q | Q FAIL 해소 |
| UB_SIZE 증가 | UB, Q (adaptive 시) | UB FAIL 해소, adaptive tile 증가 → Q tile-ops 감소 |
| SM_SIZE 증가 | SM | 더 큰 모델 weight 수용 |
| N_CORES 변경 | Q (sequential) | 코어당 layer 수 변동 |
| Ping-pong 제거 | UB | UB 실효 크기 2배 |
| Weight streaming | UB | weight를 UB에서 제외 → UB 대폭 완화 |
| SM overlap | SM | SM peak 요구량 감소 (max(in,out) vs in+out) |
| 여유 계수 M | Q | M=1.0 낙관, M=2.0 보수 |

### 6.2 변경 예시

#### "UB를 256 KiB로 늘리면?"
```
변경: UB_SIZE = 262144 (256 KiB), PP=true → UB_EFF = 131072 (128 KiB)
영향: UB FAIL layers 감소, adaptive tile 증가 → Q tile-ops 감소
```

#### "SM을 8 MiB로 늘리면?"
```
변경: SM_SIZE = 8388608 (8 MiB)
영향: v10n 완전 PASS, v10s weight(6.9MB)+input(1.2MB)=8.1MB → 여전히 초과
     v10s를 위해서는 SM ≥ 9 MiB 필요
```

#### "Weight streaming을 지원하면?"
```
변경: weight_in_ub = false
계산 변경: UB_need = input_tile + output_tile + META (weight 제외)
영향: UB FAIL 대폭 감소 (weight_exceeds_ub 원인 전부 해소)
```

---

## 7. 기준 결과 (Baseline)

### 7.1 기준 하드웨어 스펙

| 파라미터 | 값 |
|:---|:---|
| N_CORES | 12 |
| Q_SIZE | 131,072 bytes (128 KiB) |
| Q_REC | 256 bytes |
| Q_MAX | 512 records |
| UB_SIZE | 262,144 bytes (256 KiB) |
| Ping-pong | true → UB_EFF = 131,072 bytes (128 KiB) |
| SM_SIZE | 4,194,304 bytes (4 MiB) |
| DTYPE | i8 (1 byte) |
| META | 512 bytes |
| M (여유계수) | 1.5 |

### 7.2 기준 모델 스펙

| 모델 | Conv layers | Total weight | Input | Max single weight |
|:---|---:|:---|:---|:---|
| YOLOv10n | 83 | 2.19 MB | 640×640×3 i8 = 1.17 MB | 144 KB |
| YOLOv10s | 87 | 6.90 MB | 640×640×3 i8 = 1.17 MB | 512 KB |

### 7.3 기준 결과

| 모델 | 타일링 | Q | UB | SM | **종합** |
|:---|:---|:---|:---|:---|:---|
| v10n | Adaptive_Half (128K) | FAIL (6577/512) | FAIL (79/83) | PASS | **FAIL** |
| v10n | Adaptive_Full (256K) | PASS (236/512) | PASS (83/83) | PASS | **PASS** |
| v10s | Adaptive_Half (128K) | FAIL (27954/512) | FAIL (58/87) | FAIL | **FAIL** |
| v10s | Adaptive_Full (256K) | FAIL (6621/512) | FAIL (79/87) | FAIL | **FAIL** |

**핵심 발견**:
- v10n은 UB 전체(256K) 사용 + Adaptive tiling으로 **PASS 가능**
- v10s는 SM 초과(weight 6.9MB > 4MB)로 **물리적 불가**
- UB가 가장 심각한 병목 (weight 크기가 UB를 지배)

### 7.4 분석 이력

| 날짜 | 시나리오 | 핵심 설정 변경 | 결과 |
|:---|:---|:---|:---|
| 2026-04-01 | 01 Baseline | 기본 | 6/6 FAIL |
| 2026-04-01 | 02 Adaptive Full | UB full(128K) 추가 | v10n Adaptive Full = COND |
| 2026-04-01 | 03 UB 256K + Sequential | UB 256K, sequential, SM overlap | **v10n Adaptive Full = PASS** |
| 2026-04-01 | 04 Safe Boundary | Q+UB 역산 | v10n 전체 PASS (margin 88KB) |

---

## 8. 분석 요청 프롬프트 템플릿

하드웨어 스펙 변경 시 아래 프롬프트를 복사하여 값만 변경하면 동일한 형식의 분석 보고서가 생성된다.

```
다음 하드웨어 스펙으로 feasibility 분석을 수행해줘.
출력은 "HW Feasibility 분석 지식 문서"의 §5 출력 포맷을 따를 것.

## 하드웨어 스펙
- 코어: 【12 pCore】
- Q: 【128 KiB/core, 256B/record, 512 records max】
- UB: 【256 KiB/core, ping-pong = 128 KiB】
- SM: 【4 MiB 공유】

## 타겟 모델
- 【YOLOv10n】 (conv layer 분석 데이터 참조)

## 타일링 전략
1. 【Adaptive Half (UB half 기준)】
2. 【Adaptive Full (UB 전체 기준)】

## 운용 정책
- Weight: 【SM에 1회 전량 적재, UB에 weight 동시 적재】
- SM overlap: 【true】
- 여유 계수: 【1.5】
- 실행 모델: 【sequential】

## 출력 요구사항
- §5.1 분석 조건 요약
- §5.2 종합 판정표
- §5.3 자원별 상세 분석 (Q, UB, SM 각각)
- §5.4 Safe Boundary 분석
- §5.5 권고사항
- FAIL layer는 계산 과정을 단계별로 전개할 것
```

---

## 부록 A. 해상도별 타일 수 참조표

| 해상도 | 8×8 | 16×16 | 32×32 | 48×48 | 64×64 |
|:---|---:|---:|---:|---:|---:|
| 320×320 | 1,600 | 400 | 100 | 49 | 25 |
| 160×160 | 400 | 100 | 25 | 16 | 9 |
| 80×80 | 100 | 25 | 9 | 4 | 4 |
| 40×40 | 25 | 9 | 4 | 1 | 1 |
| 20×20 | 9 | 4 | 1 | 1 | 1 |

## 부록 B. 자주 묻는 질문 (HW 설계자용)

**Q: "UB를 X KiB로 바꾸면 어떤 layer까지 돌아가나요?"**
→ Safe Boundary 분석의 `max_weight` 열을 확인. `max_weight ≥ 현재 weight`인 layer만 PASS.

**Q: "SM을 몇 MiB로 해야 v10s가 돌아가나요?"**
→ `SM ≥ total_weight + input_image = 6.90 + 1.17 = 8.07 MB` → **최소 9 MiB** (피크 여유 포함)

**Q: "Q record 수를 늘리면 효과가 있나요?"**
→ Q는 대부분 시나리오에서 PASS. Adaptive tiling에서만 FAIL하므로, UB 확장이 우선.

**Q: "Weight streaming을 지원하면 UB 문제가 해결되나요?"**
→ `weight_exceeds_ub` 원인의 FAIL은 전부 해소. `tile_too_large`는 여전히 존재 가능하나 매우 드묾.

**Q: "새로운 모델을 분석하려면?"**
→ 1) ONNX 모델에서 conv layer 분석 (layer별 kernel, stride, channel, weight size 추출)
→ 2) `model_data.json`에 추가
→ 3) 분석 프롬프트 실행

---

## 부록 C. 계산 도구

분석을 자동화하는 Python 스크립트:

| 파일 | 역할 |
|:---|:---|
| `calculate.py` | Q/UB/SM/Safe Boundary 전체 계산 |
| `config.json` | 하드웨어 스펙 + 운용 정책 설정 |
| `model_data.json` | 모델별 layer 데이터 |
| `parse_md.py` | Conv layer 분석 마크다운 → JSON 변환 |

```bash
# 실행
python calculate.py                    # 기본 config
python calculate.py --config my.json   # 커스텀 config

# 출력
results/feasibility_summary.csv       # 종합 판정
results/queue_analysis.csv            # Queue 상세
results/ub_analysis.csv               # UB 상세
results/sm_analysis.csv               # SM 상세
results/safe_boundary.csv             # Safe Boundary
results/results.json                  # 전체 결과 (JSON)
```
