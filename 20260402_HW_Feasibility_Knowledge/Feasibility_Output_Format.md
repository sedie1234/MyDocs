# Feasibility 분석 결과 출력 포맷 (Claude Team Project용)

> **이 파일의 용도**: Claude Team Project에서 feasibility 분석을 요청받았을 때, 결과를 일관된 형식으로 반환하기 위한 포맷 정의.  
> **규칙**: 이 포맷을 벗어나는 분석 결과를 반환하지 말 것. 모든 FAIL에는 계산 과정을 명시할 것.

---

## 포맷 구조

분석 결과는 반드시 **7개 섹션**을 포함해야 한다. 순서를 지킬 것.

```
1. 분석 조건 요약
2. 종합 판정표
3. Queue 분석 상세
4. UB 분석 상세
5. SM 분석 상세
6. Safe Boundary 분석
7. 권고사항
```

---

## 섹션 1. 분석 조건 요약

**목적**: 어떤 값으로 분석했는지 한눈에 파악.

```markdown
## 1. 분석 조건

| 항목 | 값 |
|:---|:---|
| 대상 모델 | {모델명} ({layer수} layers, weight {총weight_MB:.2f} MB) |
| 추론 경로 | {one2one / one2many / both} |
| 코어 | {N_CORES} pCore |
| Q | {Q_SIZE_KiB} KiB/core = {Q_MAX} records × {Q_REC}B |
| UB (per core) | {UB_SIZE_KiB} KiB, ping-pong {on/off} → 실효 {UB_EFF_KiB} KiB |
| SM (공유) | {SM_SIZE_MiB} MiB |
| 입력 | {H}×{W}×{C} {dtype} = {input_MB:.2f} MB |
| 타일링 전략 | {전략 목록, 쉼표 구분} |
| Weight 정책 | {sm_once + ub_full / sm_once + ub_streaming / ...} |
| SM overlap | {true/false} |
| 여유 계수 | {M}× |
| 실행 모델 | {parallel/sequential} |
| Metadata/tile | {META} bytes |
```

---

## 섹션 2. 종합 판정표

**목적**: 모든 케이스의 결과를 한 테이블에. PASS/FAIL 즉시 식별.

```markdown
## 2. 종합 판정표

| 모델 | 타일링 | Q | UB | SM | **종합** |
|:---|:---|:---|:---|:---|:---|
| {model} | {tiling} | **{Q_verdict}** ({worst_records}/{Q_MAX}, {pct:.0f}%) | **{UB_verdict}** ({pass_count}/{total_layers}) | **{SM_verdict}** | **{overall}** |
```

**표기 규칙**:
- PASS: 볼드 없이 `PASS`
- FAIL: `**FAIL**` (볼드)
- 각 셀에 verdict + (핵심 수치) 포함

---

## 섹션 3. Queue 분석 상세

**목적**: 코어별 record 수와 사용률 표시. FAIL 시 어느 코어가 넘치는지 식별.

```markdown
## 3. Queue 분석

### 계산 과정
```
Total tile-ops = {conv_tile_ops} (conv) + {non_conv_tile_ops} (non-conv) = {total_tile_ops}

[Sequential model]
Core 할당: {N_layers} layers ÷ {N_CORES} cores = core당 {base}~{base+1} layers

Core별 records = (core tile-ops) + (core layers × 2) + 1
```

### 코어별 상세

| Core | Layers | Conv layers | Non-conv ops | Tile-ops | Records | 사용률 | Verdict |
|---:|:---|---:|---:|---:|---:|:---|:---|
| 0 | {start}-{end} | {n} | {nc} | {tile_ops} | {records} | {records/Q_MAX*100:.1f}% | {verdict} |
| ... | | | | | | | |

### 판정
- **Worst core**: Core {id}, {records} records / {Q_MAX} max = **{pct:.1f}%**
- **여유**: {Q_MAX - records} records
- **Verdict: {verdict}**
```

---

## 섹션 4. UB 분석 상세

**목적**: 모든 layer의 UB 적재량 표시. FAIL layer는 계산식 전개.

```markdown
## 4. UB 분석

### 적재 모델
weight 전량 → 남은 공간에 input_tile + output_tile + metadata
**계산식**: `UB_need = W_bytes + (in_h × in_w × C_in) + (T × T × C_out) + META`

### 전체 요약
- PASS: {pass_count}/{total} layers
- FAIL: {fail_count}/{total} layers
  - weight 초과: {wt_exceed_count}개
  - tile 초과: {tile_exceed_count}개
- **Verdict: {verdict}**

### FAIL layer 상세 (계산 전개)

**Layer [{idx}] {name}** ({type}, {kernel}, stride={S})

| 항목 | 계산식 | 값 |
|:---|:---|---:|
| Weight | - | {W_bytes} B |
| Input tile 크기 | S×T + K - S = {S}×{T} + {K} - {S} | {in_h}×{in_w} |
| Input tile bytes | {in_h}×{in_w}×{C_in}×{D_BYTES} | {in_bytes} B |
| Output tile 크기 | T×T | {T}×{T} |
| Output tile bytes | {T}×{T}×{C_out}×{D_BYTES} | {out_bytes} B |
| Metadata | - | {META} B |
| **Total UB need** | {W} + {in} + {out} + {META} | **{total} B** |
| **Budget** | UB_EFF | **{budget} B** |
| **초과량** | total - budget | **{excess} B ({excess_pct:.1f}%)** |
| **원인** | | **{weight_exceeds_ub / tile_too_large}** |

### PASS layer 요약 (compact)

| Idx | Layer | Type | Weight | In tile | Out tile | Total | Budget | Margin | Verdict |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|:---|
| {idx} | {name} | {type} | {wt} | {in} | {out} | {total} | {budget} | {margin} | PASS |
```

---

## 섹션 5. SM 분석 상세

**목적**: 정적/피크 두 가지 검사 결과. 어느 layer에서 SM이 넘치는지 표시.

```markdown
## 5. SM 분석

### 정적 검사 (추론 시작 전)

| 항목 | 값 | 비고 |
|:---|---:|:---|
| Total weight | {total_wt} B = {MB:.2f} MB | Σ 모든 layer weight |
| Input image | {input_bytes} B = {MB:.2f} MB | {H}×{W}×{C}×{D} |
| **합산** | **{static_total} B = {MB:.2f} MB** | |
| SM 용량 | {SM_SIZE} B = {MB:.2f} MB | |
| **Margin** | **{SM_SIZE - static_total} B** | {양수→여유, 음수→초과} |
| **Verdict** | **{PASS/FAIL}** | |

### 피크 검사 (layer 전이 중)

overlap={true/false}
계산식: peak = total_weight + {max(in,out) / in+out}

**FAIL layers**:

| Layer | Input FM | Output FM | FM need | Peak | SM | Margin | Verdict |
|---:|---:|---:|---:|---:|---:|---:|:---|
| {idx} | {in_bytes} | {out_bytes} | {fm_need} | {peak} | {SM_SIZE} | {margin} | **FAIL** |

**판정**: {peak_fail_count}개 layer에서 SM 초과
```

---

## 섹션 6. Safe Boundary 분석

**목적**: HW 설계자가 "이 layer의 weight 한계가 얼마인가"를 직접 확인.

```markdown
## 6. Safe Boundary (Q+UB 동시 만족 한계)

**의미**: 현재 HW 스펙에서 각 layer가 허용할 수 있는 최대 weight 크기.
- max_weight ≥ 현재 weight → OK (안전)
- max_weight < 현재 weight → FAIL (weight 축소 또는 HW 확장 필요)

### 위험 layer (margin < 50% 또는 FAIL)

| Idx | Core | Layer | Type | Out HW | 현재 Wt | 최대 Wt | Margin | Min T | Tiles | OK |
|---:|---:|:---|:---|:---|---:|---:|---:|---:|---:|:---|
| {idx} | {core} | {name} | {type} | {H}×{W} | {cur_wt} | {max_wt} | {margin} | {T} | {tiles} | {OK/FAIL} |

### 가장 빡빡한 layer
- Layer: [{idx}] {name}
- 현재 weight: {cur} bytes
- 최대 허용: {max} bytes
- **Margin: {margin} bytes ({margin_pct:.1f}%)**

### 전체 통계
- OK: {ok_count}/{total} layers
- FAIL: {fail_count}/{total} layers
- 최소 margin: {min_margin} bytes (Layer [{idx}])
- 최대 margin: {max_margin} bytes (Layer [{idx}])
```

---

## 섹션 7. 권고사항

**목적**: 분석 결과에 기반한 조치 방안 제시.

```markdown
## 7. 권고사항

### 현재 상태
- 실행 가능: {모델 목록}
- 실행 불가: {모델 목록} (원인: {Q/UB/SM})

### 병목 자원 순위
| 순위 | 자원 | FAIL 수 | 심각도 | 핵심 원인 |
|---:|:---|---:|:---|:---|
| 1 | {자원} | {n} | {높음/중간/낮음} | {원인} |

### 해결 방안

| 방안 | 변경 내용 | 예상 효과 | 전제 조건 | 우선순위 |
|:---|:---|:---|:---|:---|
| {방안명} | {변경할 HW 파라미터} | {FAIL→PASS 예상 수} | {ISA 지원 등} | {높음/중간/낮음} |

### 런타임 적재 검증

런타임 관점에서 추론 실행 가능 여부를 확인하는 체크리스트.
Feasibility 분석과 별도로, 적재 절차 자체가 성립하는지 검증한다.

| 검증 항목 | 조건 | 판정 |
|:---|:---|:---|
| Q 적재 | 코어당 records ≤ Q_MAX | {PASS/FAIL} |
| Weight 적재 | total_weight ≤ SM_SIZE - input_size | {PASS/FAIL} |
| aCore 프로그램 | aCore 프로그램 크기 ≤ eFlash or 지정 메모리 | {PASS/FAIL/N/A} |
| 입력 적재 | input_size ≤ SM 여유 공간 | {PASS/FAIL} |
| 출력 읽기 | output 주소가 SM 범위 내 | {PASS/FAIL} |

**적재 순서 확인**:
1. ① Weight → SM ② Q records → 각 pCore ③ aCore 프로그램 → 각 aCore : 모두 성공?
2. ④ 입력 → SM 지정 주소 : 성공?
3. ⑤ NPU 시작 가능? (모든 사전 조건 충족)
4. ⑥ 출력 주소에서 결과 읽기 가능?

### 다음 단계
1. {구체적 action item}
2. {구체적 action item}
```

---

## 포맷 적용 규칙

### 필수 원칙

1. **모든 값에 단위 표기**: bytes, KiB, MiB, records, %, 개
2. **FAIL에는 반드시 계산 전개**: 어떤 값 + 어떤 값 = 총 얼마 > 한계 얼마
3. **사용률(%) 표기**: `실제값/한계값 × 100%`
4. **여유(margin) 표기**: `한계 - 실제` (양수=여유, 음수=초과)
5. **원인 분류**: `weight_exceeds_ub`, `tile_too_large`, `sm_static_exceed`, `sm_peak_exceed`

### 수치 표기

| 범위 | 표기 |
|:---|:---|
| < 1 KiB | bytes 그대로 (예: 432 B) |
| 1 KiB ~ 1 MiB | KiB 변환 + bytes 병기 (예: 144 KiB = 147,456 B) |
| ≥ 1 MiB | MiB 변환 소수점 2자리 (예: 2.19 MB) |
| 비율 | 소수점 1자리 % (예: 67.0%) |

### FAIL layer 계산 전개 예시

```
Layer [47] model.8.m.0.cv1 (Conv 3×3, stride=1)

  Weight:      147,456 B  (C_in=128, C_out=128, K=3×3, i8 → 128×128×3×3×1)
  Input tile:  34 × 34 × 128 = 147,968 B
               (in_h = S×T + K - S = 1×32 + 3 - 1 = 34)
  Output tile: 32 × 32 × 128 = 131,072 B
  Metadata:    512 B

  Total:       147,456 + 147,968 + 131,072 + 512 = 427,008 B
  Budget:      131,072 B (UB_EFF = 128 KiB)
  초과:        295,936 B (225.8%)

  원인: weight(147,456) 단독으로 UB_EFF(131,072)를 초과
       → weight_exceeds_ub
       → Weight streaming 미지원 시 해결 불가
```

---

## CSV 출력 정의

분석 결과와 함께 아래 CSV 파일을 생성한다.

### feasibility_summary.csv

```
Model, Tiling, TotalTileOps, Q_WorstCore, Q_Max, Q_Verdict, UB_Pass, UB_Fail, UB_Verdict, SM_Verdict, SM_PeakFail, Overall
```

### queue_analysis.csv

```
Model, Tiling, Core, LayerRange, ConvLayers, NonConvOps, TileOps, InstrRecords, Q_Max, Verdict
```

### ub_analysis.csv

```
Model, Tiling, Index, Name, Type, Kernel, Stride, InputShape, OutputShape, Weight_B, Remaining_for_tiles, InTile_HW, InTile_B, OutTile_HW, OutTile_B, TileIO_B, Meta_B, TotalUB, Budget, WeightFitsUB, FailReason, Verdict
```

### sm_analysis.csv

```
Model, Check, WeightBytes, FMBytes, Total, SM_Size, Verdict
```

### safe_boundary.csv

```
Model, Index, Core, Name, Type, Kernel, Stride, OutHW, OutC, InC, CurWeight, MaxWeight, Margin, WeightOK, MinT_Q, Tiles_Q, TileShare, InTile_HW, InTile_B, OutTile_B, T_AdaptFull, Tiles_AdaptFull
```
