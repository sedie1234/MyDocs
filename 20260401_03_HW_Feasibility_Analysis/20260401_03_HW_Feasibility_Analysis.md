# Queue / UB / SM 동작 가능 여부 분석 보고서 (v3)

생성일: 2026-04-01 | 이전 버전: v1(Baseline), v2(Adaptive Full 추가)

## 1. 분석 조건 요약

| 항목 | 값 |
|:---|:---|
| 대상 모델 | YOLOv10n (83 conv layers, 2.19 MB weight) / YOLOv10s (87 conv layers, 6.90 MB weight) |
| 추론 경로 | backbone + neck + one2one detect head (one2many 제외) |
| 코어 구성 | 12 pCore only (aCore 완전 제외) |
| **실행 모델** | **Sequential** (Core0 -> Core1 -> ... -> Core11 파이프라인) |
| Per-core Q | 128 KiB = 131,072 bytes (256B records x 512 max) |
| **Per-core UB (pUB)** | **256 KiB = 262,144 bytes** (ping-pong 시 half = 128 KiB) |
| SM (공유) | 4 MiB = 4,194,304 bytes (12코어 공유) |
| **SM overlap** | **Input/Output FM이 동일 SM 공간 공유 가능** (peak = weight + max(in_FM, out_FM)) |
| eFlash | 없음 (aCore 전용 자원, 분석 제외) |
| **UB 모델** | **Weight-first** (weight 전량 UB 적재, 잔여 공간에 input 슬라이스) |
| 타일링 전략 | (1) 16x16 고정, (2) 32x32 고정, (3) Adaptive Half (128 KiB), (4) Adaptive Full (256 KiB) |
| Weight 로딩 | 추론 전 1회 SM에 적재 (weight stationary), 추론 중 재적재 없음 |
| 입력 이미지 | 640x640x3 i8 = 1,228,800 bytes (1.17 MB), SM에 1회 적재 |
| 데이터 타입 | Activation: i8 (1B), Weight: s8 (1B) |
| 불균형 마진 | 1.5x (코어 분배 불균형 대비) |
| **판정 기준** | **PASS / FAIL only** (COND 폐지) |
| **분석 케이스** | **8 cases = 2 models x 4 tiling strategies** |

## 2. 이전 버전 대비 변경사항 (v1 -> v2 -> v3)

| 항목 | v1 (Baseline) | v2 | **v3** |
|:---|:---|:---|:---|
| UB 크기 | 128 KiB | 128 KiB | **256 KiB** |
| UB half (ping-pong) | 64 KiB | 64 KiB | **128 KiB** |
| 실행 모델 | 병렬 (균등 분배) | 병렬 (균등 분배) | **Sequential** (Core0->Core11) |
| Q 분석 방법 | total / 12 x 1.5 | total / 12 x 1.5 | **코어별 레이어 할당 후 직접 계산** |
| SM overlap | 불가 (in + out 동시) | 불가 (in + out 동시) | **가능** (peak = weight + max(in, out)) |
| UB 모델 | tile IO + weight | tile IO + weight | **Weight-first** (weight 전량 -> 잔여에 input slice) |
| Halo buffer | 미고려 | 미고려 | **SM overlap 시 halo buffer 요구량 계산** |
| 판정 기준 | PASS/COND/FAIL | PASS/COND/FAIL | **PASS/FAIL only** |
| 타일링 케이스 | 3개 | 4개 (+ Adaptive Full) | 4개 (동일) |
| Adaptive Half budget | 64 KiB | 64 KiB | **128 KiB** |
| Adaptive Full budget | N/A | 128 KiB | **256 KiB** |

**v3 핵심 변경의 영향:**
- **UB 256 KiB**: v2에서 weight > 128 KiB로 FAIL이던 레이어들이 해소됨 (v10n: 4개 FAIL -> 0개)
- **Sequential 실행 모델**: 레이어를 코어에 순차 할당하므로 코어별 부하가 불균형. 초기 고해상도 레이어가 Core0에 집중
- **SM overlap**: input/output FM이 공간을 공유하여 peak SM 사용량 감소. v10n에서 SM FAIL 해소
- **Halo buffer**: Stride-2 conv 등에서 overlap 전환 시 halo(경계 행) 데이터를 UB에 백업 필요

## 3. 하드웨어 자원 요약

### 3.1 Instruction Queue (Q)
- 크기: 128 KiB per pCore = 131,072 bytes
- 256-byte 고정 크기 instruction record
- 최대 512 records per core
- Command types: LAYER_START, LAYER_RUN, LAYER_END, LOOP_JUMP, EPOCH_COMMIT, STOP, NOP

### 3.2 Unified Buffer (UB / pUB)
- 크기: **256 KiB** per pCore = 262,144 bytes
- 16-bank SRAM, ping-pong windowing 지원
- Ping-pong (half): **128 KiB** 실효 사용량
- Single-buffer (full): **256 KiB** 실효 사용량 (Adaptive Full에서 사용)
- **Weight-first 모델**: weight 전량 적재 후 잔여 공간에 input tile + output tile + metadata

### 3.3 Shared Memory (SM)
- 크기: 4 MiB = 4,194,304 bytes
- 16 bank, NoC 경유 접근
- 용도: Weight 상주, 입력 이미지, 코어 간 중간 데이터(feature map)
- **Overlap 정책**: Input FM과 Output FM이 동일 SM 공간 공유 가능
  - Peak = Weight + max(Input_FM, Output_FM)
  - Halo buffer 필요 시 UB에 백업

## 4. 모델 요약

### 4.1 YOLOv10n

| 항목 | 값 |
|:---|:---|
| Conv layers (inference) | 83 (Standard=22, 1x1=40, DW=21) |
| 총 weight (inference) | 2,292,576 bytes = 2.19 MB |
| 비Conv ops (추정) | ~30 (add, concat, pool, matmul, softmax, resize 등) |
| 총 ops (추정) | ~113 |
| Max channel width | 256 (backbone/neck), 80 (head cv3) |
| 최대 단일 weight | 147,456 bytes = 144.0 KB (Conv 3x3, C=128->128) |

**해상도별 레이어 분포 (v10n, 83 conv layers):**

| 출력 해상도 | 레이어 수 | 대표 채널 범위 |
|:---|---:|:---|
| 320x320 | 1 | C_out=16 |
| 160x160 | 5 | C=16~48 |
| 80x80 | 19 | C=32~192 |
| 40x40 | 25 | C=64~384 |
| 20x20 | 33 | C=80~512 |
| **합계** | **83** | |

### 4.2 YOLOv10s

| 항목 | 값 |
|:---|:---|
| Conv layers (inference) | 87 (Standard=22, 1x1=41, DW=24) |
| 총 weight (inference) | 7,240,032 bytes = 6.90 MB |
| 비Conv ops (추정) | ~30 |
| 총 ops (추정) | ~117 |
| Max channel width | 512 (backbone/neck), 128 (head cv3) |
| 최대 단일 weight | 524,288 bytes = 512.0 KB (1x1, C=1024->512) |

**해상도별 레이어 분포 (v10s, 87 conv layers):**

| 출력 해상도 | 레이어 수 | 대표 채널 범위 |
|:---|---:|:---|
| 320x320 | 1 | C_out=32 |
| 160x160 | 5 | C=32~96 |
| 80x80 | 19 | C=64~384 |
| 40x40 | 25 | C=64~768 |
| 20x20 | 37 | C=80~1024 |
| **합계** | **87** | |

## 5. Queue 분석

### 5.1 Sequential 코어 할당

v3에서는 병렬 균등 분배 대신 **Sequential 실행 모델**을 적용한다. 레이어를 Core 번호 순으로 할당:
- v10n: 83 layers / 12 cores = 7 layers/core (마지막 Core11은 6 layers)
- v10s: 87 layers / 12 cores = 8 or 7 layers/core

각 코어의 instruction records = conv tile-ops + non-conv ops + overhead (LAYER_START/END per layer)

### 5.2 타일링별 결과

**종합 결과표:**

| Model | Tiling | Total TileOps | Worst Core | Worst Records | Q Max (512) | 판정 |
|:---|:---|---:|:---|---:|---:|:---|
| v10n | 16x16 | 3,788 | Core0 (L0-6) | 1,150 | 512 | **FAIL** |
| v10n | 32x32 | 1,282 | Core0 (L0-6) | 320 | 512 | **PASS** |
| v10n | Adaptive Half | 15,622 | Core9 (L63-69) | 6,577 | 512 | **FAIL** |
| **v10n** | **Adaptive Full** | **1,219** | **Core5 (L35-41)** | **236** | **512** | **PASS** |
| v10s | 16x16 | 3,774 | Core0 (L0-7) | 1,177 | 512 | **FAIL** |
| v10s | 32x32 | 1,261 | Core0 (L0-7) | 331 | 512 | **PASS** |
| v10s | Adaptive Half | 112,353 | Core5 (L38-44) | 27,954 | 512 | **FAIL** |
| v10s | Adaptive Full | 11,491 | Core9 (L66-72) | 6,621 | 512 | **FAIL** |

### 5.3 Core별 상세 (v10n, 4 tiling 전략)

**v10n 16x16 (FAIL — worst Core0: 1,150 records):**

| Core | Layer Range | Conv Layers | NonConv Ops | Tile Ops | Instr Records | 판정 |
|:---|:---|---:|---:|---:|---:|:---|
| Core0 | 0-6 | 7 | 5 | 1,125 | 1,150 | **FAIL** |
| Core1 | 7-13 | 7 | 4 | 275 | 298 | PASS |
| Core2 | 14-20 | 7 | 10 | 153 | 188 | PASS |
| Core3 | 21-27 | 7 | 25 | 183 | 248 | PASS |
| Core4 | 28-34 | 7 | 15 | 88 | 133 | PASS |
| Core5 | 35-41 | 7 | 29 | 340 | 413 | PASS |
| Core6 | 42-48 | 7 | 14 | 285 | 328 | PASS |
| Core7 | 49-55 | 7 | 25 | 183 | 248 | PASS |
| Core8 | 56-62 | 7 | 29 | 346 | 419 | PASS |
| Core9 | 63-69 | 7 | 29 | 330 | 403 | PASS |
| Core10 | 70-76 | 7 | 14 | 301 | 344 | PASS |
| Core11 | 77-82 | 6 | 25 | 179 | 242 | PASS |

> Core0 FAIL 원인: Layer 0 (320x320 output, 400 tiles)이 전체의 35%를 차지

**v10n Adaptive Full (PASS — worst Core5: 236 records):**

| Core | Layer Range | Conv Layers | NonConv Ops | Tile Ops | Instr Records | 판정 |
|:---|:---|---:|---:|---:|---:|:---|
| Core0 | 0-6 | 7 | 5 | 119 | 144 | PASS |
| Core1 | 7-13 | 7 | 4 | 54 | 77 | PASS |
| Core2 | 14-20 | 7 | 10 | 111 | 146 | PASS |
| Core3 | 21-27 | 7 | 25 | 119 | 184 | PASS |
| Core4 | 28-34 | 7 | 15 | 73 | 118 | PASS |
| Core5 | 35-41 | 7 | 29 | 163 | 236 | PASS |
| Core6 | 42-48 | 7 | 14 | 83 | 126 | PASS |
| Core7 | 49-55 | 7 | 25 | 119 | 184 | PASS |
| Core8 | 56-62 | 7 | 29 | 93 | 166 | PASS |
| Core9 | 63-69 | 7 | 29 | 102 | 175 | PASS |
| Core10 | 70-76 | 7 | 14 | 119 | 162 | PASS |
| Core11 | 77-82 | 6 | 25 | 64 | 127 | PASS |

> Worst core = Core5 (236/512), 여유 53.9%. 모든 코어 PASS.

**v10s Core별 상세 (Adaptive Full, FAIL):**

| Core | Layer Range | Conv Layers | NonConv Ops | Tile Ops | Instr Records | 판정 |
|:---|:---|---:|---:|---:|---:|:---|
| Core0 | 0-7 | 8 | 5 | 223 | 250 | PASS |
| Core1 | 8-15 | 8 | 14 | 249 | 294 | PASS |
| Core2 | 16-23 | 8 | 25 | 632 | 699 | **FAIL** |
| Core3 | 24-30 | 7 | 15 | 433 | 478 | PASS |
| Core4 | 31-37 | 7 | 15 | 877 | 922 | **FAIL** |
| Core5 | 38-44 | 7 | 29 | 957 | 1,030 | **FAIL** |
| Core6 | 45-51 | 7 | 14 | 226 | 269 | PASS |
| Core7 | 52-58 | 7 | 25 | 581 | 646 | **FAIL** |
| Core8 | 59-65 | 7 | 19 | 522 | 575 | **FAIL** |
| Core9 | 66-72 | 7 | 29 | 6,548 | 6,621 | **FAIL** |
| Core10 | 73-79 | 7 | 14 | 124 | 167 | PASS |
| Core11 | 80-86 | 7 | 25 | 119 | 184 | PASS |

> v10s worst = Core9 (6,621 records). Weight > UB인 레이어에서 T=1 -> tile 폭발.

## 6. Unified Buffer 분석

### 6.1 Weight-first 모델 설명

v3에서는 **Weight-first UB 모델**을 적용:

```
1. Weight 전량을 UB에 적재 (K*K*Cin*Cout for standard conv)
2. Remaining = UB_budget - Weight - Metadata(512B)
3. Remaining 공간에 input_tile + output_tile을 fit
4. Weight > UB_budget이면 해당 레이어는 즉시 FAIL (weight_exceeds_ub)
5. Weight <= UB이나 잔여 공간에 tile이 안 들어가면 FAIL (tile_too_large)
```

UB budget:
- Ping-pong (16x16, 32x32, Adaptive Half): **128 KiB** (half of 256 KiB)
- Single-buffer (Adaptive Full): **256 KiB**

### 6.2 타일링별 결과

| Model | Tiling | UB Budget | PASS | FAIL | FAIL 원인 분포 | 판정 |
|:---|:---|---:|---:|---:|:---|:---|
| v10n | 16x16 | 262,144 | 78 | 5 | tile_too_large: 5 | **FAIL** |
| v10n | 32x32 | 262,144 | 59 | 24 | tile_too_large: 24 | **FAIL** |
| v10n | Adaptive Half | 131,072 | 79 | 4 | tile_too_large: 4 | **FAIL** |
| **v10n** | **Adaptive Full** | **262,144** | **83** | **0** | **-** | **PASS** |
| v10s | 16x16 | 262,144 | 56 | 31 | weight_exceeds_ub: 5, tile_too_large: 26 | **FAIL** |
| v10s | 32x32 | 262,144 | 28 | 59 | weight_exceeds_ub: 5, tile_too_large: 54 | **FAIL** |
| v10s | Adaptive Half | 131,072 | 58 | 29 | tile_too_large: 29 | **FAIL** |
| v10s | Adaptive Full | 262,144 | 79 | 8 | weight_exceeds_ub: 8 | **FAIL** |

### 6.3 v10n Adaptive Full — 전 레이어 PASS

v2에서 FAIL이던 4개 레이어가 UB 256 KiB로 전부 해소:

| Layer | Name | Weight (B) | v2 (UB 128K) | v3 (UB 256K) | Remaining | Tile | 변화 |
|---:|:---|---:|:---|:---|---:|:---|:---|
| 24 | model.8.m.0.cv1.conv | 147,456 | FAIL (wt>128K) | PASS | 114,176 | 22x22 in / 20x20 out | 해소 |
| 25 | model.8.m.0.cv2.conv | 147,456 | FAIL (wt>128K) | PASS | 114,176 | 22x22 in / 20x20 out | 해소 |
| 28 | model.9.cv2.conv | 131,072 | FAIL (wt+meta>128K) | PASS | 130,560 | 13x13 in / 13x13 out | 해소 |
| 65 | one2one_cv2.2.0 | 147,456 | FAIL (wt>128K) | PASS | 114,176 | 19x19 in / 17x17 out | 해소 |

**v10n 최대 UB 사용 레이어 (Adaptive Full):**

| Layer | Name | Weight (B) | TileIO (B) | Meta | Total UB | Budget | 잔여 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 24 | model.8.m.0.cv1.conv | 147,456 | 113,152 | 512 | 261,120 | 262,144 | 1,024 |
| 25 | model.8.m.0.cv2.conv | 147,456 | 113,152 | 512 | 261,120 | 262,144 | 1,024 |
| 65 | one2one_cv2.2.0 | 147,456 | 110,912 | 512 | 258,880 | 262,144 | 3,264 |
| 3 | model.2.m.0.cv1.conv | 2,304 | 259,232 | 512 | 262,048 | 262,144 | 96 |

> Layer 3이 잔여 96B로 가장 빡빡하지만 PASS.

### 6.4 v10s Adaptive Full — FAIL 레이어 상세 (weight > 256 KiB)

| Layer | Name | Type | Weight (B) | Weight (KB) | UB Budget | 초과량 |
|---:|:---|:---|---:|---:|---:|---:|
| 23 | model.8.cv1.conv | 1x1 | 262,144 | 256.0 | 262,144 | +1,536 (meta) |
| 30 | model.8.cv2.conv | 1x1 | 393,216 | 384.0 | 262,144 | +132,864 |
| 32 | model.9.cv2.conv | 1x1 | 524,288 | 512.0 | 262,144 | +264,192 |
| 33 | model.10.cv1.conv | 1x1 | 262,144 | 256.0 | 262,144 | +1,536 (meta) |
| 39 | model.10.cv2.conv | 1x1 | 262,144 | 256.0 | 262,144 | +1,536 (meta) |
| 55 | model.22.cv1.conv | 1x1 | 393,216 | 384.0 | 262,144 | +132,864 |
| 62 | model.22.cv2.conv | 1x1 | 393,216 | 384.0 | 262,144 | +132,864 |
| 69 | one2one_cv2.2.0 | Conv 3x3 | 294,912 | 288.0 | 262,144 | +37,952 |

> 8개 레이어 모두 weight 자체가 UB(256 KiB)를 초과하거나, weight + metadata가 초과. 최대 초과는 Layer 32 (weight 512 KB = UB의 2배).

### 6.5 v10n 16x16 고정 타일 FAIL 상세 (5개)

| Layer | Name | Type | Weight (B) | Tile Size | TotalUB | Budget | FAIL 원인 |
|---:|:---|:---|---:|:---|---:|---:|:---|
| 22 | model.7.cv2.conv | DW 3x3 s2 | 2,304 | 33x33 in / 16x16 out | 347,136 | 262,144 | tile_too_large |
| 26 | model.8.cv2.conv | 1x1 | 98,304 | 16x16 in / 16x16 out | 262,656 | 262,144 | tile_too_large (+512B) |
| 28 | model.9.cv2.conv | 1x1 | 131,072 | 16x16 in / 16x16 out | 328,192 | 262,144 | tile_too_large |
| 51 | model.22.cv1.conv | 1x1 | 98,304 | 16x16 in / 16x16 out | 262,656 | 262,144 | tile_too_large (+512B) |
| 58 | model.22.cv2.conv | 1x1 | 98,304 | 16x16 in / 16x16 out | 262,656 | 262,144 | tile_too_large (+512B) |

## 7. Shared Memory 분석

### 7.1 정적 할당 (Weight + Input)

| 항목 | v10n | v10s |
|:---|---:|---:|
| Weight (s8) | 2,292,576 B (2.19 MB) | 7,240,032 B (6.90 MB) |
| Input image (640x640x3 i8) | 1,228,800 B (1.17 MB) | 1,228,800 B (1.17 MB) |
| **정적 합계 (no overlap)** | **3,521,376 B (3.36 MB)** | **8,468,832 B (8.08 MB)** |
| SM 용량 (4 MiB) | 4,194,304 B | 4,194,304 B |
| **정적 적합 여부** | **PASS** | **FAIL** (weight 단독 6.90 MB > 4 MB) |

### 7.2 Peak with Overlap

v3에서는 Input FM과 Output FM이 SM 공간을 공유할 수 있으므로:

```
peak_SM = total_weight + max(input_FM, output_FM)
```

이전 버전(v1/v2)에서는 `peak_SM = total_weight + input_FM + output_FM` 이었으므로, overlap으로 peak가 크게 감소한다.

**v10n SM Peak 분석 (overlap 적용, 전 레이어 PASS):**

| Layer | Name | Input FM (B) | Output FM (B) | max(in,out) | Peak (B) | Peak (MB) | SM | 잔여 | 판정 |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| 0 | model.0.conv | 1,228,800 | 1,638,400 | 1,638,400 | 3,930,976 | 3.75 | 4,194,304 | 263,328 | PASS |
| 1 | model.1.conv | 1,638,400 | 819,200 | 1,638,400 | 3,930,976 | 3.75 | 4,194,304 | 263,328 | PASS |
| 5 | model.2.cv2.conv | 1,228,800 | 819,200 | 1,228,800 | 3,521,376 | 3.36 | 4,194,304 | 672,928 | PASS |
| 12 | model.4.cv2.conv | 819,200 | 409,600 | 819,200 | 3,111,776 | 2.97 | 4,194,304 | 1,082,528 | PASS |
| 82 | one2one_cv3.2.2 | 32,000 | 32,000 | 32,000 | 2,324,576 | 2.22 | 4,194,304 | 1,869,728 | PASS |

> **핵심 변화**: v2에서 FAIL이던 Layer 0, 1, 5가 overlap 적용으로 전부 PASS.
> - Layer 0: v2 peak 5,159,776 -> v3 peak 3,930,976 (1.23 MB 감소)
> - Layer 1: v2 peak 4,749,776 -> v3 peak 3,930,976 (0.82 MB 감소)
> - Layer 5: v2 peak 4,340,576 -> v3 peak 3,521,376 (0.82 MB 감소)

**v10n 전체 SM 결과: 83/83 PASS (peak fail = 0)**

**v10s SM 분석: 전 레이어 FAIL (87/87)**

v10s는 weight 단독(6.90 MB)이 SM(4 MB)을 초과하므로, overlap 여부와 무관하게 전 레이어 FAIL:

| Layer | Name | max(in,out) | Peak (B) | SM | 판정 |
|---:|:---|---:|---:|---:|:---|
| 0 | model.0.conv | 4,505,600 | 10,516,832 | 4,194,304 | FAIL |
| 70 | model.23.one2one_cv3.2.0.1 | 51,200 | 7,265,632 | 4,194,304 | FAIL |
| 86 | (최소 FM) | 83,200 | 7,291,232 | 4,194,304 | FAIL |

> v10s 최소 peak = 7,265,632 B (6.93 MB). SM의 1.73배.

### 7.3 Halo buffer 요구량

SM overlap 모드에서 stride-2 conv 등의 경우, output이 input 영역을 덮어쓰기 전에 다음 타일의 halo(경계) 행을 UB에 백업해야 한다.

**v10n Halo buffer 상위 레이어:**

| Layer | Name | Type | Halo Rows | Halo Bytes | 비고 |
|---:|:---|:---|---:|---:|:---|
| 54 | model.22.m.0.cv1.2.conv.conv | DW 7x7 s1 | 6 | 30,720 (30 KB) | **v10n 최대** |
| 14 | model.5.cv2.conv | DW 3x3 s2 | 2 | 20,480 (20 KB) | |
| 22 | model.7.cv2.conv | DW 3x3 s2 | 2 | 20,480 (20 KB) | |
| 1 | model.1.conv | Conv 3x3 s2 | 2 | 10,240 (10 KB) | |
| 6 | model.3.conv | Conv 3x3 s2 | 2 | 10,240 (10 KB) | |
| 0 | model.0.conv | Conv 3x3 s2 | 2 | 3,840 (3.75 KB) | C_in=3이라 작음 |

> v10n 최대 halo = **30 KB** (Layer 54, DW 7x7). UB 256 KiB 대비 11.7% 사용.

**v10s Halo buffer 상위 레이어:**

| Layer | Name | Type | Halo Rows | Halo Bytes | 비고 |
|---:|:---|:---|---:|---:|:---|
| 26 | model.8.m.0.cv1.2.conv.conv | DW 7x7 s1 | 6 | 61,440 (60 KB) | **v10s 최대** |
| 54 | model.22.m.0.cv1.2.conv.conv | DW 7x7 s1 | 6 | 61,440 (60 KB) | |
| 14 | model.5.cv2.conv | DW 3x3 s2 | 2 | 40,960 (40 KB) | |
| 22 | model.7.cv2.conv | DW 3x3 s2 | 2 | 40,960 (40 KB) | |
| 1 | model.1.conv | Conv 3x3 s2 | 2 | 20,480 (20 KB) | |

> v10s 최대 halo = **60 KB** (DW 7x7, C=512). UB 256 KiB 대비 23.4%.
> Halo buffer는 output 기록 전에 UB로 pre-copy 필요. Weight-first 모델에서 weight + halo가 UB에 동시 상주해야 하므로, 대형 weight + 대형 halo 레이어에서 UB 압박 가능.

## 8. 종합 교차 분석

### 8.1 8-Case 판정 매트릭스

| Model | Tiling | Q (512 rec) | UB (256 KiB) | SM (4 MiB, overlap) | **종합** |
|:---|:---|:---|:---|:---|:---|
| v10n | 16x16 | **FAIL** (Core0: 1,150) | **FAIL** (5 fail) | PASS | **FAIL** |
| v10n | 32x32 | **PASS** (Core0: 320) | **FAIL** (24 fail) | PASS | **FAIL** |
| v10n | Adaptive Half | **FAIL** (Core9: 6,577) | **FAIL** (4 fail) | PASS | **FAIL** |
| **v10n** | **Adaptive Full** | **PASS** (Core5: 236) | **PASS** (83/83) | **PASS** | **PASS** |
| v10s | 16x16 | **FAIL** (Core0: 1,177) | **FAIL** (31 fail) | **FAIL** (wt>SM) | **FAIL** |
| v10s | 32x32 | **PASS** (Core0: 331) | **FAIL** (59 fail) | **FAIL** (wt>SM) | **FAIL** |
| v10s | Adaptive Half | **FAIL** (Core5: 27,954) | **FAIL** (29 fail) | **FAIL** (wt>SM) | **FAIL** |
| v10s | Adaptive Full | **FAIL** (Core9: 6,621) | **FAIL** (8 fail, wt>UB) | **FAIL** (wt>SM) | **FAIL** |

### 8.2 종합 판정 요약

| Model | Tiling | 종합 |
|:---|:---|:---|
| v10n | 16x16 | FAIL |
| v10n | 32x32 | FAIL |
| v10n | Adaptive Half | FAIL |
| **v10n** | **Adaptive Full** | **PASS** |
| v10s | 16x16 | FAIL |
| v10s | 32x32 | FAIL |
| v10s | Adaptive Half | FAIL |
| v10s | Adaptive Full | FAIL |

> **v10n Adaptive Full이 8개 케이스 중 유일한 PASS.**

### 8.3 PASS 조건 분석 (v10n Adaptive Full)

| 자원 | 수치 | 한계 | 여유율 |
|:---|:---|:---|:---|
| Q | worst Core5: 236 records | 512 records | 53.9% |
| UB | worst Layer 3: 262,048 B | 262,144 B | 0.04% (96 B) |
| SM | worst Layer 0: 3,930,976 B | 4,194,304 B | 6.3% (257 KB) |

> **UB가 가장 빡빡한 자원** (Layer 3에서 96 B 여유). Q는 충분한 여유.

## 9. 버전별 비교 (v1 -> v2 -> v3)

### 9.1 조건 변경 이력

| 항목 | v1 | v2 | v3 |
|:---|:---|:---|:---|
| UB 크기 | 128 KiB | 128 KiB | **256 KiB** |
| 실행 모델 | 병렬 균등 | 병렬 균등 | **Sequential** |
| SM overlap | 불가 | 불가 | **가능** |
| UB 모델 | tile IO + weight | tile IO + weight | **Weight-first** |
| 판정 기준 | PASS/COND/FAIL | PASS/COND/FAIL | **PASS/FAIL** |
| Adaptive Full budget | N/A | 128 KiB | **256 KiB** |
| Adaptive Half budget | 64 KiB | 64 KiB | **128 KiB** |

### 9.2 v10n 결과 비교

| 항목 | v1 | v2 | v3 |
|:---|:---|:---|:---|
| 최선 종합 | 6/6 FAIL | **COND** (Adaptive Full) | **PASS** (Adaptive Full) |
| Q 최선 | PASS (16x16, 343 avg) | PASS (Adaptive Full, 326 avg) | **PASS** (Adaptive Full, Core5: 236) |
| UB FAIL 최소 | 11개 (Adaptive Half) | 4개 (Adaptive Full) | **0개** (Adaptive Full) |
| SM FAIL | 3개 (Layer 0,1,5) | 3개 (Layer 0,1,5) | **0개** (overlap 해소) |

### 9.3 v10s 결과 비교

| 항목 | v1 | v2 | v3 |
|:---|:---|:---|:---|
| 최선 종합 | FAIL | FAIL | **FAIL** |
| SM | FAIL (wt 6.90 > 4 MB) | FAIL | **FAIL** (구조적 한계) |
| UB FAIL 최소 | 38개 | 29개 (Adaptive Full) | **8개** (Adaptive Full, wt>UB) |
| Q 최선 | PASS (32x32, 125 avg) | PASS (32x32, 125 avg) | **PASS** (32x32, Core0: 331) |

### 9.4 변경사항별 영향 분석

| 변경사항 | v10n 영향 | v10s 영향 |
|:---|:---|:---|
| UB 128K -> 256K | UB FAIL 4->0 (v2 FAIL 레이어 전부 해소) | UB FAIL 29->8 (wt 128~256K 범위 해소, >256K 잔존) |
| Sequential 실행 | Q 분석이 코어별 직접 계산으로 정밀화. Core0 불균형 노출 | Core0 불균형 동일 |
| SM overlap | SM FAIL 3->0 (peak 계산식 변경) | 영향 없음 (wt > SM) |
| Weight-first UB | 동일 결과 (명시적 모델링) | 동일 결과 |
| COND 폐지 | v2 COND -> v3 PASS (실질 개선) | 변화 없음 (전부 FAIL) |

## 10. 결론 및 권고사항

### 10.1 주요 결론

1. **v10n Adaptive Full = PASS.** 8개 분석 케이스 중 유일한 완전 통과. UB 256 KiB + SM overlap이 결합되어 v2의 잔여 이슈(UB 4개 FAIL, SM 3개 FAIL)가 모두 해소.

2. **v10n의 여유는 극소.** UB에서 Layer 3이 96 B 여유, SM에서 Layer 0이 257 KB 여유. 실 구현 시 metadata 오버헤드 관리가 중요.

3. **v10s는 모든 조합에서 FAIL.** 근본 원인은 weight(6.90 MB) > SM(4 MB). UB 256 KiB에서도 8개 레이어의 weight가 UB를 초과(최대 512 KB). SM 확장 또는 weight streaming 없이는 해결 불가.

4. **Sequential 실행 모델에서 Core0 불균형 확인.** 16x16 타일링 시 Core0에 고해상도 레이어(320x320 output)가 집중되어 Q FAIL. Adaptive Full에서는 타일 수가 충분히 적어 문제 없음.

5. **Halo buffer는 관리 가능 수준.** v10n 최대 30 KB, v10s 최대 60 KB. UB 대비 12~23% 수준으로, weight-first 모델에서 추가 UB 압박은 제한적.

### 10.2 v10n 실현 시나리오

```
전략: Adaptive Full (256 KiB single buffer) + SM overlap

Q:  PASS — worst Core5: 236/512 records (여유 53.9%)
UB: PASS — worst Layer 3: 262,048/262,144 B (여유 96 B)
SM: PASS — worst Layer 0: 3,930,976/4,194,304 B (여유 257 KB)

Trade-off: Ping-pong 미사용으로 DMA-compute overlap 불가.
          각 타일에서 DMA 완료 후 compute 시작 (latency 증가).
          그러나 전 레이어 PASS하는 유일한 조합.
```

### 10.3 v10s 해결을 위한 필수 변경

| 병목 | 현재 | 필요 조건 |
|:---|:---|:---|
| SM | 4 MiB | **8 MiB 이상** (weight 6.90 + input 1.17 + FM headroom) |
| UB (8개 FAIL) | 256 KiB | **512 KiB** 또는 weight tiling/streaming |
| Q (Adaptive Full) | 512 records | Q 재로딩 허용 또는 weight streaming으로 tile 수 감소 |

### 10.4 권고사항

**단기 (현재 아키텍처 내):**
1. **v10n: Adaptive Full 전략 확정.** 유일한 PASS 케이스. 컴파일러에서 레이어별 adaptive tile 크기 계산 로직 구현 필요.
2. **Halo buffer 관리 로직 구현.** SM overlap 전환 시 halo pre-copy를 UB로 수행하는 DMA 시퀀스 필요.
3. **UB metadata 512 B 검증.** Layer 3에서 96 B 여유이므로 실제 metadata 오버헤드가 512 B 이하인지 확인 필수.

**중기 (아키텍처 수정 시):**
1. **SM 확장 (8 MiB -> 16 MiB)**: v10s weight 수용 가능. 8 MiB에서도 weight(6.90) + FM peak이 빡빡하므로 16 MiB 권장.
2. **UB 확장 (512 KiB)**: v10s의 weight > UB 레이어 8개 -> 1개(Layer 32, 512 KB weight)로 감소.
3. **Weight streaming**: weight를 SM에서 직접 참조하거나 slice 단위 DMA로 UB 부담 해소. v10s 해결의 핵심.
