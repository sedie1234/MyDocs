# Queue / UB / SM 동작 가능 여부 분석 보고서 (v2)

생성일: 2026-04-01 | Baseline: 20260401_Queue_UB_SM_Feasibility_Analysis.md

## 1. 분석 조건 요약

| 항목 | 값 |
|:---|:---|
| 대상 모델 | YOLOv10n (83 conv layers, 2.19 MB weight) / YOLOv10s (87 conv layers, 6.90 MB weight) |
| 추론 경로 | backbone + neck + one2one detect head (one2many 제외) |
| 코어 구성 | 12 pCore only (aCore 완전 제외) |
| Per-core Q | 128 KiB = 131,072 bytes (256B records x 512 max) |
| Per-core UB (pUB) | 128 KiB = 131,072 bytes (ping-pong 시 half = 64 KiB) |
| SM (공유) | 4 MiB = 4,194,304 bytes (12코어 공유) |
| eFlash | 없음 (aCore 전용 자원, 분석 제외) |
| 타일링 전략 | (1) 16x16 고정, (2) 32x32 고정, (3) Adaptive Half (64 KiB), (4) **Adaptive Full (128 KiB)** |
| Weight 로딩 | 추론 전 1회 SM에 적재, 추론 중 재적재 없음. 타일 연산 시 필요한 전체 weight를 UB로 이동. |
| 입력 이미지 | 640x640x3 i8 = 1,228,800 bytes (1.17 MB), SM에 1회 적재 |
| 데이터 타입 | Activation: i8 (1B), Weight: s8 (1B) |
| 불균형 마진 | 1.5x (코어 분배 불균형 대비) |
| Instruction 오버헤드 | LAYER_START, LAYER_END, EPOCH_COMMIT, STOP, NOP 포함 |
| 비포함 오버헤드 | Semaphore, Profiling |
| **분석 케이스** | **8 cases = 2 models x 4 tiling strategies** |

## 2. Baseline 대비 변경사항

| 항목 | Baseline (v1) | v2 |
|:---|:---|:---|
| 타일링 전략 수 | 3 (16x16, 32x32, Adaptive Half) | **4** (+ Adaptive Full) |
| 총 분석 케이스 | 6 (2 models x 3 tilings) | **8** (2 models x 4 tilings) |
| Adaptive Full 제약 | 없음 | input_tile + output_tile + weight + 512 <= **128 KiB** |
| Adaptive Full 특성 | 없음 | Ping-pong 미사용, UB 전체(128 KiB)를 single buffer로 활용 |
| Weight 정책 | 동일 | 명시적 확인: tile 계산에 필요한 전체 weight를 UB로 이동 (standard conv는 K*K*Cin*Cout 전량) |

**Adaptive Full의 의미**: Ping-pong double-buffering을 포기하고 UB 전체 128 KiB를 하나의 작업 버퍼로 사용. 이로 인해 DMA overlap이 불가하여 latency가 증가하지만, 더 큰 타일을 사용할 수 있어 총 tile 수가 감소하고 Q 사용량이 줄어듦.

## 3. 하드웨어 자원 요약

### 3.1 Instruction Queue (Q)
- 크기: 128 KiB per pCore = 131,072 bytes
- 256-byte 고정 크기 instruction record
- 최대 512 records per core
- Command types: LAYER_START, LAYER_RUN (메인 연산), LAYER_END, LOOP_JUMP, EPOCH_COMMIT, STOP, NOP

### 3.2 Unified Buffer (UB / pUB)
- 크기: 128 KiB per pCore = 131,072 bytes
- 16-bank SRAM, ping-pong windowing 지원
- Ping-pong (half): 64 KiB 실효 사용량
- **Single-buffer (full): 128 KiB 실효 사용량** (Adaptive Full에서 사용)
- 타일 연산 시 필요: input_tile + output_tile + weight + metadata(~512B)

### 3.3 Shared Memory (SM)
- 크기: 4 MiB = 4,194,304 bytes
- 16 bank, NoC 경유 접근
- 용도: Weight 상주, 입력 이미지, 코어 간 중간 데이터(feature map)

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

## 5. Queue 분석 (8 cases)

### 5.1 타일링별 tile-ops 계산

각 해상도에서 출력 feature map을 T x T 단위로 분할:
```
tiles = ceil(H_out / T) * ceil(W_out / T)
```

**고정 타일링 타일 수:**

| 출력 해상도 | 16x16 tiles | 32x32 tiles |
|:---|---:|---:|
| 320x320 | 20x20 = 400 | 10x10 = 100 |
| 160x160 | 10x10 = 100 | 5x5 = 25 |
| 80x80 | 5x5 = 25 | 3x3 = 9 |
| 40x40 | 3x3 = 9 | 2x2 = 4 |
| 20x20 | 2x2 = 4 | 1x1 = 1 |

**Adaptive 타일링**: UB budget 내에 input_tile + output_tile + weight + metadata(512B)가 들어가는 최대 T를 레이어별로 계산.

```
Standard Conv (K*K, stride S):
  input_tile = (S*T + K - S)^2 * C_in
  output_tile = T^2 * C_out
  weight = K_h * K_w * C_in * C_out (full weight)

1x1 Conv:
  input_tile = T^2 * C_in
  output_tile = T^2 * C_out
  weight = C_in * C_out

DW Conv:
  input_tile = (S*T + K - S)^2 * C_in
  output_tile = T^2 * C_out
  weight = K_h * K_w * 1 * C_out
```

**Conv tile-ops (타일-레이어 연산 총 수):**

| Model | 16x16 | 32x32 | Adaptive Half | **Adaptive Full** |
|:---|---:|---:|---:|---:|
| v10n | 1,758 | 540 | 6,817 | **2,156** |
| v10s | 1,774 | 544 | 50,774 | **28,157** |

> **핵심 변화**: Adaptive Full에서 v10n conv tile-ops가 6,817 -> 2,156으로 68% 감소. 이는 weight > 64KiB인 레이어들에서 T=1 (tile 수 폭발)이 해소되어 적절한 T 값을 확보했기 때문.
>
> v10s는 여전히 28,157로 높음. weight > 128KiB인 레이어가 29개(T=1 유지)로, 이들이 tile 수를 지배.

**비Conv ops tile-ops 추정:**
- 16x16: ~755 (해상도별 가중 평균)
- 32x32 / Adaptive Half / Adaptive Full: ~220

**Instruction 오버헤드:**
- Per layer: LAYER_START(1) + LAYER_END(1) = 2 records
- Per model: EPOCH_COMMIT(1) + STOP(1) = 2 records
- v10n: (83 + 30) x 2 + 2 = 228 records
- v10s: (87 + 30) x 2 + 2 = 236 records

### 5.2 Core당 instruction record 계산

```
total_records = total_tile_ops + overhead_records
per_core_raw = total_records / 12
per_core_margined = per_core_raw * 1.5
```

| Model | Tiling | Conv Tiles | NonConv Tiles | Overhead | Total Records | Per Core (raw) | Per Core (x1.5) | Limit | 잔여 | 판정 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| v10n | 16x16 | 1,758 | 755 | 228 | 2,741 | 228.4 | **342.6** | 512 | 169.4 | **PASS** |
| v10n | 32x32 | 540 | 220 | 228 | 988 | 82.3 | **123.5** | 512 | 388.5 | **PASS** |
| v10n | Adaptive Half | 6,817 | 220 | 228 | 7,265 | 605.4 | **908.1** | 512 | -396.1 | **FAIL** |
| v10n | **Adaptive Full** | **2,156** | 220 | 228 | **2,604** | 217.0 | **325.5** | 512 | **186.5** | **PASS** |
| v10s | 16x16 | 1,774 | 755 | 236 | 2,765 | 230.4 | **345.6** | 512 | 166.4 | **PASS** |
| v10s | 32x32 | 544 | 220 | 236 | 1,000 | 83.3 | **125.0** | 512 | 387.0 | **PASS** |
| v10s | Adaptive Half | 50,774 | 220 | 236 | 51,230 | 4,269.2 | **6,403.8** | 512 | -5,891.8 | **FAIL** |
| v10s | **Adaptive Full** | **28,157** | 220 | 236 | **28,613** | 2,384.4 | **3,576.6** | 512 | **-3,064.6** | **FAIL** |

### 5.3 판정 결과표

| Model | Tiling | Per Core (x1.5) | Limit | 판정 |
|:---|:---|---:|---:|:---|
| v10n | 16x16 | 342.6 | 512 | **PASS** |
| v10n | 32x32 | 123.5 | 512 | **PASS** |
| v10n | Adaptive Half | 908.1 | 512 | **FAIL** |
| v10n | **Adaptive Full** | **325.5** | 512 | **PASS** |
| v10s | 16x16 | 345.6 | 512 | **PASS** |
| v10s | 32x32 | 125.0 | 512 | **PASS** |
| v10s | Adaptive Half | 6,403.8 | 512 | **FAIL** |
| v10s | **Adaptive Full** | **3,576.6** | 512 | **FAIL** |

**Baseline 대비 핵심 변화:**
- **v10n Adaptive Full: FAIL -> PASS**. 908.1 -> 325.5 records로 Q 사용량 64% 감소. 16x16 고정 타일(342.6)보다도 적은 record 소비.
- v10s Adaptive Full: 여전히 FAIL이나, 6,403.8 -> 3,576.6으로 44% 감소. Weight > 128KiB 레이어가 다수(29개)라 T=1 문제 미해소.

## 6. Unified Buffer 분석 (8 cases)

### 6.1 고정 타일 분석 (16x16, 32x32)

**16x16 타일링:**

| Model | Fit Half (64KiB) | Fail Half | Fit Full (128KiB) | Fail Full | 판정 |
|:---|---:|---:|---:|---:|:---|
| v10n | 31/83 | 52 | 57/83 | 26 | **FAIL** |
| v10s | 11/87 | 76 | 31/87 | 56 | **FAIL** |

UB half(64KiB) 초과 주요 원인:
- 40x40 해상도 Conv 3x3 (C=64): input=18x18x64=20,736 + output=16x16x64=16,384 + weight=36,864 = 74,496B > 65,536
- DW stride-2 (C=128~256): input halo 확대로 in_tile 급증
- 1x1 (C_in=256~512): input+output 합이 큰 경우

**32x32 타일링:**

| Model | Fit Half (64KiB) | Fail Half | Fit Full (128KiB) | Fail Full | 판정 |
|:---|---:|---:|---:|---:|:---|
| v10n | 4/83 | 79 | 21/83 | 62 | **FAIL** |
| v10s | 2/87 | 85 | 8/87 | 79 | **FAIL** |

32x32에서는 input/output tile 자체가 거대해짐. 32x32 output tile, C=64: 32*32*64 = 65,536B = 정확히 UB half. input tile은 halo로 인해 더 크므로 거의 모든 레이어 초과.

### 6.2 Adaptive Half (64 KiB) 분석

**v10n (Adaptive Half): 83 layers 중 11개 fail**

Adaptive Half는 UB half(64 KiB) 내에 맞도록 최대 T를 자동 계산. 그러나 weight 자체가 64KiB를 초과하는 레이어에서는 T=1에서도 fit 불가.

UB half 초과 레이어 (v10n, Adaptive Half):

| Layer | Name | Type | Weight (B) | T | UB_total | 초과 원인 |
|---:|:---|:---|---:|---:|---:|:---|
| 23 | model.8.cv1.conv | 1x1 | 65,536 | 1 | 66,560 | weight=64KB, T=1에서도 weight+meta > half |
| 24 | model.8.m.0.cv1.conv | Conv | 147,456 | 1 | 149,248 | weight=144KB >> half |
| 25 | model.8.m.0.cv2.conv | Conv | 147,456 | 1 | 149,248 | weight=144KB >> half |
| 26 | model.8.cv2.conv | 1x1 | 98,304 | 1 | 99,456 | weight=96KB > half |
| 28 | model.9.cv2.conv | 1x1 | 131,072 | 1 | 132,352 | weight=128KB >> half |
| 29 | model.10.cv1.conv | 1x1 | 65,536 | 1 | 66,560 | weight=64KB |
| 35 | model.10.cv2.conv | 1x1 | 65,536 | 1 | 66,560 | weight=64KB |
| 51 | model.22.cv1.conv | 1x1 | 98,304 | 1 | 99,456 | weight=96KB |
| 58 | model.22.cv2.conv | 1x1 | 98,304 | 1 | 99,456 | weight=96KB |
| 62 | one2one_cv2.1.0 | Conv | 73,728 | 1 | 75,456 | weight=72KB |
| 65 | one2one_cv2.2.0 | Conv | 147,456 | 1 | 150,336 | weight=144KB |

**v10s (Adaptive Half): 87 layers 중 38개 fail**

v10s에서 weight > 64KiB인 레이어가 38개로, 대부분 backbone 중반 이후(model.6~) 레이어.

### 6.3 Adaptive Full (128 KiB) 분석

Adaptive Full은 ping-pong을 포기하고 UB 전체 128 KiB를 활용. 이로 인해 Adaptive Half에서 fail하던 weight 64~128KiB 구간의 레이어들이 해소됨.

**v10n (Adaptive Full): 83 layers 중 4개 fail**

| Layer | Name | Type | Weight (B) | T | UB_total | 초과 원인 |
|---:|:---|:---|---:|---:|---:|:---|
| 24 | model.8.m.0.cv1.conv | Conv 3x3 | 147,456 | 1 | 149,248 | weight=144KB > 128KiB |
| 25 | model.8.m.0.cv2.conv | Conv 3x3 | 147,456 | 1 | 149,248 | weight=144KB > 128KiB |
| 28 | model.9.cv2.conv | 1x1 | 131,072 | 1 | 132,352 | weight=128KB + meta > 128KiB |
| 65 | one2one_cv2.2.0 | Conv 3x3 | 147,456 | 1 | 150,336 | weight=144KB > 128KiB |

> Baseline Adaptive Half에서 11개 fail -> Adaptive Full에서 **4개 fail**로 감소 (7개 해소).
>
> 해소된 7개 레이어: #23(64KB), #26(96KB), #29(64KB), #35(64KB), #51(96KB), #58(96KB), #62(72KB).
> 이들은 weight가 64~96KB 범위로, Full budget(128KiB)에서는 T >= 7~15를 확보하여 fit.

**Adaptive Half -> Full 전환 효과 (v10n 해소된 레이어 상세):**

| Layer | Name | Weight (B) | Half: T/Tiles/UB | Full: T/Tiles/UB | 변화 |
|---:|:---|---:|:---|:---|:---|
| 23 | model.8.cv1.conv | 65,536 | T=1, 400tiles, 66,560 (FAIL) | T=11, 4tiles, 128,000 (PASS) | tiles 99% 감소 |
| 26 | model.8.cv2.conv | 98,304 | T=1, 400tiles, 99,456 (FAIL) | T=7, 9tiles, 130,176 (PASS) | tiles 98% 감소 |
| 29 | model.10.cv1.conv | 65,536 | T=1, 400tiles, 66,560 (FAIL) | T=11, 4tiles, 128,000 (PASS) | tiles 99% 감소 |
| 35 | model.10.cv2.conv | 65,536 | T=1, 400tiles, 66,560 (FAIL) | T=11, 4tiles, 128,000 (PASS) | tiles 99% 감소 |
| 51 | model.22.cv1.conv | 98,304 | T=1, 400tiles, 99,456 (FAIL) | T=7, 9tiles, 130,176 (PASS) | tiles 98% 감소 |
| 58 | model.22.cv2.conv | 98,304 | T=1, 400tiles, 99,456 (FAIL) | T=7, 9tiles, 130,176 (PASS) | tiles 98% 감소 |
| 62 | one2one_cv2.1.0 | 73,728 | T=1, 1,600tiles, 75,456 (FAIL) | T=15, 9tiles, 125,632 (PASS) | tiles 99% 감소 |

**v10s (Adaptive Full): 87 layers 중 29개 fail**

v10s에서 weight > 128KiB인 레이어가 29개. Adaptive Half의 38개에서 9개 감소 (weight 64~128KB 범위의 9개 해소).

v10s Adaptive Full fail 레이어 (weight > 128 KiB):

| Layer | Name | Type | Weight (B) | UB_total (T=1) |
|---:|:---|:---|---:|---:|
| 16-19 | model.6.m.0/1.cv1/cv2 | Conv 3x3 | 147,456 ea | 149,248 |
| 20-21 | model.6.cv2/7.cv1 | 1x1 | 131,072 ea | 132,352 |
| 23 | model.8.cv1.conv | 1x1 | 262,144 | 263,680 |
| 25 | model.8.m.0.cv1.1 | 1x1 | 131,072 | 132,352 |
| 28 | model.8.m.0.cv1.3 | 1x1 | 131,072 | 132,352 |
| 30 | model.8.cv2.conv | 1x1 | 393,216 | 395,008 |
| 31 | model.9.cv1.conv | 1x1 | 131,072 | 132,352 |
| 32 | model.9.cv2.conv | 1x1 | 524,288 | 526,336 |
| 33 | model.10.cv1.conv | 1x1 | 262,144 | 263,680 |
| 34 | model.10.attn.qkv | 1x1 | 131,072 | 132,352 |
| 37-38 | model.10.ffn.0/1 | 1x1 | 131,072 ea | 132,352 |
| 39 | model.10.cv2.conv | 1x1 | 262,144 | 263,680 |
| 40 | model.13.cv1.conv | 1x1 | 196,608 | 198,144 |
| 41-42 | model.13.m.0.cv1/cv2 | Conv 3x3 | 147,456 ea | 149,248 |
| 48 | model.17.conv | Conv 3x3 | 147,456 | 149,248 |
| 50-51 | model.19.m.0.cv1/cv2 | Conv 3x3 | 147,456 ea | 149,248 |
| 55 | model.22.cv1.conv | 1x1 | 393,216 | 395,008 |
| 57 | model.22.m.0.cv1.1 | 1x1 | 131,072 | 132,352 |
| 60 | model.22.m.0.cv1.3 | 1x1 | 131,072 | 132,352 |
| 62 | model.22.cv2.conv | 1x1 | 393,216 | 395,008 |
| 66 | one2one_cv2.1.0 | Conv 3x3 | 147,456 | 150,336 |
| 69 | one2one_cv2.2.0 | Conv 3x3 | 294,912 | 300,096 |

v10s에서 Adaptive Half -> Full로 해소된 9개 레이어:

| Layer | Name | Weight | Half -> Full |
|---:|:---|---:|:---|
| 6 | model.3.conv | 73,728 | T=1,6400tiles -> T=11,64tiles |
| 15 | model.6.cv1.conv | 65,536 | T=1,1600tiles -> T=11,16tiles |
| 36 | model.10.attn.proj | 65,536 | T=1,400tiles -> T=11,4tiles |
| 43 | model.13.cv2.conv | 98,304 | T=1,1600tiles -> T=7,36tiles |
| 49 | model.19.cv1.conv | 98,304 | T=1,1600tiles -> T=7,36tiles |
| 52 | model.19.cv2.conv | 98,304 | T=1,1600tiles -> T=7,36tiles |
| 53 | model.20.cv1.conv | 65,536 | T=1,1600tiles -> T=11,16tiles |
| 63 | one2one_cv2.0.0 | 73,728 | T=1,6400tiles -> T=15,36tiles |
| 83 | one2one_cv3.2.0.1 | 65,536 | T=1,400tiles -> T=10,4tiles |

### 6.4 판정 결과표

| Model | Tiling | Fit Half (64KiB) | Fail Half | Fit Full (128KiB) | Fail Full | 판정 |
|:---|:---|---:|---:|---:|---:|:---|
| v10n | 16x16 | 31/83 | 52 | 57/83 | 26 | **FAIL** |
| v10n | 32x32 | 4/83 | 79 | 21/83 | 62 | **FAIL** |
| v10n | Adaptive Half | 72/83 | 11 | 79/83 | 4 | **COND** |
| v10n | **Adaptive Full** | N/A | N/A | **79/83** | **4** | **COND** |
| v10s | 16x16 | 11/87 | 76 | 31/87 | 56 | **FAIL** |
| v10s | 32x32 | 2/87 | 85 | 8/87 | 79 | **FAIL** |
| v10s | Adaptive Half | 49/87 | 38 | 58/87 | 29 | **FAIL** |
| v10s | **Adaptive Full** | N/A | N/A | **58/87** | **29** | **FAIL** |

> **COND** = 조건부 PASS. v10n Adaptive Full에서 4개 레이어만 fail. 이들은 weight > 128KiB인 레이어로, weight tiling 또는 weight streaming 기법으로 해결 가능.
>
> Adaptive Full의 UB 판정 기준은 Fit Full(128KiB). Ping-pong을 사용하지 않으므로 Half 기준은 적용하지 않음.

## 7. Shared Memory 분석

### 7.1 정적 할당 (Weight + Input)

SM에 추론 시작 전 적재해야 할 정적 데이터:

| 항목 | v10n | v10s |
|:---|---:|---:|
| Weight (s8) | 2,292,576 B (2.19 MB) | 7,240,032 B (6.90 MB) |
| Input image (640x640x3 i8) | 1,228,800 B (1.17 MB) | 1,228,800 B (1.17 MB) |
| **정적 합계** | **3,521,376 B (3.36 MB)** | **8,468,832 B (8.08 MB)** |
| SM 용량 (4 MiB) | 4,194,304 B | 4,194,304 B |
| 잔여 공간 | **672,928 B (657 KB)** | **-4,274,528 B (-4.08 MB)** |
| **정적 적합 여부** | **YES** | **NO** |

> **v10s는 weight 단독(6.90MB)이 SM 전체 용량(4MB)을 초과**하여 즉시 FAIL.

### 7.2 레이어 전이 Peak

레이어 실행 중 SM에는 weight(상주) + 현재 입력 FM + 현재 출력 FM이 동시에 존재:

```
peak_SM = weight_total + input_FM_bytes + output_FM_bytes
```

**v10n 레이어 전이 분석 (FAIL하는 3개 레이어):**

| Layer | Name | Input FM | Output FM | Peak | SM (4 MiB) | Delta | 판정 |
|---:|:---|---:|---:|---:|---:|---:|:---|
| 0 | model.0.conv | 1,228,800 | 1,638,400 | 5,159,776 (4.92 MB) | 4,194,304 | -965,472 | **FAIL** |
| 1 | model.1.conv | 1,638,400 | 819,200 | 4,749,776 (4.53 MB) | 4,194,304 | -555,472 | **FAIL** |
| 5 | model.2.cv2.conv | 1,228,800 | 819,200 | 4,340,576 (4.14 MB) | 4,194,304 | -146,272 | **FAIL** |

다음 레이어부터 PASS:

| Layer | Name | Peak | 잔여 |
|---:|:---|---:|---:|
| 2 | model.2.cv1.conv | 3,930,976 (3.75 MB) | 263,328 |
| 6 | model.3.conv | 3,930,208 (3.75 MB) | 264,096 |
| 23 | model.8.cv1.conv | 2,792,576 (2.66 MB) | 1,401,728 |
| 82 | one2one_cv3.2.2 | 2,355,376 (2.25 MB) | 1,838,928 |

**v10s: 전 레이어 FAIL (87/87)**

v10s는 weight=6.90MB > SM=4MB이므로 모든 레이어에서 peak > 4MB. 최소 peak = 6.90 + 0.025 + 0.025 = 6.95 MB (20x20 최소 FM 레이어).

### 7.3 판정 결과표

| Model | Weight SM 적합 | 정적(W+Input) 적합 | Peak 초과 레이어 | 최악 Peak | 판정 |
|:---|:---|:---|---:|:---|:---|
| v10n | YES (2.19 < 4 MB) | YES (3.36 MB) | 3/83 (Layer 0,1,5) | 4.92 MB | **COND** |
| v10s | **NO** (6.90 > 4 MB) | **NO** (8.08 MB) | 87/87 | N/A | **FAIL** |

> SM 분석 결과는 Baseline과 동일. 타일링 전략과 무관한 구조적 제약.

## 8. 종합 교차 분석 (8 cases)

### 8.1 8-Case 판정 매트릭스

| Model | Tiling | Q (512 rec) | UB | SM (4 MiB) | **종합** |
|:---|:---|:---|:---|:---|:---|
| v10n | 16x16 | **PASS** (343/512) | **FAIL** (52 half fail, 26 full fail) | **COND** (3 fail) | **FAIL** |
| v10n | 32x32 | **PASS** (124/512) | **FAIL** (79 half fail, 62 full fail) | **COND** (3 fail) | **FAIL** |
| v10n | Adaptive Half | **FAIL** (908/512) | **COND** (11 half fail, 4 full fail) | **COND** (3 fail) | **FAIL** |
| v10n | **Adaptive Full** | **PASS** (326/512) | **COND** (4 full fail) | **COND** (3 fail) | **COND** |
| v10s | 16x16 | **PASS** (346/512) | **FAIL** (76 half fail, 56 full fail) | **FAIL** (weight>SM) | **FAIL** |
| v10s | 32x32 | **PASS** (125/512) | **FAIL** (85 half fail, 79 full fail) | **FAIL** (weight>SM) | **FAIL** |
| v10s | Adaptive Half | **FAIL** (6,404/512) | **FAIL** (38 half fail) | **FAIL** (weight>SM) | **FAIL** |
| v10s | **Adaptive Full** | **FAIL** (3,577/512) | **FAIL** (29 full fail) | **FAIL** (weight>SM) | **FAIL** |

### 8.2 종합 판정 요약

| Model | Tiling | 종합 |
|:---|:---|:---|
| v10n | 16x16 | FAIL |
| v10n | 32x32 | FAIL |
| v10n | Adaptive Half | FAIL |
| **v10n** | **Adaptive Full** | **COND** |
| v10s | 16x16 | FAIL |
| v10s | 32x32 | FAIL |
| v10s | Adaptive Half | FAIL |
| v10s | Adaptive Full | FAIL |

> **v10n Adaptive Full이 유일한 COND 케이스.** Baseline에서는 6개 전부 FAIL이었으나, Adaptive Full 도입으로 v10n에 COND 케이스가 발생.

## 9. Baseline 대비 개선 효과

### 9.1 v10n 개선 비교

| 자원 | Baseline 최선 | v2 Adaptive Full | 변화 |
|:---|:---|:---|:---|
| Q | 16x16 PASS (343/512) | Adaptive Full PASS (326/512) | 동등 수준, Full이 약간 더 여유 |
| UB | Adaptive Half COND (11 fail) | Adaptive Full COND (4 fail) | **7개 레이어 추가 해소** |
| SM | COND (3 fail) | COND (3 fail) | 변동 없음 (구조적 제약) |
| **종합** | **6 cases 전부 FAIL** | **Adaptive Full = COND** | **최초 COND 케이스 출현** |

### 9.2 v10n Adaptive Full의 잔여 이슈

**UB 4개 fail 레이어 (weight > 128 KiB):**

| Layer | Name | Weight | UB 초과량 | 해결 방안 |
|---:|:---|---:|---:|:---|
| 24 | model.8.m.0.cv1 | 144 KB | +17.8 KB | weight tiling (Cout 분할) |
| 25 | model.8.m.0.cv2 | 144 KB | +17.8 KB | weight tiling (Cout 분할) |
| 28 | model.9.cv2 | 128 KB | +1.3 KB | metadata 축소 or weight streaming |
| 65 | one2one_cv2.2.0 | 144 KB | +18.8 KB | weight tiling (Cout 분할) |

> Layer 28 (model.9.cv2.conv)은 weight=131,072B + 1x1xCin+Cout + 512 = 132,352B로, **단 1,280B(1.25KB) 초과**. Metadata를 512B -> 0B로 줄이면 fit 가능(weight 정확히 128KiB).

**SM 3개 fail 레이어:**

| Layer | Name | 초과량 | 해결 방안 |
|---:|:---|---:|:---|
| 0 | model.0.conv | 965 KB | 입력 이미지 타일 단위 SM 적재 |
| 1 | model.1.conv | 556 KB | Layer 0 출력을 타일 단위로 처리 |
| 5 | model.2.cv2.conv | 146 KB | C2f concat 입력 FM 관리 최적화 |

### 9.3 v10s 변화

v10s는 SM 제약(weight 6.90MB > SM 4MB)이 근본적으로 해결 불가하므로, Adaptive Full 도입에도 종합 FAIL 유지. 다만 UB 관점에서:
- Adaptive Half: 38개 fail -> Adaptive Full: 29개 fail (9개 감소)
- Q: 6,404 -> 3,577 (44% 감소, 그래도 FAIL)

## 10. 결론 및 권고사항

### 10.1 주요 결론

1. **v10n Adaptive Full이 유일한 COND 케이스.** Baseline 6-case 분석에서는 모든 조합이 FAIL이었으나, ping-pong을 포기하고 UB 전체를 활용하는 Adaptive Full 전략 도입으로 v10n에서 최초의 COND(조건부 통과) 결과 확인.

2. **Adaptive Full의 핵심 효과**: Weight 64~128KB 구간 레이어에서 T=1 -> T=7~15로 전환되어, tile 수가 98~99% 감소. 이것이 Q 사용량 급감(908 -> 326 records)의 직접적 원인.

3. **v10n 잔여 이슈는 7개 레이어**: UB 4개(weight > 128KB) + SM 3개(초기 고해상도). 둘 다 우회 전략 존재.

4. **v10s는 여전히 불가.** Weight(6.90MB) > SM(4MB) 제약은 어떤 타일링으로도 해결 불가. SM 확장 또는 외부 메모리 필수.

### 10.2 v10n 실현 가능 시나리오 (Best-case)

```
전략: Adaptive Full (128KiB single buffer) + Weight Streaming + Layer 0~1 특별처리

Q:  PASS (325.5 / 512 records, 여유 36.4%)
UB: 4개 fail 레이어에 weight tiling 적용
    - Layer 24,25,65: Cout 방향 2분할 → weight 72KB/slice → fit
    - Layer 28: metadata 축소 또는 weight streaming → fit
SM: Weight 2.19MB + Input 1.17MB = 3.36MB
    Layer 0~1: 입력을 타일 단위로 SM에 적재 (전체 1.17MB 대신 타일만)
    Layer 5: C2f concat 입력 FM 관리 최적화
```

**Trade-off**: Ping-pong 미사용으로 DMA-compute overlap이 불가. 각 타일에서 DMA 완료 후 compute 시작해야 하므로 latency 증가. 그러나 tile 수 자체가 적으므로 총 inference 시간은 case-by-case 비교 필요.

### 10.3 권고사항

**단기 (현재 아키텍처 내):**

1. **v10n: Adaptive Full 전략 채택 + 4개 레이어 weight tiling**
   - 가장 현실적인 단일 전략
   - Weight tiling은 Cout 방향 분할로 구현 가능 (partial sum accumulation 필요)
   - SM Layer 0~1 특별처리는 입력 타일 단위 적재로 해결

2. **Hybrid 전략 검토**: Adaptive Full 기본 + 특정 레이어만 ping-pong 사용
   - Weight <= 64KB 레이어(72개): ping-pong으로 DMA overlap 활용
   - Weight 64~128KB 레이어(7개): single buffer(full)
   - Weight > 128KB 레이어(4개): weight tiling + single buffer

**중장기 (아키텍처 변경 시):**

3. **SM 확장**: 8 MiB 이상으로 확장 시 v10s도 수용 가능
4. **External Memory 활용**: Weight를 외부 메모리에 두고 layer 단위로 SM에 적재
5. **UB 확장**: 256 KiB로 확장 시 v10n의 4개 fail 레이어도 완전 해소

---

## 부록 A: Adaptive Half 타일 목록 (v10n)

| Layer | Name | Out HxW | T | Tiles | UB Total | Fit Half |
|---:|:---|:---|---:|---:|---:|:---|
| 0 | model.0.conv | 320x320 | 47 | 49 | 63,363 | YES |
| 1 | model.1.conv | 160x160 | 24 | 49 | 61,968 | YES |
| 2 | model.2.cv1.conv | 160x160 | 31 | 36 | 63,040 | YES |
| 3 | model.2.m.0.cv1.conv | 160x160 | 43 | 16 | 64,800 | YES |
| 4 | model.2.m.0.cv2.conv | 160x160 | 43 | 16 | 64,800 | YES |
| 5 | model.2.cv2.conv | 160x160 | 28 | 36 | 64,768 | YES |
| 6 | model.3.conv | 80x80 | 15 | 36 | 64,096 | YES |
| 7 | model.4.cv1.conv | 80x80 | 21 | 16 | 61,056 | YES |
| 8 | model.4.m.0.cv1.conv | 80x80 | 28 | 9 | 63,616 | YES |
| 9 | model.4.m.0.cv2.conv | 80x80 | 28 | 9 | 63,616 | YES |
| 10 | model.4.m.1.cv1.conv | 80x80 | 28 | 9 | 63,616 | YES |
| 11 | model.4.m.1.cv2.conv | 80x80 | 28 | 9 | 63,616 | YES |
| 12 | model.4.cv2.conv | 80x80 | 17 | 25 | 64,192 | YES |
| 13 | model.5.cv1.conv | 80x80 | 17 | 25 | 64,192 | YES |
| 14 | model.5.cv2.conv | 40x40 | 9 | 25 | 58,240 | YES |
| 15 | model.6.cv1.conv | 40x40 | 13 | 16 | 60,160 | YES |
| 16 | model.6.m.0.cv1.conv | 40x40 | 13 | 16 | 62,592 | YES |
| 17 | model.6.m.0.cv2.conv | 40x40 | 13 | 16 | 62,592 | YES |
| 18 | model.6.m.1.cv1.conv | 40x40 | 13 | 16 | 62,592 | YES |
| 19 | model.6.m.1.cv2.conv | 40x40 | 13 | 16 | 62,592 | YES |
| 20 | model.6.cv2.conv | 40x40 | 9 | 25 | 64,384 | YES |
| 21 | model.7.cv1.conv | 40x40 | 9 | 25 | 64,384 | YES |
| 22 | model.7.cv2.conv | 20x20 | 6 | 16 | 55,296 | YES |
| 23 | model.8.cv1.conv | 20x20 | 1 | 400 | 66,560 | **NO** |
| 24 | model.8.m.0.cv1.conv | 20x20 | 1 | 400 | 149,248 | **NO** |
| 25 | model.8.m.0.cv2.conv | 20x20 | 1 | 400 | 149,248 | **NO** |
| 26 | model.8.cv2.conv | 20x20 | 1 | 400 | 99,456 | **NO** |
| 27 | model.9.cv1.conv | 20x20 | 9 | 9 | 64,384 | YES |
| 28 | model.9.cv2.conv | 20x20 | 1 | 400 | 132,352 | **NO** |
| 29 | model.10.cv1.conv | 20x20 | 1 | 400 | 66,560 | **NO** |
| 30 | model.10.attn.qkv.conv | 20x20 | 9 | 9 | 64,384 | YES |
| 31 | model.10.attn.pe.conv | 20x20 | 14 | 4 | 59,520 | YES |
| 32 | model.10.attn.proj.conv | 20x20 | 13 | 4 | 60,160 | YES |
| 33 | model.10.ffn.0.conv | 20x20 | 9 | 9 | 64,384 | YES |
| 34 | model.10.ffn.1.conv | 20x20 | 9 | 9 | 64,384 | YES |
| 35 | model.10.cv2.conv | 20x20 | 1 | 400 | 66,560 | **NO** |
| 36 | model.13.cv1.conv | 40x40 | 5 | 64 | 62,464 | YES |
| 37 | model.13.m.0.cv1.conv | 40x40 | 13 | 16 | 62,592 | YES |
| 38 | model.13.m.0.cv2.conv | 40x40 | 13 | 16 | 62,592 | YES |
| 39 | model.13.cv2.conv | 40x40 | 11 | 16 | 63,808 | YES |
| 40 | model.16.cv1.conv | 80x80 | 14 | 36 | 62,976 | YES |
| 41 | model.16.m.0.cv1.conv | 80x80 | 28 | 9 | 63,616 | YES |
| 42 | model.16.m.0.cv2.conv | 80x80 | 28 | 9 | 63,616 | YES |
| 43 | model.16.cv2.conv | 80x80 | 19 | 25 | 64,416 | YES |
| 44 | model.17.conv | 40x40 | 8 | 25 | 59,968 | YES |
| 45 | model.19.cv1.conv | 40x40 | 11 | 16 | 63,808 | YES |
| 46 | model.19.m.0.cv1.conv | 40x40 | 13 | 16 | 62,592 | YES |
| 47 | model.19.m.0.cv2.conv | 40x40 | 13 | 16 | 62,592 | YES |
| 48 | model.19.cv2.conv | 40x40 | 11 | 16 | 63,808 | YES |
| 49 | model.20.cv1.conv | 40x40 | 13 | 16 | 60,160 | YES |
| 50 | model.20.cv2.conv | 20x20 | 9 | 9 | 58,240 | YES |
| 51 | model.22.cv1.conv | 20x20 | 1 | 400 | 99,456 | **NO** |
| 52 | model.22.m.0.cv1.0.conv | 20x20 | 14 | 4 | 59,520 | YES |
| 53 | model.22.m.0.cv1.1.conv | 20x20 | 9 | 9 | 64,384 | YES |
| 54 | model.22.m.0.cv1.2.conv.conv | 20x20 | 6 | 16 | 59,136 | YES |
| 55 | model.22.m.0.cv1.2.conv1.conv | 20x20 | 10 | 4 | 65,280 | YES |
| 56 | model.22.m.0.cv1.3.conv | 20x20 | 9 | 9 | 64,384 | YES |
| 57 | model.22.m.0.cv1.4.conv | 20x20 | 14 | 4 | 59,520 | YES |
| 58 | model.22.cv2.conv | 20x20 | 1 | 400 | 99,456 | **NO** |
| 59 | one2one_cv2.0.0 | 80x80 | 13 | 49 | 62,592 | YES |
| 60 | one2one_cv2.0.1 | 80x80 | 13 | 49 | 62,592 | YES |
| 61 | one2one_cv2.0.2 | 80x80 | 21 | 16 | 61,056 | YES |
| 62 | one2one_cv2.1.0 | 40x40 | 1 | 1,600 | 75,456 | **NO** |
| 63 | one2one_cv2.1.1 | 40x40 | 13 | 16 | 62,592 | YES |
| 64 | one2one_cv2.1.2 | 40x40 | 21 | 4 | 61,056 | YES |
| 65 | one2one_cv2.2.0 | 20x20 | 1 | 400 | 150,336 | **NO** |
| 66 | one2one_cv2.2.1 | 20x20 | 13 | 4 | 62,592 | YES |
| 67 | one2one_cv2.2.2 | 20x20 | 20 | 1 | 55,808 | YES |
| 68 | one2one_cv3.0.0.0 | 80x80 | 21 | 16 | 63,168 | YES |
| 69 | one2one_cv3.0.0.1 | 80x80 | 20 | 16 | 63,232 | YES |
| 70 | one2one_cv3.0.1.0 | 80x80 | 19 | 25 | 65,392 | YES |
| 71 | one2one_cv3.0.1.1 | 80x80 | 19 | 25 | 64,672 | YES |
| 72 | one2one_cv3.0.2 | 80x80 | 19 | 25 | 64,672 | YES |
| 73 | one2one_cv3.1.0.0 | 40x40 | 14 | 9 | 59,520 | YES |
| 74 | one2one_cv3.1.0.1 | 40x40 | 16 | 9 | 64,000 | YES |
| 75 | one2one_cv3.1.1.0 | 40x40 | 19 | 9 | 65,392 | YES |
| 76 | one2one_cv3.1.1.1 | 40x40 | 19 | 9 | 64,672 | YES |
| 77 | one2one_cv3.1.2 | 40x40 | 19 | 9 | 64,672 | YES |
| 78 | one2one_cv3.2.0.0 | 20x20 | 10 | 4 | 65,280 | YES |
| 79 | one2one_cv3.2.0.1 | 20x20 | 11 | 4 | 61,648 | YES |
| 80 | one2one_cv3.2.1.0 | 20x20 | 19 | 4 | 65,392 | YES |
| 81 | one2one_cv3.2.1.1 | 20x20 | 19 | 4 | 64,672 | YES |
| 82 | one2one_cv3.2.2 | 20x20 | 19 | 4 | 64,672 | YES |

## 부록 B: Adaptive Full 타일 목록 (v10n)

| Layer | Name | Out HxW | T_half | T_full | Tiles_half | Tiles_full | UB_half | UB_full | Fit Full |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|:---|
| 0 | model.0.conv | 320x320 | 47 | 67 | 49 | 25 | 63,363 | 127,443 | YES |
| 1 | model.1.conv | 160x160 | 24 | 35 | 49 | 25 | 61,968 | 124,976 | YES |
| 2 | model.2.cv1.conv | 160x160 | 31 | 44 | 36 | 16 | 63,040 | 125,440 | YES |
| 3 | model.2.m.0.cv1.conv | 160x160 | 43 | 62 | 16 | 9 | 64,800 | 129,856 | YES |
| 4 | model.2.m.0.cv2.conv | 160x160 | 43 | 62 | 16 | 9 | 64,800 | 129,856 | YES |
| 5 | model.2.cv2.conv | 160x160 | 28 | 40 | 36 | 16 | 64,768 | 130,048 | YES |
| 6 | model.3.conv | 80x80 | 15 | 23 | 36 | 16 | 64,096 | 123,488 | YES |
| 7 | model.4.cv1.conv | 80x80 | 21 | 31 | 16 | 9 | 61,056 | 127,616 | YES |
| 8 | model.4.m.0.cv1.conv | 80x80 | 28 | 42 | 9 | 4 | 63,616 | 128,128 | YES |
| 9 | model.4.m.0.cv2.conv | 80x80 | 28 | 42 | 9 | 4 | 63,616 | 128,128 | YES |
| 10 | model.4.m.1.cv1.conv | 80x80 | 28 | 42 | 9 | 4 | 63,616 | 128,128 | YES |
| 11 | model.4.m.1.cv2.conv | 80x80 | 28 | 42 | 9 | 4 | 63,616 | 128,128 | YES |
| 12 | model.4.cv2.conv | 80x80 | 17 | 25 | 25 | 16 | 64,192 | 128,704 | YES |
| 13 | model.5.cv1.conv | 80x80 | 17 | 25 | 25 | 16 | 64,192 | 128,704 | YES |
| 14 | model.5.cv2.conv | 40x40 | 9 | 13 | 25 | 16 | 58,240 | 116,608 | YES |
| 15 | model.6.cv1.conv | 40x40 | 13 | 21 | 16 | 4 | 60,160 | 129,792 | YES |
| 16 | model.6.m.0.cv1.conv | 40x40 | 13 | 26 | 16 | 4 | 62,592 | 130,816 | YES |
| 17 | model.6.m.0.cv2.conv | 40x40 | 13 | 26 | 16 | 4 | 62,592 | 130,816 | YES |
| 18 | model.6.m.1.cv1.conv | 40x40 | 13 | 26 | 16 | 4 | 62,592 | 130,816 | YES |
| 19 | model.6.m.1.cv2.conv | 40x40 | 13 | 26 | 16 | 4 | 62,592 | 130,816 | YES |
| 20 | model.6.cv2.conv | 40x40 | 9 | 15 | 25 | 9 | 64,384 | 119,680 | YES |
| 21 | model.7.cv1.conv | 40x40 | 9 | 15 | 25 | 9 | 64,384 | 119,680 | YES |
| 22 | model.7.cv2.conv | 20x20 | 6 | 9 | 16 | 9 | 55,296 | 115,968 | YES |
| 23 | model.8.cv1.conv | 20x20 | 1 | **11** | 400 | **4** | 66,560 | 128,000 | **YES** |
| 24 | model.8.m.0.cv1.conv | 20x20 | 1 | 1 | 400 | 400 | 149,248 | 149,248 | **NO** |
| 25 | model.8.m.0.cv2.conv | 20x20 | 1 | 1 | 400 | 400 | 149,248 | 149,248 | **NO** |
| 26 | model.8.cv2.conv | 20x20 | 1 | **7** | 400 | **9** | 99,456 | 130,176 | **YES** |
| 27 | model.9.cv1.conv | 20x20 | 9 | 15 | 9 | 4 | 64,384 | 119,680 | YES |
| 28 | model.9.cv2.conv | 20x20 | 1 | 1 | 400 | 400 | 132,352 | 132,352 | **NO** |
| 29 | model.10.cv1.conv | 20x20 | 1 | **11** | 400 | **4** | 66,560 | 128,000 | **YES** |
| 30 | model.10.attn.qkv.conv | 20x20 | 9 | 15 | 9 | 4 | 64,384 | 119,680 | YES |
| 31 | model.10.attn.pe.conv | 20x20 | 14 | 20 | 4 | 1 | 59,520 | 114,816 | YES |
| 32 | model.10.attn.proj.conv | 20x20 | 13 | 20 | 4 | 1 | 60,160 | 119,296 | YES |
| 33 | model.10.ffn.0.conv | 20x20 | 9 | 15 | 9 | 4 | 64,384 | 119,680 | YES |
| 34 | model.10.ffn.1.conv | 20x20 | 9 | 15 | 9 | 4 | 64,384 | 119,680 | YES |
| 35 | model.10.cv2.conv | 20x20 | 1 | **11** | 400 | **4** | 66,560 | 128,000 | **YES** |
| 36 | model.13.cv1.conv | 40x40 | 5 | 12 | 64 | 16 | 62,464 | 123,392 | YES |
| 37 | model.13.m.0.cv1.conv | 40x40 | 13 | 26 | 16 | 4 | 62,592 | 130,816 | YES |
| 38 | model.13.m.0.cv2.conv | 40x40 | 13 | 26 | 16 | 4 | 62,592 | 130,816 | YES |
| 39 | model.13.cv2.conv | 40x40 | 11 | 18 | 16 | 9 | 63,808 | 128,768 | YES |
| 40 | model.16.cv1.conv | 80x80 | 14 | 21 | 36 | 16 | 62,976 | 125,696 | YES |
| 41 | model.16.m.0.cv1.conv | 80x80 | 28 | 42 | 9 | 4 | 63,616 | 128,128 | YES |
| 42 | model.16.m.0.cv2.conv | 80x80 | 28 | 42 | 9 | 4 | 63,616 | 128,128 | YES |
| 43 | model.16.cv2.conv | 80x80 | 19 | 27 | 25 | 9 | 64,416 | 123,296 | YES |
| 44 | model.17.conv | 40x40 | 8 | 16 | 25 | 9 | 59,968 | 123,456 | YES |
| 45 | model.19.cv1.conv | 40x40 | 11 | 18 | 16 | 9 | 63,808 | 128,768 | YES |
| 46 | model.19.m.0.cv1.conv | 40x40 | 13 | 26 | 16 | 4 | 62,592 | 130,816 | YES |
| 47 | model.19.m.0.cv2.conv | 40x40 | 13 | 26 | 16 | 4 | 62,592 | 130,816 | YES |
| 48 | model.19.cv2.conv | 40x40 | 11 | 18 | 16 | 9 | 63,808 | 128,768 | YES |
| 49 | model.20.cv1.conv | 40x40 | 13 | 21 | 16 | 4 | 60,160 | 129,792 | YES |
| 50 | model.20.cv2.conv | 20x20 | 9 | 13 | 9 | 4 | 58,240 | 116,608 | YES |
| 51 | model.22.cv1.conv | 20x20 | 1 | **7** | 400 | **9** | 99,456 | 130,176 | **YES** |
| 52 | model.22.m.0.cv1.0.conv | 20x20 | 14 | 20 | 4 | 1 | 59,520 | 114,816 | YES |
| 53 | model.22.m.0.cv1.1.conv | 20x20 | 9 | 15 | 9 | 4 | 64,384 | 119,680 | YES |
| 54 | model.22.m.0.cv1.2.conv.conv | 20x20 | 6 | 11 | 16 | 4 | 59,136 | 118,016 | YES |
| 55 | model.22.m.0.cv1.2.conv1.conv | 20x20 | 10 | 14 | 4 | 4 | 65,280 | 118,528 | YES |
| 56 | model.22.m.0.cv1.3.conv | 20x20 | 9 | 15 | 9 | 4 | 64,384 | 119,680 | YES |
| 57 | model.22.m.0.cv1.4.conv | 20x20 | 14 | 20 | 4 | 1 | 59,520 | 114,816 | YES |
| 58 | model.22.cv2.conv | 20x20 | 1 | **7** | 400 | **9** | 99,456 | 130,176 | **YES** |
| 59 | one2one_cv2.0.0 | 80x80 | 13 | 26 | 49 | 16 | 62,592 | 130,816 | YES |
| 60 | one2one_cv2.0.1 | 80x80 | 13 | 26 | 49 | 16 | 62,592 | 130,816 | YES |
| 61 | one2one_cv2.0.2 | 80x80 | 21 | 31 | 16 | 9 | 61,056 | 127,616 | YES |
| 62 | one2one_cv2.1.0 | 40x40 | 1 | **15** | 1,600 | **9** | 75,456 | 125,632 | **YES** |
| 63 | one2one_cv2.1.1 | 40x40 | 13 | 26 | 16 | 4 | 62,592 | 130,816 | YES |
| 64 | one2one_cv2.1.2 | 40x40 | 21 | 31 | 4 | 4 | 61,056 | 127,616 | YES |
| 65 | one2one_cv2.2.0 | 20x20 | 1 | 1 | 400 | 400 | 150,336 | 150,336 | **NO** |
| 66 | one2one_cv2.2.1 | 20x20 | 13 | 20 | 4 | 1 | 62,592 | 93,952 | YES |
| 67 | one2one_cv2.2.2 | 20x20 | 20 | 20 | 1 | 1 | 55,808 | 55,808 | YES |
| 68 | one2one_cv3.0.0.0 | 80x80 | 21 | 30 | 16 | 9 | 63,168 | 124,224 | YES |
| 69 | one2one_cv3.0.0.1 | 80x80 | 20 | 29 | 16 | 9 | 63,232 | 126,736 | YES |
| 70 | one2one_cv3.0.1.0 | 80x80 | 19 | 27 | 25 | 9 | 65,392 | 126,832 | YES |
| 71 | one2one_cv3.0.1.1 | 80x80 | 19 | 27 | 25 | 9 | 64,672 | 123,552 | YES |
| 72 | one2one_cv3.0.2 | 80x80 | 19 | 27 | 25 | 9 | 64,672 | 123,552 | YES |
| 73 | one2one_cv3.1.0.0 | 40x40 | 14 | 21 | 9 | 4 | 59,520 | 125,824 | YES |
| 74 | one2one_cv3.1.0.1 | 40x40 | 16 | 24 | 9 | 4 | 64,000 | 130,560 | YES |
| 75 | one2one_cv3.1.1.0 | 40x40 | 19 | 27 | 9 | 4 | 65,392 | 126,832 | YES |
| 76 | one2one_cv3.1.1.1 | 40x40 | 19 | 27 | 9 | 4 | 64,672 | 123,552 | YES |
| 77 | one2one_cv3.1.2 | 40x40 | 19 | 27 | 9 | 4 | 64,672 | 123,552 | YES |
| 78 | one2one_cv3.2.0.0 | 20x20 | 10 | 14 | 4 | 4 | 65,280 | 118,528 | YES |
| 79 | one2one_cv3.2.0.1 | 20x20 | 11 | 18 | 4 | 4 | 61,648 | 129,856 | YES |
| 80 | one2one_cv3.2.1.0 | 20x20 | 19 | 20 | 4 | 1 | 65,392 | 71,952 | YES |
| 81 | one2one_cv3.2.1.1 | 20x20 | 19 | 20 | 4 | 1 | 64,672 | 70,912 | YES |
| 82 | one2one_cv3.2.2 | 20x20 | 19 | 20 | 4 | 1 | 64,672 | 70,912 | YES |

---

**참조 파일:**
- Conv Layer Analysis: `docs/agent_results/etc/20260401_yolov10_conv_layer_analysis.md`
- Baseline Report: `docs/agent_results/reports/20260401_Queue_UB_SM_Feasibility_Analysis.md`
- CSV (Queue): `docs/agent_results/reports/20260401_02_queue_analysis.csv`
- CSV (UB): `docs/agent_results/reports/20260401_02_ub_analysis.csv`
- CSV (SM): `docs/agent_results/reports/20260401_02_sm_analysis.csv`
- CSV (Summary): `docs/agent_results/reports/20260401_02_feasibility_summary.csv`
