# Queue / UB / SM 동작 가능 여부 분석 보고서

생성일: 2026-04-01

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
| 타일링 전략 | (1) 16x16 고정, (2) 32x32 고정, (3) Adaptive (UB half에 맞는 최대 타일) |
| Weight 로딩 | 추론 전 1회 SM에 적재, 추론 중 재적재 없음 |
| 입력 이미지 | 640x640x3 i8 = 1,228,800 bytes (1.17 MB), SM에 1회 적재 |
| 데이터 타입 | Activation: i8 (1B), Weight: s8 (1B) |
| 불균형 마진 | 1.5x (코어 분배 불균형 대비) |
| Instruction 오버헤드 | LAYER_START, LAYER_END, EPOCH_COMMIT, STOP, NOP 포함 |
| 비포함 오버헤드 | Semaphore, Profiling |

## 2. 하드웨어 자원 요약

### 2.1 Instruction Queue (Q)
- 크기: 128 KiB per pCore = 131,072 bytes
- 256-byte 고정 크기 instruction record
- 최대 512 records per core
- Record 구조: command header (W0-W3), DMA slots (W4-W15), STREAM/SYNC (W16-W21), COMPUTE (W22-W25), AUX/QUANT (W26-W28), PROF/DEBUG (W29-W31)
- Command types: LAYER_START, LAYER_RUN (메인 연산), LAYER_END, LOOP_JUMP, EPOCH_COMMIT, STOP, NOP

### 2.2 Unified Buffer (UB / pUB)
- 크기: 128 KiB per pCore = 131,072 bytes
- 16-bank SRAM, ping-pong windowing 지원
- Ping-pong double-buffering 시 실효 사용량 = 64 KiB (half)
- 타일 연산 시 필요: input_tile + output_tile + weight + metadata(~512B)

### 2.3 Shared Memory (SM)
- 크기: 4 MiB = 4,194,304 bytes
- 16 bank, NoC 경유 접근
- 용도: Weight 상주, 입력 이미지, 코어 간 중간 데이터(feature map)
- Weight는 추론 전 1회 적재 후 변경 없음

## 3. 모델 요약

### 3.1 YOLOv10n

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

### 3.2 YOLOv10s

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

## 4. Queue 분석

### 4.1 타일 수 계산 방법

각 해상도에서 출력 feature map을 tile_size x tile_size 단위로 분할:

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

**Adaptive 타일링:** UB half (64 KiB) 내에 input_tile + output_tile + weight + metadata(512B)가 들어가는 최대 T를 레이어별로 계산. 타일 크기 공식:

```
Standard Conv (K×K, stride S):
  input_tile = (S*T + K - S) * (S*T + K - S) * C_in
  output_tile = T * T * C_out
  weight = K_h * K_w * C_in_per_group * C_out

1x1 Conv:
  input_tile = T * T * C_in
  output_tile = T * T * C_out

DW Conv:
  input_tile = (S*T + K - S) * (S*T + K - S) * C_in
  output_tile = T * T * C_out
  weight = K_h * K_w * 1 * C_out
```

### 4.2 Tile-ops 계산 결과

**Conv tile-ops (타일-레이어 연산 총 수):**

| Model | 16x16 | 32x32 | Adaptive |
|:---|---:|---:|---:|
| v10n | 1,758 | 540 | 6,817 |
| v10s | 1,774 | 544 | 50,774 |

> Adaptive에서 v10s가 50,774로 급증하는 이유: weight가 큰 레이어(128KB~512KB)에서 T=1까지 축소되어 1600~6400 tiles/layer 발생. 특히 model.6 (Conv 3x3, C=128->128, weight=144KB), model.9.cv2 (1x1, C=1024->512, weight=512KB) 등.

**비Conv ops tile-ops 추정:**
- 16x16: 755 (해상도별 가중 평균)
- 32x32 / Adaptive: 220

**Instruction 오버헤드:**
- Per layer: LAYER_START(1) + LAYER_END(1) = 2 records
- Per model: EPOCH_COMMIT(1) + STOP(1) = 2 records
- v10n: (83 + 30) × 2 + 2 = 228 records
- v10s: (87 + 30) × 2 + 2 = 236 records

### 4.3 Per-core Record 계산

```
total_records = total_tile_ops + overhead_records
per_core_raw = total_records / 12
per_core_margined = per_core_raw × 1.5
```

| Model | Tiling | Total Records | Per Core (raw) | Per Core (x1.5) | Limit | 판정 |
|:---|:---|---:|---:|---:|---:|:---|
| v10n | 16x16 | 2,741 | 228.4 | **342.6** | 512 | **PASS** |
| v10n | 32x32 | 988 | 82.3 | **123.5** | 512 | **PASS** |
| v10n | Adaptive | 7,265 | 605.4 | **908.1** | 512 | **FAIL** |
| v10s | 16x16 | 2,765 | 230.4 | **345.6** | 512 | **PASS** |
| v10s | 32x32 | 1,000 | 83.3 | **125.0** | 512 | **PASS** |
| v10s | Adaptive | 51,230 | 4,269.2 | **6,403.8** | 512 | **FAIL** |

### 4.4 Queue 분석 요약

- **16x16, 32x32**: 양 모델 모두 PASS. 32x32가 여유분 가장 큼 (v10n: 388.5 records 여유).
- **Adaptive**: 양 모델 모두 FAIL. weight가 큰 레이어에서 T=1이 되면 타일 수가 H×W (최대 6400)로 폭발.
- **근본 원인**: Adaptive tiling은 UB 사용률을 최적화하지만, weight > UB_half인 레이어에서 T=1로 퇴화하여 Q 용량 초과.

## 5. Unified Buffer 분석

### 5.1 타일별 메모리 요구량 계산

각 레이어에 대해 다음을 계산:
```
UB_required = input_tile_bytes + output_tile_bytes + weight_bytes + metadata(512B)
```

Ping-pong double-buffering 사용 시 UB_half = 64 KiB에 fit 해야 함.
Single-buffering 시 UB_full = 128 KiB에 fit.

### 5.2 16x16 타일링 UB 분석

**v10n (16x16): 83 layers 중 52개 fail (UB half), 26개 fail (UB full)**

UB half(64KiB) 초과 주요 원인:
- 40x40 해상도 Conv 3x3 (C=64): input=18x18x64=20,736 + output=16x16x64=16,384 + weight=36,864 = 74,496B > 65,536
- DW stride-2 (C=128~256): input halo 확대로 in_tile 급증
- 1x1 (C_in=256~512): input+output 합이 큰 경우

**v10s (16x16): 87 layers 중 76개 fail (UB half), 56개 fail (UB full)**

v10s는 채널 2배로 weight도 2배 이상. 16x16 타일에서도 weight만 128KB 이상인 레이어 다수.

**UB half 초과 대표 레이어 (v10n, 16x16):**

| Layer | Name | 원인 | UB 요구량 | 초과량 |
|---:|:---|:---|---:|---:|
| 6 | model.3.conv | Conv 3x3 s=2, w=18KB | 70,176 | +4,640 |
| 14 | model.5.cv2.conv | DW s=2, C=128, in_tile 크기 | 173,824 | +108,288 |
| 24 | model.8.m.0.cv1.conv | Conv 3x3, w=144KB | 163,072 | +97,536 |
| 62 | one2one_cv2.1.0 | Conv 3x3, C=128->64, w=72KB | 110,592 | +45,056 |
| 65 | one2one_cv2.2.0 | Conv 3x3, C=256->64, w=144KB | 163,072 | +97,536 |

### 5.3 32x32 타일링 UB 분석

**v10n (32x32): 83 layers 중 80개 fail (UB half), 71개 fail (UB full)**

32x32에서는 input/output tile 자체가 거대해짐:
- 32x32 output tile, C=64: 32*32*64 = 65,536B = 정확히 UB half
- input tile은 halo로 인해 더 크므로 거의 모든 레이어 초과

**v10s (32x32): 87 layers 중 86개 fail (UB half), 84개 fail (UB full)**

**UB half 초과 대표 레이어 (v10n, 32x32):**

| Layer | Name | UB 요구량 | 초과 원인 |
|---:|:---|---:|:---|
| 0 | model.0.conv | 67,984 | out=32x32x16=16KB + in=66x66x3=13KB + w=432B |
| 1 | model.1.conv | 105,488 | stride-2로 in=66x66x16=67.6KB |
| 6 | model.3.conv | 219,680 | stride-2, in=66x66x32=135.2KB + out=65.5KB |
| 12 | model.4.cv2.conv | 135,680 | 1x1, in=128ch + out=64ch at 32x32 |

> 32x32 타일링은 해상도가 40x40 이하인 레이어에서 타일 크기 > 출력 크기가 되어 효과적으로 전체 feature map을 한 번에 처리하려 함. 이 경우 타일링의 의미가 없어지며, 대부분 초과.

### 5.4 Adaptive 타일링 UB 분석

Adaptive 타일링은 UB half에 맞도록 최대 T를 자동 계산. **그러나 weight 자체가 64KiB를 초과하는 레이어에서는 T=1에서도 fit 불가.**

**v10n (Adaptive): 83 layers 중 11개 fail (UB half), 4개 fail (UB full)**

UB half 초과 레이어 (v10n, Adaptive):

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

UB full(128KiB) 초과 레이어 (v10n): layers 24, 25, 28, 65 (weight=131~147KB)

**v10s (Adaptive): 87 layers 중 38개 fail (UB half), 29개 fail (UB full)**

v10s의 critical weight-overflow 레이어:

| Layer | Name | Weight (B) | UB_total (T=1) |
|---:|:---|---:|---:|
| 6 | model.3.conv | 73,728 | 74,944 |
| 15 | model.6.cv1.conv | 65,536 | 66,560 |
| 16-19 | model.6.m.0/1 | 147,456 ea | 149,248 |
| 20-21 | model.6/7.cv2/cv1 | 131,072 ea | 132,352 |
| 23 | model.8.cv1.conv | 262,144 | 263,680 |
| 30 | model.8.cv2.conv | 393,216 | 395,008 |
| 32 | model.9.cv2.conv | 524,288 | 526,336 |
| 33 | model.10.cv1.conv | 262,144 | 263,680 |
| 40 | model.13.cv1.conv | 196,608 | 198,144 |
| 55 | model.22.cv1.conv | 393,216 | 395,008 |
| 62 | model.22.cv2.conv | 393,216 | 395,008 |
| 69 | one2one_cv2.2.0 | 294,912 | 300,096 |

> v10s에서 weight > 128KiB인 레이어가 다수. 이들은 UB full(128KiB)에서도 weight 단독으로 초과하므로, **어떤 타일 크기로도 UB에 적재 불가**.

### 5.5 Adaptive 타일 크기 상세 (v10n 대표)

| 출력 해상도 | 대표 레이어 | Type | C_in->C_out | T | tiles | UB 사용량 |
|:---|:---|:---|:---|---:|---:|---:|
| 320x320 | model.0.conv | Conv s=2 | 3->16 | 47 | 49 | 63,363 |
| 160x160 | model.2.cv1.conv | 1x1 | 32->32 | 31 | 36 | 63,040 |
| 160x160 | model.2.m.0.cv1 | Conv | 16->16 | 43 | 16 | 64,800 |
| 80x80 | model.4.cv1.conv | 1x1 | 64->64 | 21 | 16 | 61,056 |
| 80x80 | model.4.m.0.cv1 | Conv | 32->32 | 28 | 9 | 63,616 |
| 40x40 | model.6.cv1.conv | 1x1 | 128->128 | 13 | 16 | 60,160 |
| 40x40 | model.6.m.0.cv1 | Conv | 64->64 | 13 | 16 | 62,592 |
| 20x20 | model.8.cv1.conv | 1x1 | 256->256 | **1** | **400** | 66,560 (FAIL) |
| 20x20 | model.9.cv1.conv | 1x1 | 256->128 | 9 | 9 | 64,384 |
| 20x20 | model.8.m.0.cv1 | Conv | 128->128 | **1** | **400** | 149,248 (FAIL) |

### 5.6 판정 결과표

| Model | Tiling | Fit Half (64KiB) | Fail Half | Fit Full (128KiB) | Fail Full | 판정 |
|:---|:---|---:|---:|---:|---:|:---|
| v10n | 16x16 | 31/83 | 52 | 57/83 | 26 | **FAIL** |
| v10n | 32x32 | 3/83 | 80 | 12/83 | 71 | **FAIL** |
| v10n | Adaptive | 72/83 | 11 | 79/83 | 4 | **COND** |
| v10s | 16x16 | 11/87 | 76 | 31/87 | 56 | **FAIL** |
| v10s | 32x32 | 1/87 | 86 | 3/87 | 84 | **FAIL** |
| v10s | Adaptive | 49/87 | 38 | 58/87 | 29 | **FAIL** |

> **COND** = 조건부 PASS. v10n Adaptive에서 11개 레이어만 fail. 이들은 weight > 64KiB인 레이어로, weight를 SM에서 직접 streaming하거나 weight tiling 기법을 적용하면 해결 가능.

## 6. Shared Memory 분석

### 6.1 정적 할당 (Weight + Input)

SM에 추론 시작 전 적재해야 할 정적 데이터:

| 항목 | v10n | v10s |
|:---|---:|---:|
| Weight (s8) | 2,292,576 B (2.19 MB) | 7,240,032 B (6.90 MB) |
| Input image (640x640x3 i8) | 1,228,800 B (1.17 MB) | 1,228,800 B (1.17 MB) |
| **정적 합계** | **3,521,376 B (3.36 MB)** | **8,468,832 B (8.08 MB)** |
| SM 용량 (4 MiB) | 4,194,304 B | 4,194,304 B |
| **정적 적합 여부** | **YES** (잔여 673 KB) | **NO** (4.08 MB 초과) |

> **v10s는 weight 단독(6.90MB)이 SM 전체 용량(4MB)을 초과**하여 즉시 FAIL.

### 6.2 레이어 전이 시 Peak 사용량

레이어 실행 중 SM에는 weight(상주) + 현재 입력 FM + 현재 출력 FM이 동시에 존재:

```
peak_SM = weight_total + input_FM_bytes + output_FM_bytes
```

**v10n 레이어 전이 분석 (FAIL하는 3개 레이어):**

| Layer | Name | Input FM | Output FM | Peak | 잔여 | 판정 |
|---:|:---|---:|---:|---:|---:|:---|
| 0 | model.0.conv | 1,228,800 (640x640x3) | 1,638,400 (320x320x16) | **5,159,776 (4.92 MB)** | -965,472 | **FAIL** |
| 1 | model.1.conv | 1,638,400 (320x320x16) | 819,200 (160x160x32) | **4,750,176 (4.53 MB)** | -555,872 | **FAIL** |
| 5 | model.2.cv2.conv | 1,228,800 (160x160x48) | 819,200 (160x160x32) | **4,340,576 (4.14 MB)** | -146,272 | **FAIL** |
| 2 | model.2.cv1.conv | 819,200 (160x160x32) | 819,200 (160x160x32) | 3,930,976 (3.75 MB) | 263,328 | PASS |

- Layer 2부터는 FM 크기가 줄어들어 SM에 적합
- Layer 0~1은 초기 고해상도 FM이 원인
- Layer 5 (model.2.cv2.conv)는 C2f concat 출력(C=48)으로 입력 FM이 큼

**v10n FM 크기 변화 (주요 전환점):**

| 해상도 | 최대 FM 크기 | 비고 |
|:---|---:|:---|
| 640x640x3 | 1.17 MB | 원본 입력 |
| 320x320x16 | 1.56 MB | Layer 0 출력 (최대) |
| 160x160x48 | 1.17 MB | C2f split+cat 입력 |
| 160x160x32 | 0.78 MB | |
| 80x80x192 | 1.17 MB | Neck concat 입력 |
| 80x80x128 | 0.78 MB | |
| 40x40x384 | 0.59 MB | |
| 20x20x512 | 0.20 MB | |

**v10s: 전 레이어 FAIL (87/87)**

v10s는 weight=6.90MB > SM=4MB이므로 모든 레이어에서 peak > 4MB. 최소 peak = 6.90 + 0.025 + 0.025 = 6.95 MB (20x20 최소 FM 레이어).

### 6.3 Skip Connection / Concat 고려

YOLOv10은 residual add, concat, C2f split 등이 있어 복수의 FM이 동시에 SM에 존재해야 하는 경우 발생:

**v10n concat 동시 존재 FM:**
- model.13 (neck): model.8 출력(20x20x256=100KB) + upsample(model.10, 40x40x256=400KB) → concat input 2개 동시 존재: 500KB
- model.16 (neck): model.4 출력(80x80x64=400KB) + upsample(model.13, 80x80x128=800KB) → 1.2MB
- 이 추가 FM은 Layer 0~1 이후에 발생하므로, weight + 동시 FM 합산이 SM 초과 여부 확인 필요

가장 큰 동시 존재 시나리오 (v10n):
```
model.16 concat: weight(2.19MB) + FM_80x80x192(1.17MB) + FM_80x80x128(0.78MB) + FM_80x80x64(0.39MB)
= 2.19 + 1.17 + 0.78 + 0.39 = 4.53 MB > 4 MB  → FAIL
```

실제로 concat의 입력 FM들은 이전 레이어들의 출력이 SM에 잔류하는 형태이므로, **v10n도 neck concat 시점에서 SM 초과 가능성 있음**.

### 6.4 판정 결과표

| Model | Weight SM 적합 | 정적(W+Input) 적합 | Peak 초과 레이어 | 최악 Peak | 판정 |
|:---|:---|:---|---:|:---|:---|
| v10n | YES (2.19 < 4 MB) | YES (3.36 MB) | 3/83 (Layer 0,1,5) | 4.92 MB | **COND** |
| v10s | **NO** (6.90 > 4 MB) | **NO** (8.08 MB) | 87/87 | 11.59 MB | **FAIL** |

> **v10n COND**: Layer 0~1, 5에서 초과. 해결 방안:
> 1. Layer 0~1을 특별 처리 (입력 이미지를 타일 단위로 SM에 적재, 전체 적재 안 함)
> 2. Weight를 SM에서 부분적으로만 적재 (현재 실행 중 레이어 weight만 SM 보유)
> 3. 초기 레이어를 aCore에서 처리 (현재 분석 범위 외)

> **v10s FAIL**: Weight 단독으로 SM 초과. 근본적으로 4 MiB SM에 적재 불가. 해결 불가 (현재 아키텍처 제약 내).

## 7. 종합 교차 분석

### 7.1 6-Case 판정 매트릭스

| Model | Tiling | Q (512 rec) | UB (64/128 KiB) | SM (4 MiB) | **종합** |
|:---|:---|:---|:---|:---|:---|
| v10n | 16x16 | **PASS** (343/512) | **FAIL** (52/83 half fail) | **COND** (3 fail) | **FAIL** |
| v10n | 32x32 | **PASS** (124/512) | **FAIL** (80/83 half fail) | **COND** (3 fail) | **FAIL** |
| v10n | Adaptive | **FAIL** (908/512) | **COND** (11/83 half fail) | **COND** (3 fail) | **FAIL** |
| v10s | 16x16 | **PASS** (346/512) | **FAIL** (76/87 half fail) | **FAIL** (weight>SM) | **FAIL** |
| v10s | 32x32 | **PASS** (125/512) | **FAIL** (86/87 half fail) | **FAIL** (weight>SM) | **FAIL** |
| v10s | Adaptive | **FAIL** (6404/512) | **FAIL** (38/87 half fail) | **FAIL** (weight>SM) | **FAIL** |

> **6개 케이스 모두 종합 FAIL.** 3개 자원(Q, UB, SM) 모두 동시에 PASS인 조합이 없음.

### 7.2 병목 분석

**Q 병목:**
- 16x16, 32x32에서는 문제 없음
- Adaptive에서만 FAIL: weight > UB_half인 레이어가 T=1로 퇴화하여 tile 수 폭발

**UB 병목 (가장 심각):**
- 고정 타일(16x16, 32x32)에서는 대다수 레이어가 weight + tile 합산으로 초과
- Adaptive에서도 weight > 64KiB 레이어는 해결 불가
- v10n: 11개 레이어 (weight 64~147KB), v10s: 38개 레이어 (weight 64~524KB)
- **근본 원인: UB 128KiB에 weight + input_tile + output_tile을 모두 담아야 한다는 가정**

**SM 병목:**
- v10s: Weight 자체(6.90MB)가 SM(4MB) 초과 → 물리적 불가
- v10n: 초기 고해상도 레이어(0~1, 5)에서만 초과 → 우회 가능

### 7.3 자원별 실현 가능성 순위

| 순위 | 자원 | 실현성 | 비고 |
|---:|:---|:---|:---|
| 1 | Q | 높음 | 16x16/32x32에서 양 모델 PASS. Adaptive FAIL은 타일링 전략 문제 |
| 2 | SM (v10n) | 중간 | 정적 할당 OK, 초기 3개 레이어만 우회 필요 |
| 3 | UB (v10n Adaptive) | 중간-낮음 | 11개 레이어만 fail, weight streaming으로 해결 가능성 |
| 4 | SM (v10s) | 불가 | Weight 자체가 SM 초과. 외부 메모리 없이 해결 불가 |

## 8. 결론 및 권고사항

### 8.1 주요 결론

1. **YOLOv10s는 현재 아키텍처에서 실행 불가능.** Weight(6.90MB)이 SM(4MiB)을 72.5% 초과하며, 어떤 타일링 전략으로도 해결할 수 없음.

2. **YOLOv10n은 조건부 실현 가능.** 다만 3가지 자원이 동시에 PASS하는 단일 타일링 전략은 존재하지 않음:
   - 16x16/32x32: Q는 PASS, UB는 FAIL, SM은 COND
   - Adaptive: Q는 FAIL, UB는 COND, SM은 COND

3. **UB가 가장 심각한 병목.** Weight를 UB에 동시 적재해야 한다는 전제가 대부분의 FAIL 원인.

### 8.2 권고사항

**단기 (현재 아키텍처 내):**

1. **Hybrid 타일링**: 16x16 기본 + weight > 64KiB 레이어만 별도 처리
   - Q: 16x16 기준 PASS (342/512)
   - UB: weight streaming (SM에서 weight를 tile 단위로 DMA) 적용 시 대부분 해결 가능
   - SM: Layer 0~1 특별 처리 (입력 타일 단위 적재)

2. **Weight Streaming 기법 검토**: Weight를 UB에 전량 올리지 않고, SM에서 필요한 부분만 streaming
   - 이 경우 UB 요구량 = input_tile + output_tile + weight_slice + metadata
   - Weight를 C_out 방향으로 slicing하면 대부분 해결 가능

3. **v10n Layer 0~1 전용 처리**: 초기 고해상도 레이어를 aCore에서 처리하거나, 입력을 타일 단위로 SM에 적재

**중장기 (아키텍처 변경 시):**

4. **SM 확장**: 8 MiB 이상으로 확장 시 v10s도 수용 가능 (weight 6.90 + input 1.17 = 8.07 MB)
5. **External Memory (eFlash/DRAM) 활용**: Weight를 외부 메모리에 두고 layer 단위로 SM에 적재
6. **UB 확장 또는 Weight 전용 버퍼 추가**: Weight를 UB와 별도 경로로 공급

### 8.3 v10n 실현 가능 시나리오 (Best-case)

```
전략: 16x16 고정 타일 + Weight Streaming + Layer 0~1 특별처리

Q:  PASS (342.6 / 512 records, 여유 33%)
UB: Weight streaming 적용 시 → weight를 SM에서 직접 참조 또는 slice 적재
    input_tile(16x16) + output_tile(16x16) = 최대 ~32KB (C=128 기준)
    + weight_slice + metadata → 대부분 64KiB 이내
SM: Weight 2.19MB + Input 1.17MB = 3.36MB
    Layer 0 출력 FM: 1.56MB → weight+input+output = 4.92MB > 4MB
    → Layer 0 입력을 타일 단위로 SM에 적재 (전체 1.17MB 대신 타일만)
    → 또는 Layer 0~1을 aCore에서 처리
```

이 시나리오에서 v10n은 실현 가능하나, **Weight Streaming 메커니즘이 ISA/하드웨어 레벨에서 지원되어야 함.**

---

## 부록 A: Adaptive 타일 크기 전체 목록 (v10n)

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

## 부록 B: Adaptive 타일 크기 전체 목록 (v10s)

| Layer | Name | Out HxW | T | Tiles | UB Total | Fit Half |
|---:|:---|:---|---:|---:|---:|:---|
| 0 | model.0.conv | 320x320 | 38 | 81 | 65,371 | YES |
| 1 | model.1.conv | 160x160 | 15 | 121 | 64,096 | YES |
| 2 | model.2.cv1.conv | 160x160 | 21 | 64 | 61,056 | YES |
| 3 | model.2.m.0.cv1.conv | 160x160 | 28 | 36 | 63,616 | YES |
| 4 | model.2.m.0.cv2.conv | 160x160 | 28 | 36 | 63,616 | YES |
| 5 | model.2.cv2.conv | 160x160 | 19 | 81 | 64,416 | YES |
| 6 | model.3.conv | 80x80 | 1 | 6,400 | 74,944 | **NO** |
| 7 | model.4.cv1.conv | 80x80 | 13 | 49 | 60,160 | YES |
| 8 | model.4.m.0.cv1.conv | 80x80 | 13 | 49 | 62,592 | YES |
| 9 | model.4.m.0.cv2.conv | 80x80 | 13 | 49 | 62,592 | YES |
| 10 | model.4.m.1.cv1.conv | 80x80 | 13 | 49 | 62,592 | YES |
| 11 | model.4.m.1.cv2.conv | 80x80 | 13 | 49 | 62,592 | YES |
| 12 | model.4.cv2.conv | 80x80 | 9 | 81 | 64,384 | YES |
| 13 | model.5.cv1.conv | 80x80 | 9 | 81 | 64,384 | YES |
| 14 | model.5.cv2.conv | 40x40 | 6 | 49 | 55,296 | YES |
| 15 | model.6.cv1.conv | 40x40 | 1 | 1,600 | 66,560 | **NO** |
| 16 | model.6.m.0.cv1.conv | 40x40 | 1 | 1,600 | 149,248 | **NO** |
| 17 | model.6.m.0.cv2.conv | 40x40 | 1 | 1,600 | 149,248 | **NO** |
| 18 | model.6.m.1.cv1.conv | 40x40 | 1 | 1,600 | 149,248 | **NO** |
| 19 | model.6.m.1.cv2.conv | 40x40 | 1 | 1,600 | 149,248 | **NO** |
| 20 | model.6.cv2.conv | 40x40 | 1 | 1,600 | 132,352 | **NO** |
| 21 | model.7.cv1.conv | 40x40 | 1 | 1,600 | 132,352 | **NO** |
| 22 | model.7.cv2.conv | 20x20 | 4 | 25 | 54,784 | YES |
| 23 | model.8.cv1.conv | 20x20 | 1 | 400 | 263,680 | **NO** |
| 24 | model.8.m.0.cv1.0.conv | 20x20 | 10 | 4 | 65,280 | YES |
| 25 | model.8.m.0.cv1.1.conv | 20x20 | 1 | 400 | 132,352 | **NO** |
| 26 | model.8.m.0.cv1.2.conv.conv | 20x20 | 2 | 100 | 60,416 | YES |
| 27 | model.8.m.0.cv1.2.conv1.conv | 20x20 | 6 | 16 | 56,320 | YES |
| 28 | model.8.m.0.cv1.3.conv | 20x20 | 1 | 400 | 132,352 | **NO** |
| 29 | model.8.m.0.cv1.4.conv | 20x20 | 10 | 4 | 65,280 | YES |
| 30 | model.8.cv2.conv | 20x20 | 1 | 400 | 395,008 | **NO** |
| 31 | model.9.cv1.conv | 20x20 | 1 | 400 | 132,352 | **NO** |
| 32 | model.9.cv2.conv | 20x20 | 1 | 400 | 526,336 | **NO** |
| 33 | model.10.cv1.conv | 20x20 | 1 | 400 | 263,680 | **NO** |
| 34 | model.10.attn.qkv.conv | 20x20 | 1 | 400 | 132,352 | **NO** |
| 35 | model.10.attn.pe.conv | 20x20 | 10 | 4 | 65,280 | YES |
| 36 | model.10.attn.proj.conv | 20x20 | 1 | 400 | 66,560 | **NO** |
| 37 | model.10.ffn.0.conv | 20x20 | 1 | 400 | 132,352 | **NO** |
| 38 | model.10.ffn.1.conv | 20x20 | 1 | 400 | 132,352 | **NO** |
| 39 | model.10.cv2.conv | 20x20 | 1 | 400 | 263,680 | **NO** |
| 40 | model.13.cv1.conv | 40x40 | 1 | 1,600 | 198,144 | **NO** |
| 41 | model.13.m.0.cv1.conv | 40x40 | 1 | 1,600 | 149,248 | **NO** |
| 42 | model.13.m.0.cv2.conv | 40x40 | 1 | 1,600 | 149,248 | **NO** |
| 43 | model.13.cv2.conv | 40x40 | 1 | 1,600 | 99,456 | **NO** |
| 44 | model.16.cv1.conv | 80x80 | 5 | 256 | 62,464 | YES |
| 45 | model.16.m.0.cv1.conv | 80x80 | 13 | 49 | 62,592 | YES |
| 46 | model.16.m.0.cv2.conv | 80x80 | 13 | 49 | 62,592 | YES |
| 47 | model.16.cv2.conv | 80x80 | 11 | 64 | 63,808 | YES |
| 48 | model.17.conv | 40x40 | 1 | 1,600 | 149,248 | **NO** |
| 49 | model.19.cv1.conv | 40x40 | 1 | 1,600 | 99,456 | **NO** |
| 50 | model.19.m.0.cv1.conv | 40x40 | 1 | 1,600 | 149,248 | **NO** |
| 51 | model.19.m.0.cv2.conv | 40x40 | 1 | 1,600 | 149,248 | **NO** |
| 52 | model.19.cv2.conv | 40x40 | 1 | 1,600 | 99,456 | **NO** |
| 53 | model.20.cv1.conv | 40x40 | 1 | 1,600 | 66,560 | **NO** |
| 54 | model.20.cv2.conv | 20x20 | 6 | 16 | 55,296 | YES |
| 55 | model.22.cv1.conv | 20x20 | 1 | 400 | 395,008 | **NO** |
| 56 | model.22.m.0.cv1.0.conv | 20x20 | 10 | 4 | 65,280 | YES |
| 57 | model.22.m.0.cv1.1.conv | 20x20 | 1 | 400 | 132,352 | **NO** |
| 58 | model.22.m.0.cv1.2.conv.conv | 20x20 | 2 | 100 | 60,416 | YES |
| 59 | model.22.m.0.cv1.2.conv1.conv | 20x20 | 6 | 16 | 56,320 | YES |
| 60 | model.22.m.0.cv1.3.conv | 20x20 | 1 | 400 | 132,352 | **NO** |
| 61 | model.22.m.0.cv1.4.conv | 20x20 | 10 | 4 | 65,280 | YES |
| 62 | model.22.cv2.conv | 20x20 | 1 | 400 | 395,008 | **NO** |
| 63 | one2one_cv2.0.0 | 80x80 | 1 | 6,400 | 75,456 | **NO** |
| 64 | one2one_cv2.0.1 | 80x80 | 13 | 49 | 62,592 | YES |
| 65 | one2one_cv2.0.2 | 80x80 | 21 | 16 | 61,056 | YES |
| 66 | one2one_cv2.1.0 | 40x40 | 1 | 1,600 | 150,336 | **NO** |
| 67 | one2one_cv2.1.1 | 40x40 | 13 | 16 | 62,592 | YES |
| 68 | one2one_cv2.1.2 | 40x40 | 21 | 4 | 61,056 | YES |
| 69 | one2one_cv2.2.0 | 20x20 | 1 | 400 | 300,096 | **NO** |
| 70 | one2one_cv2.2.1 | 20x20 | 13 | 4 | 62,592 | YES |
| 71 | one2one_cv2.2.2 | 20x20 | 20 | 1 | 55,808 | YES |
| 72 | one2one_cv3.0.0.0 | 80x80 | 14 | 36 | 59,520 | YES |
| 73 | one2one_cv3.0.0.1 | 80x80 | 13 | 49 | 60,160 | YES |
| 74 | one2one_cv3.0.1.0 | 80x80 | 14 | 36 | 59,520 | YES |
| 75 | one2one_cv3.0.1.1 | 80x80 | 13 | 49 | 60,160 | YES |
| 76 | one2one_cv3.0.2 | 80x80 | 16 | 25 | 64,000 | YES |
| 77 | one2one_cv3.1.0.0 | 40x40 | 10 | 16 | 65,280 | YES |
| 78 | one2one_cv3.1.0.1 | 40x40 | 9 | 25 | 64,384 | YES |
| 79 | one2one_cv3.1.1.0 | 40x40 | 14 | 9 | 59,520 | YES |
| 80 | one2one_cv3.1.1.1 | 40x40 | 13 | 16 | 60,160 | YES |
| 81 | one2one_cv3.1.2 | 40x40 | 16 | 9 | 64,000 | YES |
| 82 | one2one_cv3.2.0.0 | 20x20 | 6 | 16 | 56,320 | YES |
| 83 | one2one_cv3.2.0.1 | 20x20 | 1 | 400 | 66,688 | **NO** |
| 84 | one2one_cv3.2.1.0 | 20x20 | 14 | 4 | 59,520 | YES |
| 85 | one2one_cv3.2.1.1 | 20x20 | 13 | 4 | 60,160 | YES |
| 86 | one2one_cv3.2.2 | 20x20 | 16 | 4 | 64,000 | YES |

## 부록 C: SM 레이어별 Peak 사용량 (v10n, FAIL 레이어)

| Layer | Name | Input FM (bytes) | Output FM (bytes) | Peak = Weight + In + Out | SM (4 MiB) | Delta |
|---:|:---|---:|---:|---:|---:|---:|
| 0 | model.0.conv | 1,228,800 | 1,638,400 | 5,159,776 | 4,194,304 | **-965,472** |
| 1 | model.1.conv | 1,638,400 | 819,200 | 4,749,776 | 4,194,304 | **-555,472** |
| 5 | model.2.cv2.conv | 1,228,800 | 819,200 | 4,340,576 | 4,194,304 | **-146,272** |

다음 레이어부터 PASS:

| Layer | Name | Peak | 잔여 |
|---:|:---|---:|---:|
| 2 | model.2.cv1.conv | 3,930,976 (3.75 MB) | 263,328 |
| 6 | model.3.conv | 3,930,208 (3.75 MB) | 264,096 |
| 23 | model.8.cv1.conv | 2,792,576 (2.66 MB) | 1,401,728 |
| 82 | one2one_cv3.2.2 | 2,355,376 (2.25 MB) | 1,838,928 |

---

**참조 파일:**
- Conv Layer Analysis: `docs/agent_results/etc/20260401_yolov10_conv_layer_analysis.md`
- CSV (Queue): `docs/agent_results/reports/20260401_queue_analysis.csv`
- CSV (UB): `docs/agent_results/reports/20260401_ub_analysis.csv`
- CSV (SM): `docs/agent_results/reports/20260401_sm_analysis.csv`
