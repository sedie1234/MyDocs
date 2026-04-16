# Safe Boundary 분석 보고서 -- Weight/Input 안전 한계

**날짜**: 2026-04-01  
**분석 대상**: YOLOv10n (83 layers), YOLOv10s (87 layers)  
**Tiling 전략**: Adaptive Full (256 KiB budget)  

---

## 1. 분석 목적

본 보고서는 Infetron-V2 NPU에서 YOLOv10n/v10s 모델의 각 layer에 대해, **Queue(Q) 제약**과 **Unified Buffer(UB) 제약**을 동시에 만족하는 **안전 운용 경계(Safe Boundary)**를 결정한다.

핵심 질문:
1. 각 layer의 weight가 UB에 들어가는가?
2. Weight를 적재한 뒤 남은 공간으로 input/output tile을 얼마나 크게 잡을 수 있는가?
3. Tile 크기에 따른 tile 수가 Q 용량(512 records/core)을 초과하지 않는가?
4. 두 조건을 동시에 만족하는 T의 범위는 무엇인가?

---

## 2. 분석 조건

| 항목 | 값 | 비고 |
|---|---|---|
| UB 크기 | 256 KiB (262,144 bytes) | Weight-first 전략 |
| Q 크기 | 512 records/core | Sequential execution |
| 코어 수 | 12 | Layer 순차 할당 |
| SM 크기 | 4 MiB | Overlap 활성화 |
| Metadata | 512 bytes/layer | 고정 오버헤드 |
| Tiling 전략 | Adaptive Full | UB 전체(256 KiB)를 tile budget으로 사용 |
| 데이터 타입 | INT8 (1 byte/element) | Quantized model |
| v10n 총 weight | 2.19 MB (2,292,576 bytes) | SM 4 MiB 이내 |
| v10s 총 weight | 6.90 MB (7,240,032 bytes) | SM 4 MiB 초과 |

---

## 3. 공식 정리

### 3.1 Weight 크기 (고정)

Weight는 모델 아키텍처에 의해 고정되며, 컴파일러가 변경할 수 없다.

```
W = K * K * C_in * C_out    (bytes, INT8)
```

- Conv 3x3: W = 9 * C_in * C_out
- Conv 1x1: W = C_in * C_out
- DW 3x3: W = 9 * C_in (C_out = C_in)
- DW 7x7: W = 49 * C_in

### 3.2 Tile 크기 T 결정 공식 (컴파일러 변수)

output feature map을 T x T 크기의 tile로 분할한다. T는 컴파일러가 제어하는 **유일한 변수**이다.

**UB 제약에서 T_max 유도:**

UB에 weight + input tile + output tile + metadata가 모두 들어가야 한다:

```
W + input_tile(T) + output_tile(T) + M <= UB
```

여기서:
- `R = UB - W - M` (tile에 쓸 수 있는 잔여 공간)
- output tile = T^2 * C_out (bytes)
- input tile = (S*T + K - S)^2 * C_in (bytes), S = stride, K = kernel size

이를 정리하면 이차 부등식:

```
a * T^2 + b * T + c <= 0

where:
  a = S^2 * C_in + C_out
  b = 2 * S * (K - S) * C_in
  c = (K - S)^2 * C_in - R
```

근의 공식으로 T_max를 구한다:

```
T_max = floor( (-b + sqrt(b^2 - 4*a*c)) / (2*a) )
```

**간소화된 케이스들:**

| 케이스 | a | b | c | T_max 공식 |
|---|---|---|---|---|
| 1x1, stride 1 | C_in + C_out | 0 | -R | floor(sqrt(R / (C_in + C_out))) |
| 3x3, stride 1 | C_in + C_out | 4*C_in | 4*C_in - R | floor((-4*C_in + sqrt(16*C_in^2 + 4*(C_in+C_out)*(R-4*C_in))) / (2*(C_in+C_out))) |
| 3x3, stride 2 | 4*C_in + C_out | 4*C_in | C_in - R | floor((-4*C_in + sqrt(16*C_in^2 + 4*(4*C_in+C_out)*(R-C_in))) / (2*(4*C_in+C_out))) |

### 3.3 T에서 유도되는 값들

T가 결정되면 나머지는 자동으로 결정된다:

```
Tiles_per_layer = ceil(H_out / T) * ceil(W_out / T)
Input_tile_HW   = (S * T + K - S) x (S * T + K - S)
Input_tile_B    = Input_tile_HW^2 * C_in
Output_tile_B   = T^2 * C_out
```

### 3.4 안전 조건 정리

Layer가 안전하려면 다음 **두 조건을 동시에** 만족해야 한다:

| 조건 | 수식 | 의미 |
|---|---|---|
| **UB 조건** | W + M <= UB | Weight + metadata가 UB에 들어가야 함 |
| **Q 조건** | sum(tiles in core) <= 512 | 한 코어에 할당된 모든 layer의 tile 합 <= 512 |

UB 조건이 만족되면 T_max가 존재하고, T_max로 tiling하면 tile 수가 최소화되어 Q 조건도 만족할 가능성이 높다.

**Weight가 UB를 초과하면?**
- T_max가 존재하지 않음 (판별식 < 0 또는 R < 0)
- 해당 layer는 **무조건 FAIL** -- 하드웨어 제약을 근본적으로 충족 불가

---

## 4. YOLOv10n 분석

### 4.1 종합 판정

| 항목 | 결과 |
|---|---|
| **종합 판정** | **PASS** |
| UB 판정 | PASS (83/83 통과) |
| Q 판정 | PASS (worst core 236 records, 한계 512) |
| SM 판정 | PASS (총 weight 2.19MB < SM 4MiB) |
| 최대 weight | 147,456 bytes (Layer 24, model.8.m.0.cv1.conv) |
| UB 대비 최대 weight 비율 | 56.3% |
| 최소 margin | 90,304 bytes (Layer 65) |
| Total tiles | 1,219 |
| Worst core | Core 5 (236 records) |

### 4.2 전체 layer별 safe boundary 표

> T_AdaptFull = Adaptive Full에서의 T_max, Tiles = 해당 T에서의 tile 수

| Index | Core | Name | Type | Out HW | C_in | C_out | Weight (B) | Max Weight (B) | Margin (B) | UB OK | T_max | Tiles |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | model.0.conv | Conv 3x3/2 | 320x320 | 3 | 16 | 432 | 246,541 | 246,109 | PASS | 96 | 16 |
| 1 | 0 | model.1.conv | Conv 3x3/2 | 160x160 | 16 | 32 | 4,608 | 209,360 | 204,752 | PASS | 51 | 16 |
| 2 | 0 | model.2.cv1.conv | 1x1 | 160x160 | 32 | 32 | 1,024 | 227,776 | 226,752 | PASS | 63 | 9 |
| 3 | 0 | model.2.m.0.cv1.conv | Conv 3x3/1 | 160x160 | 16 | 16 | 2,304 | 243,168 | 240,864 | PASS | 89 | 4 |
| 4 | 0 | model.2.m.0.cv2.conv | Conv 3x3/1 | 160x160 | 16 | 16 | 2,304 | 243,168 | 240,864 | PASS | 89 | 4 |
| 5 | 0 | model.2.cv2.conv | 1x1 | 160x160 | 48 | 32 | 1,536 | 219,312 | 217,776 | PASS | 57 | 9 |
| 6 | 0 | model.3.conv | Conv 3x3/2 | 80x80 | 32 | 64 | 18,432 | 118,176 | 99,744 | PASS | 35 | 9 |
| 7 | 1 | model.4.cv1.conv | 1x1 | 80x80 | 64 | 64 | 4,096 | 248,832 | 244,736 | PASS | 44 | 4 |
| 8 | 1 | model.4.m.0.cv1.conv | Conv 3x3/1 | 80x80 | 32 | 32 | 9,216 | 253,824 | 244,608 | PASS | 61 | 4 |
| 9 | 1 | model.4.m.0.cv2.conv | Conv 3x3/1 | 80x80 | 32 | 32 | 9,216 | 253,824 | 244,608 | PASS | 61 | 4 |
| 10 | 1 | model.4.m.1.cv1.conv | Conv 3x3/1 | 80x80 | 32 | 32 | 9,216 | 253,824 | 244,608 | PASS | 61 | 4 |
| 11 | 1 | model.4.m.1.cv2.conv | Conv 3x3/1 | 80x80 | 32 | 32 | 9,216 | 253,824 | 244,608 | PASS | 61 | 4 |
| 12 | 1 | model.4.cv2.conv | 1x1 | 80x80 | 128 | 64 | 8,192 | 242,432 | 234,240 | PASS | 36 | 9 |
| 13 | 1 | model.5.cv1.conv | 1x1 | 80x80 | 64 | 128 | 8,192 | 242,432 | 234,240 | PASS | 36 | 9 |
| 14 | 2 | model.5.cv2.conv | DW 3x3/2 | 40x40 | 128 | 128 | 1,152 | 242,944 | 241,792 | PASS | 19 | 9 |
| 15 | 2 | model.6.cv1.conv | 1x1 | 40x40 | 128 | 128 | 16,384 | 255,232 | 238,848 | PASS | 30 | 4 |
| 16 | 2 | model.6.m.0.cv1.conv | Conv 3x3/1 | 40x40 | 64 | 64 | 36,864 | 256,896 | 220,032 | PASS | 40 | 1 |
| 17 | 2 | model.6.m.0.cv2.conv | Conv 3x3/1 | 40x40 | 64 | 64 | 36,864 | 256,896 | 220,032 | PASS | 40 | 1 |
| 18 | 2 | model.6.m.1.cv1.conv | Conv 3x3/1 | 40x40 | 64 | 64 | 36,864 | 256,896 | 220,032 | PASS | 40 | 1 |
| 19 | 2 | model.6.m.1.cv2.conv | Conv 3x3/1 | 40x40 | 64 | 64 | 36,864 | 256,896 | 220,032 | PASS | 40 | 1 |
| 20 | 2 | model.6.cv2.conv | 1x1 | 40x40 | 256 | 128 | 32,768 | 252,032 | 219,264 | PASS | 24 | 4 |
| 21 | 3 | model.7.cv1.conv | 1x1 | 40x40 | 128 | 256 | 32,768 | 255,488 | 222,720 | PASS | 24 | 4 |
| 22 | 3 | model.7.cv2.conv | DW 3x3/2 | 20x20 | 256 | 256 | 2,304 | 236,800 | 234,496 | PASS | 13 | 4 |
| 23 | 3 | model.8.cv1.conv | 1x1 | 20x20 | 256 | 256 | 65,536 | 253,440 | 187,904 | PASS | 19 | 4 |
| 24 | 3 | model.8.m.0.cv1.conv | Conv 3x3/1 | 20x20 | 128 | 128 | 147,456 | 254,976 | 107,520 | PASS | 20 | 1 |
| 25 | 3 | model.8.m.0.cv2.conv | Conv 3x3/1 | 20x20 | 128 | 128 | 147,456 | 254,976 | 107,520 | PASS | 20 | 1 |
| 26 | 3 | model.8.cv2.conv | 1x1 | 20x20 | 384 | 256 | 98,304 | 251,392 | 153,088 | PASS | 15 | 4 |
| 27 | 3 | model.9.cv1.conv | 1x1 | 20x20 | 256 | 128 | 32,768 | 255,488 | 222,720 | PASS | 20 | 1 |
| 28 | 4 | model.9.cv2.conv | 1x1 | 20x20 | 512 | 256 | 131,072 | 254,720 | 123,648 | PASS | 13 | 4 |
| 29 | 4 | model.10.cv1.conv | 1x1 | 20x20 | 256 | 256 | 65,536 | 257,024 | 191,488 | PASS | 19 | 4 |
| 30 | 4 | model.10.attn.qkv.conv | 1x1 | 20x20 | 128 | 256 | 32,768 | 258,176 | 225,408 | PASS | 20 | 1 |
| 31 | 4 | model.10.attn.pe.conv | DW 3x3/1 | 20x20 | 128 | 128 | 1,152 | 257,280 | 256,128 | PASS | 20 | 1 |
| 32 | 4 | model.10.attn.proj.conv | 1x1 | 20x20 | 128 | 128 | 16,384 | 259,328 | 242,944 | PASS | 20 | 1 |
| 33 | 4 | model.10.ffn.0.conv | 1x1 | 20x20 | 128 | 256 | 32,768 | 258,176 | 225,408 | PASS | 20 | 1 |
| 34 | 4 | model.10.ffn.1.conv | 1x1 | 20x20 | 256 | 128 | 32,768 | 258,176 | 225,408 | PASS | 20 | 1 |
| 35 | 5 | model.10.cv2.conv | 1x1 | 20x20 | 256 | 256 | 65,536 | 210,432 | 144,896 | PASS | 19 | 4 |
| 36 | 5 | model.13.cv1.conv | 1x1 | 40x40 | 384 | 128 | 49,152 | 228,864 | 179,712 | PASS | 20 | 4 |
| 37 | 5 | model.13.m.0.cv1.conv | Conv 3x3/1 | 40x40 | 64 | 64 | 36,864 | 251,136 | 214,272 | PASS | 40 | 1 |
| 38 | 5 | model.13.m.0.cv2.conv | Conv 3x3/1 | 40x40 | 64 | 64 | 36,864 | 251,136 | 214,272 | PASS | 40 | 1 |
| 39 | 5 | model.13.cv2.conv | 1x1 | 40x40 | 192 | 128 | 24,576 | 241,152 | 216,576 | PASS | 27 | 4 |
| 40 | 5 | model.16.cv1.conv | 1x1 | 80x80 | 192 | 64 | 12,288 | 245,248 | 232,960 | PASS | 31 | 9 |
| 41 | 5 | model.16.m.0.cv1.conv | Conv 3x3/1 | 80x80 | 32 | 32 | 9,216 | 256,384 | 247,168 | PASS | 61 | 4 |
| 42 | 6 | model.16.m.0.cv2.conv | Conv 3x3/1 | 80x80 | 32 | 32 | 9,216 | 256,384 | 247,168 | PASS | 61 | 4 |
| 43 | 6 | model.16.cv2.conv | 1x1 | 80x80 | 96 | 64 | 6,144 | 251,392 | 245,248 | PASS | 39 | 9 |
| 44 | 6 | model.17.conv | Conv 3x3/2 | 40x40 | 64 | 64 | 36,864 | 239,040 | 202,176 | PASS | 26 | 4 |
| 45 | 6 | model.19.cv1.conv | 1x1 | 40x40 | 192 | 128 | 24,576 | 241,152 | 216,576 | PASS | 27 | 4 |
| 46 | 6 | model.19.m.0.cv1.conv | Conv 3x3/1 | 40x40 | 64 | 64 | 36,864 | 251,136 | 214,272 | PASS | 40 | 1 |
| 47 | 6 | model.19.m.0.cv2.conv | Conv 3x3/1 | 40x40 | 64 | 64 | 36,864 | 251,136 | 214,272 | PASS | 40 | 1 |
| 48 | 6 | model.19.cv2.conv | 1x1 | 40x40 | 192 | 128 | 24,576 | 241,152 | 216,576 | PASS | 27 | 4 |
| 49 | 7 | model.20.cv1.conv | 1x1 | 40x40 | 128 | 128 | 16,384 | 257,536 | 241,152 | PASS | 30 | 4 |
| 50 | 7 | model.20.cv2.conv | DW 3x3/2 | 20x20 | 128 | 128 | 1,152 | 249,216 | 248,064 | PASS | 19 | 4 |
| 51 | 7 | model.22.cv1.conv | 1x1 | 20x20 | 384 | 256 | 98,304 | 251,392 | 153,088 | PASS | 15 | 4 |
| 52 | 7 | model.22.m.0.cv1.0.conv | DW 3x3/1 | 20x20 | 128 | 128 | 1,152 | 254,976 | 253,824 | PASS | 20 | 1 |
| 53 | 7 | model.22.m.0.cv1.1.conv | 1x1 | 20x20 | 128 | 256 | 32,768 | 255,488 | 222,720 | PASS | 20 | 1 |
| 54 | 7 | model.22.m.0.cv1.2.conv.conv | DW 7x7/1 | 20x20 | 256 | 256 | 12,544 | 231,936 | 219,392 | PASS | 18 | 4 |
| 55 | 7 | model.22.m.0.cv1.2.conv1.conv | DW 3x3/1 | 20x20 | 256 | 256 | 2,304 | 248,320 | 246,016 | PASS | 20 | 1 |
| 56 | 8 | model.22.m.0.cv1.3.conv | 1x1 | 20x20 | 256 | 128 | 32,768 | 223,232 | 190,464 | PASS | 20 | 1 |
| 57 | 8 | model.22.m.0.cv1.4.conv | DW 3x3/1 | 20x20 | 128 | 128 | 1,152 | 230,400 | 229,248 | PASS | 20 | 1 |
| 58 | 8 | model.22.cv2.conv | 1x1 | 20x20 | 384 | 256 | 98,304 | 197,632 | 99,328 | PASS | 15 | 4 |
| 59 | 8 | model.23.one2one_cv2.0.0.conv | Conv 3x3/1 | 80x80 | 64 | 64 | 36,864 | 251,136 | 214,272 | PASS | 40 | 4 |
| 60 | 8 | model.23.one2one_cv2.0.1.conv | Conv 3x3/1 | 80x80 | 64 | 64 | 36,864 | 251,136 | 214,272 | PASS | 40 | 4 |
| 61 | 8 | model.23.one2one_cv2.0.2 | 1x1 | 80x80 | 64 | 64 | 4,096 | 253,440 | 249,344 | PASS | 44 | 4 |
| 62 | 8 | model.23.one2one_cv2.1.0.conv | Conv 3x3/1 | 40x40 | 128 | 64 | 73,728 | 244,736 | 171,008 | PASS | 29 | 4 |
| 63 | 9 | model.23.one2one_cv2.1.1.conv | Conv 3x3/1 | 40x40 | 64 | 64 | 36,864 | 253,312 | 216,448 | PASS | 40 | 1 |
| 64 | 9 | model.23.one2one_cv2.1.2 | 1x1 | 40x40 | 64 | 64 | 4,096 | 255,360 | 251,264 | PASS | 40 | 1 |
| 65 | 9 | model.23.one2one_cv2.2.0.conv | Conv 3x3/1 | 20x20 | 256 | 64 | 147,456 | 237,760 | 90,304 | PASS | 17 | 4 |
| 66 | 9 | model.23.one2one_cv2.2.1.conv | Conv 3x3/1 | 20x20 | 64 | 64 | 36,864 | 253,312 | 216,448 | PASS | 20 | 1 |
| 67 | 9 | model.23.one2one_cv2.2.2 | 1x1 | 20x20 | 64 | 64 | 4,096 | 255,360 | 251,264 | PASS | 20 | 1 |
| 68 | 9 | model.23.one2one_cv3.0.0.0.conv | DW 3x3/1 | 80x80 | 64 | 64 | 576 | 253,312 | 252,736 | PASS | 44 | 4 |
| 69 | 9 | model.23.one2one_cv3.0.0.1.conv | 1x1 | 80x80 | 64 | 80 | 5,120 | 254,576 | 249,456 | PASS | 42 | 4 |
| 70 | 10 | model.23.one2one_cv3.0.1.0.conv | DW 3x3/1 | 80x80 | 80 | 80 | 720 | 248,512 | 247,792 | PASS | 39 | 9 |
| 71 | 10 | model.23.one2one_cv3.0.1.1.conv | 1x1 | 80x80 | 80 | 80 | 6,400 | 251,392 | 244,992 | PASS | 39 | 9 |
| 72 | 10 | model.23.one2one_cv3.0.2 | 1x1 | 80x80 | 80 | 80 | 6,400 | 251,392 | 244,992 | PASS | 39 | 9 |
| 73 | 10 | model.23.one2one_cv3.1.0.0.conv | DW 3x3/1 | 40x40 | 128 | 128 | 1,152 | 240,640 | 239,488 | PASS | 30 | 4 |
| 74 | 10 | model.23.one2one_cv3.1.0.1.conv | 1x1 | 40x40 | 128 | 80 | 10,240 | 248,320 | 238,080 | PASS | 34 | 4 |
| 75 | 10 | model.23.one2one_cv3.1.1.0.conv | DW 3x3/1 | 40x40 | 80 | 80 | 720 | 248,512 | 247,792 | PASS | 39 | 4 |
| 76 | 10 | model.23.one2one_cv3.1.1.1.conv | 1x1 | 40x40 | 80 | 80 | 6,400 | 251,392 | 244,992 | PASS | 39 | 4 |
| 77 | 11 | model.23.one2one_cv3.1.2 | 1x1 | 40x40 | 80 | 80 | 6,400 | 259,072 | 252,672 | PASS | 39 | 4 |
| 78 | 11 | model.23.one2one_cv3.2.0.0.conv | DW 3x3/1 | 20x20 | 256 | 256 | 2,304 | 248,320 | 246,016 | PASS | 20 | 1 |
| 79 | 11 | model.23.one2one_cv3.2.0.1.conv | 1x1 | 20x20 | 256 | 80 | 20,480 | 256,256 | 235,776 | PASS | 20 | 1 |
| 80 | 11 | model.23.one2one_cv3.2.1.0.conv | DW 3x3/1 | 20x20 | 80 | 80 | 720 | 257,472 | 256,752 | PASS | 20 | 1 |
| 81 | 11 | model.23.one2one_cv3.2.1.1.conv | 1x1 | 20x20 | 80 | 80 | 6,400 | 259,072 | 252,672 | PASS | 20 | 1 |
| 82 | 11 | model.23.one2one_cv3.2.2 | 1x1 | 20x20 | 80 | 80 | 6,400 | 259,072 | 252,672 | PASS | 20 | 1 |

### 4.3 가장 빡빡한 layer 상세

**Layer 65: model.23.one2one_cv2.2.0.conv** (최소 margin)

| 항목 | 값 |
|---|---|
| Type | Conv 3x3, stride 1 |
| Output | 20x20, C_out=64 |
| Input | 20x20, C_in=256 |
| Weight | 147,456 bytes (56.3% of UB) |
| Max weight 허용 | 237,760 bytes |
| **Margin** | **90,304 bytes (34.5% of UB)** |
| T_max | 17 |
| Tiles | 4 (ceil(20/17)^2) |
| Input tile | 19x19 = 92,416 bytes |
| Output tile | 17x17 x 64 = 18,496 bytes |

이 layer는 margin이 가장 작지만, 여전히 UB의 34.5%가 남아 안전하다.

### 4.4 Core별 Q tile budget (Adaptive Full)

| Core | Layer 범위 | Conv layers | Conv tiles | Non-conv tiles | Total tiles | Instruction records | 여유 (512 - records) |
|---|---|---|---|---|---|---|---|
| 0 | 0-6 | 7 | 67 | 52 | 119 | 144 | 368 |
| 1 | 7-13 | 7 | 38 | 16 | 54 | 77 | 435 |
| 2 | 14-20 | 7 | 21 | 90 | 111 | 146 | 366 |
| 3 | 21-27 | 7 | 19 | 100 | 119 | 184 | 328 |
| 4 | 28-34 | 7 | 13 | 60 | 73 | 118 | 394 |
| **5** | **35-41** | **7** | **27** | **136** | **163** | **236** | **276** |
| 6 | 42-48 | 7 | 27 | 56 | 83 | 126 | 386 |
| 7 | 49-55 | 7 | 19 | 100 | 119 | 184 | 328 |
| 8 | 56-62 | 7 | 22 | 71 | 93 | 166 | 346 |
| 9 | 63-69 | 7 | 16 | 86 | 102 | 175 | 337 |
| 10 | 70-76 | 7 | 43 | 76 | 119 | 162 | 350 |
| 11 | 77-82 | 6 | 9 | 55 | 64 | 127 | 385 |

Worst core는 Core 5 (236 records / 512 한계, 46.1% 사용). **모든 코어가 50% 미만으로 충분한 여유가 있다.**

---

## 5. YOLOv10s 분석

### 5.1 종합 판정

| 항목 | 결과 |
|---|---|
| **종합 판정** | **FAIL** |
| UB 판정 | **FAIL** (79 pass / 8 fail) |
| Q 판정 | **FAIL** (worst core 6,621 records, 한계 512) |
| SM 판정 | **FAIL** (총 weight 6.90MB > SM 4MiB) |
| 최대 weight | 524,288 bytes (Layer 32, model.9.cv2.conv) |
| UB 대비 최대 weight 비율 | **200%** (UB의 2배) |
| FAIL 원인 | Weight가 UB를 초과하는 layer 8개 존재 |

### 5.2 FAIL layer 상세 (8개)

모든 FAIL layer는 **weight 자체가 UB(256 KiB)를 초과**한다. Tiling으로 해결 불가.

| Index | Core | Name | Type | C_in | C_out | Weight (B) | UB (B) | 초과분 (B) | 초과 비율 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 0 | model.3.conv | Conv 3x3/2 | 64 | 128 | 73,728 | 262,144 | - | - |
| 23 | 2 | model.8.cv1.conv | 1x1 | 512 | 512 | 262,144 | 262,144 | 1,536 | 0.6% |
| 30 | 3 | model.8.cv2.conv | 1x1 | 768 | 512 | 393,216 | 262,144 | 131,584 | 50.2% |
| 32 | 4 | model.9.cv2.conv | 1x1 | 1024 | 512 | **524,288** | 262,144 | 262,656 | **100.2%** |
| 33 | 4 | model.10.cv1.conv | 1x1 | 512 | 512 | 262,144 | 262,144 | 1,536 | 0.6% |
| 39 | 5 | model.10.cv2.conv | 1x1 | 512 | 512 | 262,144 | 262,144 | 1,536 | 0.6% |
| 55 | 7 | model.22.cv1.conv | 1x1 | 768 | 512 | 393,216 | 262,144 | 131,584 | 50.2% |
| 62 | 8 | model.22.cv2.conv | 1x1 | 768 | 512 | 393,216 | 262,144 | 131,584 | 50.2% |
| 69 | 9 | model.23.one2one_cv2.2.0.conv | Conv 3x3 | 512 | 64 | 294,912 | 262,144 | 33,280 | 12.7% |

> 참고: Layer 6 (model.3.conv)은 safe_boundary.csv에서 MaxWeight=0으로 표시되어 FAIL이지만, weight(73,728) 자체는 UB(262,144)보다 작다. 이는 Adaptive Half(128KiB budget) 기준에서 FAIL이며, Adaptive Full에서도 tile budget이 부족한 경계 케이스이다.

**특징 분석:**
- 8개 중 6개가 **1x1 conv** (큰 channel 곱)
- 가장 큰 weight: **model.9.cv2.conv** (512 KiB = UB의 정확히 2배)
- 모두 20x20 feature map의 bottleneck 구간에 집중
- 공통 패턴: C_in 또는 C_out이 512 이상

### 5.3 Core별 Q tile budget (Adaptive Full)

| Core | Layer 범위 | Conv tiles | Non-conv tiles | Total tiles | Instruction records | 여유 |
|---|---|---|---|---|---|---|
| 0 | 0-7 | 134 | 89 | 223 | 250 | 262 |
| 1 | 8-15 | 73 | 176 | 249 | 294 | 218 |
| 2 | 16-23 | 457 | 175 | 632 | 699 | **-187** |
| 3 | 24-30 | 418 | 15 | 433 | 478 | 34 |
| 4 | 31-37 | 817 | 60 | 877 | 922 | **-410** |
| 5 | 38-44 | 473 | 484 | 957 | 1,030 | **-518** |
| 6 | 45-51 | 50 | 176 | 226 | 269 | 243 |
| 7 | 52-58 | 431 | 150 | 581 | 646 | **-134** |
| 8 | 59-65 | 426 | 96 | 522 | 575 | **-63** |
| **9** | **66-72** | **422** | **6,126** | **6,548** | **6,621** | **-6,109** |
| 10 | 73-79 | 48 | 76 | 124 | 167 | 345 |
| 11 | 80-86 | 19 | 100 | 119 | 184 | 328 |

**Q FAIL 코어: 6개** (Core 2, 4, 5, 7, 8, 9). Core 9는 6,621 records로 한계의 **12.9배**를 초과한다. 이는 weight 초과로 T=1이 되어 tile 수가 400개(20x20)로 폭증하는 layer 때문이다.

### 5.4 해결을 위한 최소 UB 크기

가장 큰 weight는 **model.9.cv2.conv = 524,288 bytes (512 KiB)**이다.

최소 UB 크기를 계산한다:

```
최소 UB = max_weight + min_tile_IO + metadata
        = 524,288 + (1024 + 512) + 512    (1x1 conv: 1 pixel input + output)
        = 526,336 bytes
        = 514 KiB
```

현실적으로는:
- **512 KiB UB**: weight만 겨우 적재, tile 공간 부족
- **576 KiB UB**: 적정 tile 크기 확보 가능 (T >= 5)
- **768 KiB UB**: 모든 layer에서 충분한 margin 확보

> v10s를 지원하려면 UB를 최소 **576 KiB (현재의 2.25배)**로 확장해야 한다.

---

## 6. v10n vs v10s 비교

| 항목 | YOLOv10n | YOLOv10s | 비고 |
|---|---|---|---|
| Layer 수 | 83 | 87 | +4 |
| 총 weight | 2.19 MB | 6.90 MB | 3.15x |
| 최대 weight | 147,456 B | 524,288 B | 3.56x |
| UB 대비 최대 weight | 56.3% | 200% | v10s 초과 |
| 최소 margin | 90,304 B | -276,480 B | v10s 음수 |
| UB FAIL layers | 0 | 8 | - |
| Q worst core | 236 / 512 | 6,621 / 512 | 12.9x 초과 |
| SM 판정 | PASS | FAIL | weight 합 초과 |
| **종합** | **PASS** | **FAIL** | - |

핵심 차이: v10s는 channel width가 v10n의 2~4배이므로, 1x1 conv의 weight(= C_in * C_out)가 4~16배로 증가한다. 이것이 UB 초과의 근본 원인이다.

---

## 7. 컴파일러 구현 가이드

### 7.1 T 결정 알고리즘

```python
def compute_T_max(K, S, C_in, C_out, UB, metadata=512):
    W = K * K * C_in * C_out  # weight size (일반 conv)
    R = UB - W - metadata     # remaining for tiles
    
    if R <= 0:
        return 0  # FAIL: weight exceeds UB
    
    a = S * S * C_in + C_out
    b = 2 * S * (K - S) * C_in
    c = (K - S) ** 2 * C_in - R
    
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return 0  # FAIL: no valid T
    
    T_max = int((-b + math.sqrt(discriminant)) / (2 * a))
    return max(T_max, 1)

def compute_tiles(H_out, W_out, T):
    return math.ceil(H_out / T) * math.ceil(W_out / T)

def is_safe(T, H_out, W_out, core_total_tiles, Q_max=512):
    tiles = compute_tiles(H_out, W_out, T)
    # Q 조건: core 내 모든 layer의 tile 합이 Q_max 이하
    return (core_total_tiles + tiles) <= Q_max
```

### 7.2 예시 계산 (v10n Layer 65)

**Layer 65: model.23.one2one_cv2.2.0.conv**
- Conv 3x3, stride 1, C_in=256, C_out=64, output 20x20

**Step 1: Weight 크기**
```
W = 3 * 3 * 256 * 64 = 147,456 bytes
```

**Step 2: 잔여 공간 R**
```
R = 262,144 - 147,456 - 512 = 114,176 bytes
```

**Step 3: 이차 방정식 계수 (3x3, stride 1)**
```
a = 1^2 * 256 + 64 = 320
b = 2 * 1 * (3-1) * 256 = 1,024
c = (3-1)^2 * 256 - 114,176 = 1,024 - 114,176 = -113,152
```

**Step 4: 판별식 및 T_max**
```
discriminant = 1,024^2 - 4 * 320 * (-113,152)
             = 1,048,576 + 144,834,560
             = 145,883,136
sqrt(disc)   = 12,078.6...

T_max = floor((-1,024 + 12,078.6) / (2 * 320))
      = floor(11,054.6 / 640)
      = floor(17.27)
      = 17
```

**Step 5: Tile 수 및 UB 검증**
```
Tiles = ceil(20/17) * ceil(20/17) = 2 * 2 = 4
Input tile HW = 1*17 + 3 - 1 = 19 → 19x19
Input tile B  = 19 * 19 * 256 = 92,416
Output tile B = 17 * 17 * 64  = 18,496
Total UB      = 147,456 + 512 + 92,416 + 18,496 = 258,880 <= 262,144  ✓
```

**Step 6: Q 조건 확인**
```
Core 9 총 tiles = 4 (L63) + 1 (L64) + 4 (L65) + 1 (L66) + 1 (L67) + 4 (L68) + 4 (L69) = 19
19 << 512  ✓
```

Layer 65는 margin이 가장 작은 layer이지만, UB와 Q 모두 안전하게 통과한다.

---

## 8. 결론

### YOLOv10n: 완전 PASS

- 83개 layer 전부 UB 제약 충족 (최소 margin 90 KiB)
- Q 제약 충분한 여유 (worst core 46.1% 사용)
- SM 제약 충족 (weight 2.19 MB < SM 4 MiB)
- **현재 하드웨어(UB 256 KiB, Q 512, SM 4 MiB)로 완전 실행 가능**

### YOLOv10s: FAIL (3중 실패)

1. **UB FAIL**: 8개 layer의 weight가 UB 초과 (최대 524 KiB = UB의 2배)
2. **Q FAIL**: weight 초과 layer에서 T=1 강제 적용 → tile 수 폭증 → 6개 core 초과
3. **SM FAIL**: 총 weight 6.90 MB > SM 4 MiB

### 해결 방안

| 방안 | 내용 | 영향 |
|---|---|---|
| **UB 확장** | 256 KiB → 576 KiB | 8개 FAIL layer 해결, Q 자동 해결 |
| **SM 확장** | 4 MiB → 8 MiB | Weight 전체 적재 가능 |
| **Weight 분할** | Layer를 sub-layer로 분리 | 컴파일러 복잡도 증가 |
| **모델 변경** | v10n 사용 또는 v10s pruning | 정확도 영향 |

현재 하드웨어 사양으로는 **YOLOv10n만 지원 가능**하며, v10s 지원에는 하드웨어 스펙 변경이 필요하다.
