# 03. ISA Structure

원본: `source_hw_modeling/ISA.md`

## 기본 골격

- **Record size**: 2048 bit 고정
- **구조**: `cmd_header (3) + loop_count (32) + opcode (5) + operands + padding`
- **cmd_header** 종류:
  - `LAYER_START` — layer 단위 공통 정보 (weight DMA, shape, scale 등) 세팅
  - `LAYER_RUN` — tile 단위 연산 실행 (input/output DMA pack 포함)
  - `LAYER_END` — layer 종료 후처리 DMA
  - `LOOP_START` / `LOOP_JUMP` / `LOOP_END` — 반복 제어
  - `STOP` — queue 실행 중단 / PC=0 reset

## operands 크기

| cmd_header | operands 크기 |
|---|---:|
| LAYER_START | 842 bit (Attention 기준 최대) |
| LAYER_RUN | 342 bit (Concat 기준 최대) |
| LAYER_END | 684 bit (DMA pack ×6) |
| LOOP / STOP | loop_count 32 bit + 여분 |

각 opcode 는 자기 크기만큼만 사용하고 나머지는 `reserved (0)`.

## opcode 공간

### bit[4] = 0 — 연산기 전용 (compute op)

| opcode | 이름 | 설명 |
|---:|---|---|
| 0 | reserved | — |
| 1 | **Conv** | i8 conv + requant + activation |
| 2 | **Gemm** | matmul + requant |
| 3 | **Attention** | Q·K^T / softmax / A·V 통합 |
| 4 | **Concat** | 채널 방향 concat (DMA only) |
| 5 | **Split** | metadata-only, DMA 없음. RefCountTable 등록만 |
| 6 | **Head** | 모델-specific NMS/decode 번들 (head_subop 로 구분) |
| 7~15 | reserved | — |

### bit[4] = 1 — Elementwise (weight 없음)

sub_opcode (bit[3:0]) 로 구분:

| sub | opcode | 이름 | 용도 |
|---:|---:|---|---|
| 0 | 16 | Add | element-wise 덧셈 + requant |
| 1 | 17 | Mul | element-wise 곱 + requant |
| 2 | 18 | Sigmoid | standalone sigmoid (LUT) |
| 3 | 19 | MaxPool | max pooling |
| 4 | 20 | Resize | nearest-neighbor upsample |
| 5 | 21 | TopK | top-K 추출 |
| 6 | 22 | Transpose | 축 순열 |
| 7 | 23 | Gather | 인덱스 기반 추출 |
| 8 | 24 | Copy | 단순 복사 |
| 12 | 28 | **Softmax** | i8→i8, configurable axis, LUT exp |
| 9~11, 13~15 | | reserved | |

## DMA Pack

각 DMA 동작은 **114 bit** 의 pack 으로 record 안에 embed:

```
┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
│ transfer_type   │ src_addr │ dst_addr │ data_size│ref_count │
│ (10 bit)         │ (32 bit) │ (32 bit) │ (32 bit) │ (8 bit)  │
└─────────────────┴──────────┴──────────┴──────────┴──────────┘
   = src_buf (5) + dst_buf (5)
```

### buffer_index (5 bit)

| idx | 의미 |
|---:|---|
| 0~11 | pcore 0~11 의 UB |
| 12~23 | acore 0~11 의 UB |
| 24 | Shared Memory |
| 25 | 자기 자신 (self) |
| 26 | paired acore |
| 27 | 이전 인접 core |
| 28 | 다음 인접 core |
| 29~30 | reserved |
| 31 | dummy (DMA 안 함) |

### ref_count

- **이 DMA 로 쓴 데이터가 몇 번 읽힐 것인지** 를 예약
- RefCountTable 에 `(dst_buf, dst_addr, data_size, ref_count)` 엔트리로 기록
- consumer 의 DMA 가 이 영역을 read 하면 ref_count -= 1
- 0 이 되면 해당 entry 제거 → 영역 재사용 가능
- write lock 역할 동시 수행 (0 이 아니면 그 영역에 쓰지 못함)

## 주요 opcode 상세

### Conv (opcode 1)
- operands 605 bit 사용
- 구성: weight DMA pack + bias DMA pack + conv params + scale/zp + activation type + quant control
- conv params (114 bit): C_in, C_out, H_in, W_in, Kh, Kw, stride, pad, dilation, groups
- activation type (4 bit): 0=None, 1=ReLU, 2=SiLU, 3=GeLU, 4=Sigmoid
- **Weight stationary**: LAYER_START 에서 weight 를 한 번 올리고 LAYER_RUN 마다 input tile 바꿈

### Gemm (opcode 2)
- 538 bit 사용
- weight/bias DMA + M/K/N + scale/zp
- A[M,K] × B[K,N] = C[M,N], i32 accumulator 기본

### Concat (opcode 4)
- **30 bit** 사용 (params 만)
- 연산 없음, 순수 DMA 복사
- layer_run 에서 input1/input2/output 3 pack DMA (3 이상 입력은 multi-run)

### Split (opcode 5)
- **46 bit** 사용
- **DMA 없음** — metadata 연산. RefCountTable 에 출력 sub-region 등록만
- 연속 메모리 축 split 만 허용 (NHWC-C 등 비연속 축은 컴파일러가 Transpose 로 감쌈)
- 3 개 이상 출력은 2-output split 연쇄로 합성

### Head (opcode 6)
- **12 bit** 사용
- head_subop (4 bit) 로 모델별 post-processing 식별:
  - 0 = `yolov10n-detect` (DFL + NMS)
  - 1~15 = reserved (future: classification, segmentation)
- num_inputs (4 bit), num_outputs (4 bit)
- layer_run 은 입력 SM 주소 전달용, 실제 compute 는 HW/에뮬레이터 내부 routine 수행

### Softmax (sub_opcode 12, opcode 28)
- Elementwise 계열
- i8 → i8, configurable axis
- LUT 기반 exp 구현 (256-entry)
- **원자성**: softmax 축은 한 RUN 안에서 전체 처리. 다른 축은 타일로 쪼개 multi-RUN 허용

## Attention (opcode 3)
- 842 bit 전부 사용 (가장 큰 opcode)
- Q / K / V / O projection weight DMA + Q bias / KV bias DMA + attention params + scale/zp × 4 + quant control
- 내부적으로 Q·K^T, softmax, A·V 수행
- PSA (Partial Self Attention) 같은 작은 attention block 을 1 개 layer 로 처리

## 설계 원칙 정리

1. **Record size 고정 2048 bit** — instruction fetch 단순화
2. **cmd_header 로 record 의미 구분** — LAYER_START/RUN/END, LOOP, STOP
3. **LAYER_START 는 공통 정보 (shape, weight, scale)** — tile 마다 반복되는 LAYER_RUN 에서 추가 설정 불필요
4. **LAYER_RUN 의 operands 는 tile 단위 변동분 (input/output DMA)** — 대부분 2~3 pack
5. **DMA pack linear** — strided gather 없음, 비연속 축 조작은 Transpose 로 풀어냄
6. **ref_count 기반 sync** — core index 순서에 의존하지 않음, 데이터 흐름 기준
7. **i8 everywhere** — 모든 data tensor 1 byte/element, scale/zp 는 record 고정 필드

## 새 opcode 추가 시 체크리스트

1. **bit[4]** 결정: compute (0) 인가 elementwise (1) 인가
2. **opcode 번호** 할당: reserved 범위에서 선택
3. **LAYER_START operand** 설계: shape, param, scale/zp 등 필요한 필드 (842 bit 안에서)
4. **LAYER_RUN operand** 설계: DMA pack 수 결정 (보통 2~3 pack)
5. **인코더 구현**: `hw-profiling/scripts/onnx_to_ihnn.py` 의 `make_*_layer_start` 함수 추가
6. **시뮬 decode 추가**: `src/simulator/` 의 opcode 핸들러 + compute cycle 모델
7. **ISA.md 업데이트**: sub_opcode 표, layer_start/run 상세, opcode 전체표
8. **테스트**: 새 opcode 를 emit 하는 바이너리 생성 + 시뮬 실행

## 알려진 ISA 의사결정 기록

- **Split 은 metadata-only** (2026-04-14) — 비연속 축은 Transpose 로 감쌈
- **Softmax 는 configurable axis** (2026-04-15) — UB 를 초과하는 경우 non-softmax 축을 multi-RUN 으로 쪼갬
- **Head 는 모델별 고정 routine** (2026-04-15) — 컴파일러는 head_subop 만 전달, HW/런타임이 구현
- **NMS/decode 는 host offload** 가 기본 (2026-04-15) — on-chip 은 Head opcode 경유가 유일한 수용 경로
- **Sub/Div/Mod/Reduce/Tile/GatherElements 는 ISA 추가 안 함** (2026-04-15) — 전부 NMS/decode 영역이라 host offload 또는 Head 번들로 처리
