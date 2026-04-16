# Infetron-V2 ISA 구조 한눈에 보기

> 원본: `docs/agent_command/hw_modeling/ISA.md` (권위 있는 정의).
> 이 문서는 ISA 전체를 한 페이지에서 조망하기 위한 요약본이다.
> 최종 반영 기준일: 2026-04-15.

---

## 1. 기본 제원

| 항목 | 값 |
|---|---|
| Record 크기 | **2048 bit (고정)** |
| cmd header | 3 bit |
| loop_count | 32 bit (공통) |
| opcode | 5 bit |
| operands (layer start) | 842 bit |
| operands (layer run) | 342 bit |
| operands (layer end) | 684 bit |
| DMA pack | 114 bit |

Record layout:
```
| cmd_header(3) | loop_count(32) | opcode(5) | operands(...) | padding -> 2048 |
```

---

## 2. Core 실행 필수 규칙

모든 core 의 명령 시퀀스는 다음 4-규칙을 만족해야 한다.

1. **첫 instruction** 의 cmd header 는 `stop`.
2. **두 번째 instruction** 의 cmd header 는 `loop_start`.
3. `loop_jump` → `loop_start` 위치(2번째)로 복귀.
4. `loop_end`  → `stop` 위치(1번째)로 복귀 = core halt.

결과적으로 한 core 안에는 **stop → loop_start → (layer_start/run/end)\* → loop_jump or loop_end** 형태가 정확히 한 번 등장한다. 모든 layer 는 이 하나의 loop 안에 담긴다.

---

## 3. cmd header 7종

| header | 의미 | operand |
|---|---|---|
| `stop` | core halt point | 없음 |
| `loop_start` | loop 시작 지점 | 없음 (loop_count 는 공통 필드) |
| `loop_jump` | loop_start 로 복귀 | 없음 |
| `layer_start` | 한 layer 의 공통 정보 세팅 (shape/scale/weight DMA 등) | opcode(5) + operands(842) |
| `layer_run` | tile 단위 실행 (input/output DMA) | opcode(5) + operands(342) |
| `layer_end` | layer 종료 + 후처리 DMA (pack ×6) | opcode(5) + operands(684) |
| `loop_end` | stop 위치로 복귀 | 없음 |

> **중요**: 한 `layer_start`~`layer_end` 구간 안에서 `layer_run` 의 opcode 는 layer_start 의 opcode 와 **다를 수 있다**. 예: Conv layer_start 안에서 Sigmoid/Add 등의 Elementwise run 이 섞일 수 있다.

---

## 4. Opcode 전체표 (5bit)

bit[4]=0 → 연산기 전용 / Concat·Split·Head. bit[4]=1 → Elementwise 계열.

| opcode | 이름 | 입력 | 출력 | layer_start 용량 | 특징 |
|---:|---|---:|---:|---:|---|
| 0 | reserved | - | - | - | - |
| **1** | Conv | 1 | 1 | 606 bit | groups=1 standard / groups=C_in depthwise (별도 op 없음). requant + activation 내장 |
| **2** | Gemm | 1 | 1 | 538 bit | i8 matmul + requant |
| **3** | Attention | 1 | 1 | 842 bit | Q·Kᵀ→softmax→·V 전체 내장. Wq/Wk/Wv/Wo + bias DMA |
| **4** | Concat | 2 | 1 | 30 bit  | 채널 축 concat. weight/scale 없음 |
| **5** | Split | 1 | 2 | 46 bit  | **순수 metadata**, DMA 없음. RefCountTable 등록만 |
| **6** | Head | n | n | 12 bit  | 모델-specific post-proc (yolov10n-detect 등). head_subop 으로 분기 |
| 7~15 | reserved | | | | |
| **16** | Add | 2 | 1 | 236 bit | EW, scale/zp 3세트 (in1/in2/out) |
| **17** | Mul | 2 | 1 | 236 bit | EW |
| **18** | Sigmoid | 1 | 1 | 236 bit | EW |
| **19** | MaxPool | 1 | 1 | 236 bit | aux=kernel |
| **20** | Resize | 1 | 1 | 236 bit | aux=scale factor |
| **21** | TopK | 1 | 2 | 236 bit | scores+indices |
| **22** | Transpose | 1 | 1 | 236 bit | aux=perm hint |
| **23** | Gather | 2 | 1 | 236 bit | data+indices |
| **24** | Copy | 1 | 1 | 236 bit | 예비 데이터 복사 |
| 25~27 | reserved | | | | |
| **28** | Softmax | 1 | 1 | 238 bit | aux 재해석: softmax_axis(2)+exp_mode(1). LUT 기반 exp |
| 29~31 | reserved | | | | |

---

## 5. Layer Start operand 구성 (842 bit 통일)

### 연산기 계열 (Conv/Gemm/Attention)

공통 빌딩 블록:
- **Weight/Bias DMA pack** 114 bit × N (Conv: W+B 2개, Attention: Wq/Wk/Wv/Wo+bias_q+bias_kv 6개)
- **Params** (opcode별 상세)
- **Scale array** 32 bit × K
- **Zero-point array** 32 bit × K
- **Quant control** 3 bit: `per_channel_enable(1) + accum_mode(2)`

Conv 전용:
- conv_params 115 bit: `C_in(16) + C_out(16) + H_in(16) + W_in(16) + Kh(4) + Kw(4) + stride_h(4) + stride_w(4) + pad_top(3) + pad_bottom(3) + pad_left(3) + pad_right(3) + dilation_h(3) + dilation_w(3) + groups(16) + tensor_format(1)`
- **activation type** 4 bit: 0=None, 1=ReLU, 2=SiLU, 3=GELU, 4=Sigmoid
- H_out/W_out 은 전달 안 함 (HW 계산)

Gemm params 51 bit: `M(16)+K(16)+N(16)+format(1)+accum_mode(2)`.

Attention params 27 bit: `C(16)+num_heads(4)+head_dim(4)+format(1)+accum_mode(2)`.

### Concat params (30 bit)
`C_total(16) + H(12) + format(1) + rsv(1)`. weight/scale 없음.

### Split params (46 bit)
`C_total(16) + H(12) + W(12) + num_outputs(2, 0=2-way) + format(1) + split_axis(2) + rsv(1)`.

> **컴파일러 계약**: split_axis 는 메모리에서 연속 범위를 자르는 축이어야 함. 예) NHWC 에서 C-split 금지 → Transpose(NHWC→NCHW)→Split→Transpose 로 풀이. 3-way 이상은 2-way Split 체인으로 합성.

### Head params (12 bit)
`head_subop(4) + num_inputs(4) + num_outputs(4)`. head_subop=0 → `yolov10n-detect` (inputs=6, outputs=1).

### Elementwise 공통 operand (236 bit)
- **shape params** 43 bit: `C(16) + H(12) + W(12) + aux(3)`
- format 1 bit
- scale array 32×3 = 96 bit (in1, in2, out)
- zero-point array 32×3 = 96 bit
- 총 43+1+96+96 = **236 bit**

`aux` 의미는 sub_opcode 별로:
| sub_opcode | aux 의미 |
|---|---|
| MaxPool(3) | kernel size (≤7) |
| Resize(4) | scale factor (2/4/...) |
| Transpose(6) | perm hint |
| Softmax(12) | softmax_axis(2) + exp_mode(1) |
| 기타 | reserved=0 |

---

## 6. Layer Run operand 구성 (342 bit 통일)

| opcode | 사용 bit | DMA pack 구성 |
|---|---:|---|
| Conv / Gemm / Attention | 228 | input(114) + output(114) |
| Concat | 342 | input1(114) + input2(114) + output(114) |
| Split | 208 | **DMA 없음**. out0 region(72) + out1 region(72) + input region(64) |
| Head | variable | input DMA pack ×N (compiler 재량으로 multi-RUN) |
| Add / Mul | 342 | in1(114) + in2(114) + out(114) |
| Sigmoid / MaxPool / Resize / Transpose / Copy / Softmax | 228 | input(114) + output(114) |
| TopK | 342 | input(114) + scores(114) + indices(114) |
| Gather | 342 | data(114) + indices(114) + output(114) |

> **Split region descriptor**: out_k = `buffer_idx(5) + addr(32) + size(27) + ref_count(8)` = 72 bit. input = 동일 필드에서 ref_count 제외 = 64 bit. DMA bus 를 쓰지 않고 RefCountTable 에만 등록.

> **Softmax tiling 규칙**: softmax 축은 절대 쪼개지 않음. 타일 경계는 non-softmax 축에서만 발생.

---

## 7. Layer End operand 구성 (684 bit 고정)

- 모든 opcode 공통: **DMA pack × 6 = 684 bit**.
- 미사용 pack 은 transfer_type 의 src/dst buffer 를 **dummy(31)** 로 채움.
- 용도: layer 간 sync, 후처리 DMA (weight write-back 등).

---

## 8. DMA pack (114 bit)

```
| transfer_type(10) | src_addr(32) | dst_addr(32) | data_size(32) | read_ref_count(8) |
```

- `transfer_type` = `src_buffer(5) | dst_buffer(5)`
- `read_ref_count` : dst 데이터가 읽혀야 하는 횟수. write-lock 기능. 각 참조시 1씩 감소.

### Buffer index (5 bit)
| 값 | 의미 |
|---:|---|
| 0~11 | Pcore 0~11 의 UB |
| 12~23 | Acore 0~11 의 UB |
| 24 | Shared Memory (SM) |
| 25 | Self (자기 core UB) |
| 26 | Paired Acore |
| 27 | 이전 인접 core |
| 28 | 다음 인접 core |
| 29~30 | reserved |
| **31** | **dummy** — DMA 수행하지 않음 (공간 채움용) |

---

## 9. Sync 모델

- 전담 sync 레지스터는 없고 **read_ref_count 기반 write-lock** 으로 레이스 해결.
- Producer 의 output DMA 가 `(dst_addr, data_size, ref_count)` 로 RefCountTable 에 등록 → 해당 구간은 잠김.
- Consumer 가 해당 구간을 읽을 때마다 count-- 수행. 0 이 되면 lock 해제.
- **Lock 중첩 허용**: 같거나 겹치는 영역에 대해 여러 entry 가 공존 가능. decrement 는 (addr, size) 정확 매칭. isLocked 는 범위-겹침 OR.

---

## 10. 컴파일러가 지켜야 하는 계약 요약

1. **Core structure**: stop + 단일 loop_start/loop_end 프레임. 모든 layer 는 하나의 loop 에 담김.
2. **Depthwise Conv** 는 별도 opcode 가 아님 → Conv 의 `groups=C_in` 사용.
3. **Split**
   - 연속 메모리 축만 허용. 불가능한 축은 Transpose 로 감싸기.
   - 3-way 이상 → 2-way chain 합성.
4. **Softmax 타일**: softmax 축은 절대 쪼개지 않음.
5. **Head bundler**: post-processing 을 하나의 Head opcode 로 묶거나, `--nms-mode=offload` 로 생략.
6. **layer_end 6-pack**: 쓰지 않는 pack 은 반드시 dummy(31) 로 채움.
7. **Attention 미사용 W**: Wo/bias 가 없어도 dummy pack 으로 6 슬롯 채움.

---

## 11. 빠른 참고: 이름 → opcode

| Family | 이름 | opcode |
|---|---|---:|
| Compute | Conv | 1 |
| Compute | Gemm | 2 |
| Compute | Attention | 3 |
| Structural | Concat | 4 |
| Structural | Split | 5 |
| Post-proc | Head | 6 |
| EW | Add | 16 |
| EW | Mul | 17 |
| EW | Sigmoid | 18 |
| EW | MaxPool | 19 |
| EW | Resize | 20 |
| EW | TopK | 21 |
| EW | Transpose | 22 |
| EW | Gather | 23 |
| EW | Copy | 24 |
| EW | Softmax | 28 |

---

## 12. 변경점 (2026-04 기준 주요 항목)

- **Head opcode=6** 신설 (yolov10n-detect 등 post-proc 번들).
- **Softmax opcode=28** 확정 (Elementwise family, LUT-exp).
- **Split opcode=5** 순수 metadata (DMA 없이 RefCountTable 등록만).
- **Depthwise Conv** 는 Conv 에 흡수 (groups 로 구분).
- **Attention** 단일 opcode 에 QKVO + softmax 전부 포함 (별도 Softmax 불필요).
