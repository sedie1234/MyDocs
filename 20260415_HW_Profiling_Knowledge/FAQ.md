# FAQ — HW Profiling 자주 묻는 질문

## ISA

### Q. 새 opcode 는 어디 reserved 영역에서 할당해야 하나?

**compute-op (bit[4]=0) 범위**: 현재 7~15 가 reserved. 복잡한 layer (weight/bias/scale 필요) 나 특수 카테고리 (Head 처럼 모델 특화) 는 여기에.

**Elementwise (bit[4]=1) 범위**: 현재 9~11, 13~15 가 reserved. weight 없는 일반 연산 (e.g., Sub/Div 등이었으면 여기). Softmax(12) 가 이미 이 범위.

번호 선택 시 인접 의미 연산을 묶는 것을 선호 (Softmax=12 처럼 sigmoid/activation 근처).

### Q. DMA pack 에 왜 stride 가 없나? NHWC C-axis split 을 어떻게 처리하나?

**ISA 는 linear DMA 만 지원** — `src_addr + size` 단순 복사. 비연속 축 조작이 필요한 연산 (NHWC C 방향 split 등) 은 **컴파일러가 Transpose 로 감싸서** 처리한다:

```
NHWC 입력 → Transpose(→NCHW) → Split(axis=C, 이제 연속) → Transpose(→NHWC) × N
```

Transpose 는 실제 데이터 재배치 연산이라 cost 가 있지만 ISA 를 단순하게 유지하는 trade-off.

### Q. LAYER_RUN 에 여러 DMA pack 이 들어갈 수 있나?

YES. 2-pack (input + output) 이 기본, 3-pack (input1 + input2 + output 또는 in + out + in2) 까지 허용. 최대 342 bit 안에서 결정.

### Q. Head opcode 로 NMS 를 번들링하면 HW 는 뭘 어떻게 실행하나?

- LAYER_START 에서 `head_subop` (4 bit) 를 읽어 어느 head 인지 식별
- LAYER_RUN 에서 입력 텐서 주소들을 dummy-dst DMA pack 으로 전달 (UB 로딩 없음)
- HW 는 head_subop 에 따라 **내부 고정 routine** 을 실행 (yolov10n-detect 의 경우 DFL + cls sigmoid + NMS suppression)
- 컴파일러는 routine 의 세부 구현을 **몰라도 됨** — HW 가 스스로 해결

즉 Head 는 "하드웨어 구현된 고정 후처리 파이프라인의 심볼릭 진입점" 이다.

## 아키텍처

### Q. Pcore / Acore 차이는?

Pcore = 범용. 모든 연산 (Conv, Gemm, Attention, Elementwise 등) 가능.
Acore = Depthwise conv 전용 + 직후 activation 결합. standard Conv/Gemm 안 됨.

Pcore 와 Acore 는 **쌍** (pair) 으로 존재하며, 같은 cluster 내 pair 간 DMA 가 가장 빠르다 (PC 경유, 2~4 clock/beat).

### Q. Core 간 데이터 동기화는 어떻게 하나?

**ref_count 기반**. producer 가 DMA 로 데이터를 쓸 때 "이 데이터는 N 번 읽힐 것" 을 RefCountTable 에 기록. consumer 가 DMA read 할 때마다 count 가 감소하고, 0 이 되면 영역 재사용 가능.

Consumer 가 producer 보다 먼저 read 하려 하면 ref_count 가 write 대기를 유발 (hazard 자동 처리).

**Core index 순서와 무관** — 컴파일러가 consumer 를 낮은 core index 에, producer 를 높은 core index 에 두어도 ref_count 가 동기화를 보장한다.

### Q. SM 4 MB 는 어떻게 쪼개지나?

3 개 논리 영역:
1. **metadata (readonly)**: weight + bias + 정적 데이터. 하드웨어 write 금지
2. **input / output 슬롯**: 런타임만 읽고 쓸 수 있음
3. **activation pool**: 중간 활성값. ref_count 로 관리되며 재활용 가능

컴파일 시 할당기가 **liveness 기반 interval scheduling** 으로 activation pool 을 배치.

### Q. UB double buffering 은 언제 켜지나?

Conv 의 tile 크기가 UB 의 절반보다 작으면 자동으로 A/B 영역으로 분할해 ping-pong. 큰 Conv 에선 double buffer 를 안 쓰는 게 더 좋을 수 있음 (tile 크기가 UB 전체를 써야 해서).

## 가상 컴파일러

### Q. 가상 컴파일러와 실제 컴파일러의 관계는?

| 관점 | 실제 컴파일러 | 가상 컴파일러 |
|---|---|---|
| 위치 | `pubH/independent/` | `hw-profiling/scripts/onnx_to_ihnn.py` |
| 목적 | 실제 NPU 에 올릴 바이너리 생산 | 시뮬레이터 검증용 실험 시나리오 생성 |
| 구현 | MLIR pipeline, C++ | Python hand-made |
| 세션 | 별도 | 이 세션 |

**가상 컴파일러의 결과물은 실제 NPU 에 올릴 용도가 아님**. 시뮬레이터가 "컴파일러가 이런 식으로 짜주면 어떻게 돌아갈까" 를 검증할 수 있도록 한다.

### Q. SM peak 을 줄이는 가장 큰 레버는?

경험상 순서 (2026-04 시점):
1. **Virtual concat** (Option A) — concat output 중복 저장 제거
2. **Split metadata** (Option R) — Split 레코드 제거, C2f concat unblock
3. **Shape-only canonicalization** — Reshape 류 제거
4. **할당기 개선** (Option C) — multi-variant FFD + local search
5. **ALAP reordering** (Option E) — 현재 모델에선 이득 없어 자동 revert
6. **LRS (live range splitting)** — 필요 시 fallback, YOLOv10n 에선 구조적으로 불가

### Q. Option B (branch chain) 는 왜 구현 안 했나?

YOLOv10n 의 분기 패턴 (Conv → multi-consumer) 가 **peak 영역과 시간상 분리** 되어 있어 구현해도 peak 가 안 줄어든다. 검증 결과 600 KB 의 잠재 절감이 있지만 peak op 와 교차하지 않음. 다른 모델에서 유효할 수 있으므로 **diagnostic 코드만 남겨둠**.

### Q. 왜 detection head (model.23) 전체가 host offload 대상인가?

model.23 의 ops 를 분류해보면:
- Softmax (DFL), ReduceMax, Sub, Div, Mod, TopK, Gather, GatherElements, Tile — 모두 **NMS / decode** 영역
- 실제 상용 NPU 는 대부분 NMS 를 host CPU 에서 돌림
- ISA 에 이들 op 을 전부 추가하는 것은 **실무적 가치 낮음**
- 대신 **Head opcode** 하나로 번들링하거나 **host offload** 선택

PSA attention 의 Softmax (model.10) 는 예외 — 이것만 on-device 필요.

## 시뮬레이터

### Q. 시뮬이 내는 cycle 수치는 얼마나 정확한가?

**opcode 별로 다르다**:

| opcode | 정확도 |
|---|---|
| Conv | 정식 HW 모델 (MAC array + requant + activation) |
| Gemm | 정식 HW 모델 |
| Attention | 정식 HW 모델 (QK + softmax + AV) |
| Concat | 0 cycle (순수 DMA) |
| Split | 0 cycle (metadata) |
| Elementwise (Add/Mul/Sigmoid 등) | 정식 HW 모델 (element 당 cycle 공식) |
| **Softmax** (new) | ⚠ 현재 elementwise 일반 공식으로 근사. 실제 exp LUT + reduce + divide cost 반영 안 됨 |
| **Head** (new) | ⚠ default 1 cycle. 실제 NMS 비용과 전혀 다름 |
| Transpose/TopK/Gather 등 | ⚠ elementwise 근사 |

**DMA cycle 은 opcode 무관하게 정확** (initial_latency + beats × clock_per_beat).

**컴파일러 시점 수치 (SM peak, queue, op 수) 는 항상 valid**.

### Q. 4 MB 초과했는데 시뮬이 어떻게 끝까지 돌아가나?

시뮬레이터는 **profiling 모드** 로 동작한다:
- SM out-of-bound DMA 를 ERROR 로 기록하지만 crash 하지 않음
- 해당 write 는 0-fill 로 처리 (실제로는 쓰레기 값)
- 실행은 계속 진행됨
- 최종 리포트에 ERROR 수와 SM peak 를 출력

실 NPU 에서는 이런 ERROR 는 fatal 일 수 있다. profiling 모드의 목적은 "peak 가 얼마나 초과했는지" 측정.

### Q. WAIT_DMA_DONE 이벤트가 많으면 문제인가?

**상황에 따라 다르다**:
- 많음 + wall time 증가 → core 간 hazard 가 critical path 에 영향 → 스케줄 재조정 필요
- 많음 + wall time 동일 → 병렬 실행이 다른 경로로 숨김 → 무해

시뮬의 WAIT_DMA_DONE 수가 컴파일러 정책 변경과 함께 얼마나 변하는지 보면 중요도 판단 가능.

## 실험 운영

### Q. 결과물 디렉토리를 어떻게 관리하나?

`docs/agent_results/virtual_compiler_experiment/artifacts/` 아래 variant 별로 분리:

```
artifacts/
├── hw_profiling_common/
├── hw_profiling_decoded/
├── hw_profiling_full/
└── (historical) hw_profiling_*/
```

각 디렉토리는:
- `README.md` — 시나리오 설명 + 컴파일러 수치
- `yolov10n_qdq.ihnn` — 바이너리
- `yolov10n_qdq.json` — 메타

sim 결과물 (`dump`, `log`, `timeline`) 은 별도 시뮬 세션에서 관리.

### Q. 재현성은 어떻게 보장되나?

- `onnx_to_ihnn.py` 는 random seed 를 0xA10CA70E 로 고정
- 입력 ONNX 경로가 변하지 않는 한 동일 binary 출력
- 컴파일 시점 수치 (SM peak 등) 는 완전 재현 가능
- 시뮬 cycle 은 HW 모델 업데이트 시 변동 가능 — 모델 version 을 함께 기록

### Q. 다른 모델 (YOLOv8n, EfficientDet 등) 로 실험하려면?

1. ONNX 준비 (static QDQ int8sym 포맷)
2. `onnx_to_ihnn.py --onnx <path>` 실행
3. 해당 모델에 맞는 cut point 결정:
   - YOLOv10n 의 `--variant={common,decoded,full}` 은 /model.23 경로에 하드코딩
   - 다른 모델은 cut-after op 이름을 새로 지정해야 함
4. 특정 모델의 head 를 Head opcode 로 번들링하려면 `--head-kind` 에 새 sub_opcode 이름 추가

### Q. 새 실험 시나리오 (예: "fusion 더 적극적") 를 추가하려면?

1. `onnx_to_ihnn.py` 에 새 플래그 추가 (`argparse`)
2. 해당 플래그가 어느 단계에 영향을 주는지 결정 (extract / layout / emit / reorder)
3. 로직 추가 후 기본 실험 (yolov10n_qdq) 으로 회귀 측정
4. 기존 수치와 비교해 이득 / 손해 판단
5. README / FAQ 업데이트

## 용어

### Q. "Coalesced physical peak" vs "SM peak" 차이는?

- **SM peak**: 할당기가 계산한 "가장 높은 end address" — 주소 공간에서의 최대값
- **Coalesced physical peak**: 각 op 시점에 살아있는 텐서들의 address 범위를 union 한 "실제 물리 footprint"

둘의 차이 = **fragmentation**. 이론상 coalesced peak 에 근접한 SM 사용이 가능하지만 할당기의 greedy 특성으로 몇 KB 단위 손해가 발생함.

### Q. "HW ops" 와 ONNX node 의 차이는?

- **ONNX node**: 원본 그래프의 op 단위
- **HW op**: 컴파일러가 hw_ops list 에 등록한 단위. QDQ 노드 병합, shape-only fold, SiLU fusion 등을 거친 결과

예: ONNX Conv → DequantizeLinear → Sigmoid → QuantizeLinear → DequantizeLinear → Mul → QuantizeLinear (6 nodes) → HW op 1 개 (Conv with activation=SILU)
