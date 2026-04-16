# Infetron-V2 AI Compiler — 팀 협업 지식 문서

> **대상**: Infetron-V2 NPU와 연동하여 개발을 진행하는 HW / Runtime / Software 팀  
> **작성일**: 2026-04-02  
> **현재 상태**: Independent 파이프라인이 메인. Step 0~5 (양자화 IR 생성) 확정 완료. Step 6+ (Lowering) 실험 중.  
> ⚠️ HW 구조 및 ISA는 아직 변경 가능성이 있으므로 HW 관련 내용도 바뀔 수 있다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [하드웨어 아키텍처](#2-하드웨어-아키텍처)
3. [컴파일러 구조 — 파이프라인 개요](#3-컴파일러-구조--파이프라인-개요)
4. [모델 요구사항](#4-모델-요구사항)
5. [Independent 파이프라인 — Step 0~5 (확정)](#5-independent-파이프라인--step-05-확정)
6. [Step 6+ Lowering 파이프라인 (실험 중)](#6-step-6-lowering-파이프라인-실험-중)
7. [IREE Plugin 파이프라인 (초기 설계, 대기 중)](#7-iree-plugin-파이프라인-초기-설계-대기-중)
8. [컴파일러가 수행하는 일 — Pipeline · Pass · Pattern](#8-컴파일러가-수행하는-일--pipeline--pass--pattern)
9. [모델 컴파일 → 런타임 실행 전체 구상](#9-모델-컴파일--런타임-실행-전체-구상)
10. [환경 설정](#10-환경-설정)
11. [예제 스크립트](#11-예제-스크립트)
12. [장비 사양 참고](#12-장비-사양-참고)
13. [Git 저장소 정리](#13-git-저장소-정리)

---

## 1. 프로젝트 개요

Infetron-V2는 에지 환경을 위한 아날로그 NPU SoC다. 이 컴파일러는 표준 딥러닝 모델(ONNX)을 받아 Infetron-V2 NPU가 직접 실행할 수 있는 바이너리로 변환한다.

### 파이프라인 개발 방향

처음에는 IREE를 기반으로 한 플러그인 방식(IREE Plugin)으로 설계되었으나, 구조 변경으로 인해 현재는 **StableHLO 기반의 Independent 파이프라인이 메인**이다.

- **Independent 파이프라인**: IREE 의존성 없이 독립적으로 동작. 현재 메인 개발 방향.
- **IREE Plugin**: 초기 설계, 현재 대기 중. INF_CAP Layer까지 구현 완료 상태.

향후 HW 설계가 확정되면 Independent 파이프라인을 기반으로 본격 개발을 시작한다. 현재 코드는 실험용이지만 전체 뼈대(파이프라인 단계, Pass 구조)는 그대로 유지될 예정.

### 현재 개발 상태 (2026-04-02 기준)

| 구성 요소 | 상태 |
|---|---|
| Independent Step 0~5 (양자화 IR 생성) | ✅ 확정 완료 (main, step5-quantize-region 브랜치) |
| Independent Step 6+ (Lowering, 코어 분산) | 🔬 실험 단계 (experiment/fused-32x32-conv) |
| IREE Plugin INF_CAP Layer | ✅ 완료 (ISA 대기) |
| IREE Plugin INF_ISA / INF_EXP Layer | ⏸️ ISA 미확정 대기 |
| 최종 바이너리 생성 | ⏸️ ISA 확정 후 |

### 검증 타겟 모델

| 모델 | 용도 | 상태 |
|---|---|---|
| YOLOv10n | 비전 Detection, 데모 타겟 | Step 0~5 검증 완료 |

---

## 2. 하드웨어 아키텍처

> 참고: `ArchitectureSpecification.pdf` (2026.01.06 v1.3)  
> ⚠️ 초기 설계 문서이며, 변경 가능성이 있다.

### 2.1 NPU 구성

```
SoC
 └─ OIC (On-chip Interconnect)
      ├─ Shared Memory (SM)
      └─ NPU Cluster × 3
           └─ NPU Core × 4  →  총 12개 NPU Core
                ├─ aCore  (scalar / address 처리)
                └─ pCore  (벡터 / 행렬 연산 처리)
```

- **NPU Core 총수**: 3 Clusters × 4 Cores = **12개**
- **각 Core**: aCore + pCore 쌍으로 구성
- **NC Ring 토폴로지**: aCore 12개 링 + pCore 12개 링 (2개의 독립 링)

### 2.2 메모리 구조

| 메모리 | 위치 | 역할 | 비고 |
|---|---|---|---|
| **Shared Memory (SM)** | NPU Core 내부 | Weight 상주. aCore/pCore 공유 접근 | 순환 아님 |
| **Unified Buffer (UB)** | Core 내부 | 중간 데이터 저장 | 순환(CircularQ) 방식 |
| **DMA / UBC** | SM ↔ UB 사이 | 데이터 이동 | Prefetch DMA |

- SM에 weight가 상주한다 (순환 버퍼 아님)
- UB는 CircularQ 방식으로 중간 데이터를 순환 재사용

### 2.3 Core 내부 연산 파이프라인 (Micro-pipeline)

```
WF → DF → MM → AT → VP → WB
```

| 단계 | 이름 | 역할 |
|---|---|---|
| WF | Weight Fetch | 가중치 로드 |
| DF | Data Fetch | 입력 데이터 로드 |
| MM | Matrix Multiply | 행렬 연산 (아날로그 가속) |
| AT | Activation | 활성화 함수 적용 |
| VP | Vector Process | 벡터 후처리 |
| WB | Write Back | 결과 저장 |

### 2.4 컴파일러와의 관계

컴파일러는 layer 단위로 다음을 생성한다:
- **VLIW 명령어 스트림**: 각 Core가 실행할 명령어 묶음
- **DMA Descriptor**: SM ↔ UB 간 데이터 이동 계획
- **Core 할당 정보**: 어느 연산을 어느 Core에서 실행할지

---

## 3. 컴파일러 구조 — 파이프라인 개요

### 3.1 전체 흐름

```
fp32 ONNX 모델
      │
      ▼ [선택] onnx_prepare.py
QDQ int8 ONNX
      │
      ▼  model_to_stablehlo.py
StableHLO MLIR + 가중치 manifest
      │
      ▼  ind-compile / ind-opt  (Independent 파이프라인 — 메인)
      │
      ├─ Step 1~5:  quantized StableHLO  ← 확정
      │
      └─ Step 6+:   affine/memref IR    ← 실험 중
                    (12-core 분산)
```

### 3.2 IR 추상화 계층 (Independent 파이프라인)

컴파일러는 점진적 로워링(Progressive Lowering) 방식으로 IR을 단계적으로 낮춰간다.

```
StableHLO (fp32 + fake QDQ)    ← Python Importer 출력
       │ Step 1~5
       ▼
quantized StableHLO (i8/i32)   ← 확정
       │ Step 6+
       ▼
fused dialect IR                ← 실험 (fused.conv_silu, fused.pad 등)
       │
       ▼
affine.for + memref              ← 실험 (bufferize 완료)
       │
       ▼
12-core affine + memref          ← 실험 (upstream 2개 pass로 달성)
```

---

## 4. 모델 요구사항

### 4.1 지원 모델 형식

| 형식 | 지원 여부 | 비고 |
|---|---|---|
| **fp32 ONNX** | ✅ 지원 | onnx_prepare.py가 QDQ 변환 수행 |
| **int8 QDQ ONNX** | ✅ 지원 (권장) | 이미 양자화된 모델, Step 1 스킵 |
| PyTorch 모델 | 부분 지원 | ONNX export 후 사용 권장 |
| TFLite / TensorFlow | ❌ 미지원 | |

### 4.2 ONNX 모델 요구사항

#### 필수 조건

| 조건 | 설명 |
|---|---|
| **정적 입력 shape** | 동적 shape 미지원. 모든 차원이 컴파일 시점에 확정되어야 함. 예: `[1, 3, 640, 640]` |
| **ONNX opset** | **opset 17 고정** (개발 단계에서는 17로 진행. 컴파일러 완성 후 상위 버전 확장 가능) |
| **데이터 타입** | 가중치: int8 (QDQ 모델 기준). 활성화: int8/float32 혼용 |
| **QDQ 형식** | QDQ ONNX: QuantizeLinear / DequantizeLinear 노드 필수 |
| **Initializer에 scale/zero_point 포함** | Conv의 weight scale, bias scale이 initializer에 명시되어야 함 |

#### 양자화 방식 요구사항

| 조건 | 상세 |
|---|---|
| **양자화 방식** | **int8 대칭 양자화 (Symmetric int8) 고정** (개발 단계. 완성 후 비대칭 확장 가능) |
| **Granularity** | **per-tensor 고정** (per-channel 미지원. 첫 컴파일러 완성 후 agenda로 등록 예정) |
| **zero_point** | 대칭 양자화이므로 zero_point = 0 (고정) |
| **QDQ 포맷** | ONNX QDQ format (`QuantFormat.QDQ`) |

> **per-channel 미지원 이유**: InlineConstantsPass의 scale 처리 및 IdentityElimPass의 ZP=0 가정이 per-tensor 기준으로 설계됨.  
> **향후 agenda**: per-channel quantization은 첫 컴파일러 개발 완료 후 별도 agenda로 등록하여 확장할 예정.

#### ONNX Op 변환 지원 범위

onnx_prepare.py에서 onnx2torch 변환 시 커스텀 converter가 필요한 op 목록:

| ONNX Op | 지원 조건 | 미지원 경우 처리 |
|---|---|---|
| **QuantizeLinear** | opset 17 | - |
| **DequantizeLinear** | opset 17 | - |
| **Split** | static split sizes (initializer에 포함) | 동적 sizes → 변환 실패 |
| **Reshape** | static target shape (initializer에 포함) | 동적 shape → 변환 실패 |
| **Resize** | static scale factor (initializer에 포함) | 동적 scale → 변환 실패 |
| **Slice** | static starts/ends/axes/steps (initializer에 포함) | 동적 인덱스 → 변환 실패 |
| **TopK** | static k (initializer에 포함) | 동적 k → 변환 실패 |
| **Tile** | static repeats (initializer에 포함) | 동적 repeats → 변환 실패 |
| **Unsqueeze** | static axes (initializer에 포함) | 동적 axes → 변환 실패 |
| **Conv** (functional) | weight/bias가 initializer에 없으면 functional 방식 사용 | - |

**핵심 제약**: 위 op들의 shape/size/scale 파라미터는 모두 **컴파일 시점에 정적으로 확정**되어야 한다. 런타임에 결정되는 동적 값은 지원하지 않는다.

### 4.3 Conv 관련 요구사항

#### 지원하는 Conv 종류

| Conv 종류 | MLIR 표현 | 입력 Layout | 커널 Layout | 비고 |
|---|---|---|---|---|
| **Standard Conv 2D** | `linalg.conv_2d_nchw_fchw` | NCHW | FCHW | Stride, Dilation 지원 |
| **1×1 Conv 2D** | `linalg.generic` (4D) | NCHW | FCHW | OpFusionPass에서 별도 처리 |
| **Depthwise Conv 2D** | `linalg.depthwise_conv_2d_nhwc_hwc` | NHWC | HWC | |
| **Batch MatMul** | `linalg.batch_matmul` | - | - | transpose indexing_map 분석 |

#### Layout 제약

- **NCHW**: 현재 컴파일러가 검증한 기본 layout
- **NHWC**: depthwise conv에서 사용
- ONNX 모델이 다른 layout을 사용하면 IREE import 단계에서 변환되거나 처리 불가

#### Padding 처리

IREE는 conv의 padding을 `tensor.insert_slice`로 표현한다. VailPadPass가 이 구조를 분석하여 padding 값을 conv attribute로 이전한다.

### 4.4 Op별 i8 양자화 지원 여부

컴파일러가 i8로 변환 가능한 op 분류 (Step 5 기준):

#### compute_ops — i8 변환 대상

| Op (StableHLO) | 입력 타입 | 출력 타입 | 설명 |
|---|---|---|---|
| `stablehlo.convolution` | i8 | i32 | i8×i8 → i32 convolution |
| `stablehlo.add` | i8 | i8 | element-wise 덧셈 |
| `stablehlo.logistic` | i8 | i8 | sigmoid (NPU LUT 활용 예정) |
| `stablehlo.multiply` | i8 | i8 | element-wise 곱셈 |
| `stablehlo.reduce_window` | i8 | i8 | max pooling 등 |

#### transparent_ops — i8 타입 자동 전파

인접 op이 i8이면 자신도 i8을 사용하도록 따라간다.

| Op | 설명 |
|---|---|
| `stablehlo.broadcast_in_dim` | tensor broadcast |
| `stablehlo.reshape` | shape 변환 |
| `stablehlo.transpose` | 축 순서 변환 |
| `stablehlo.bitcast_convert` | type 재해석 |
| `stablehlo.slice` | tensor 슬라이싱 |
| `stablehlo.convert` | type 변환 |

#### never_allowed — f32 필수 (i8 변환 불가)

| Op | 이유 |
|---|---|
| `stablehlo.exponential`, `stablehlo.log` | 초월함수 |
| `stablehlo.sine`, `stablehlo.cosine`, `stablehlo.tanh` | 삼각/쌍곡선 |
| `stablehlo.sqrt`, `stablehlo.rsqrt` | 제곱근 |
| `stablehlo.divide`, `stablehlo.power` | 비선형 산술 |

#### 경계 처리 대상 (일부 f32 유지)

| Op | 이유 | 상태 |
|---|---|---|
| `stablehlo.concatenate` | operand scale 불일치 가능 | QuantCleanup에서 별도 처리 |
| `stablehlo.dot_general` | attention에서 양쪽 activation | 경우에 따라 i8 가능 |
| `stablehlo.pad` | pooling padding | transparent 이동 가능 |
| `stablehlo.gather` | interpolation index | f32 유지 |
| `stablehlo.reduce` | softmax body 내부 | f32 유지 |

### 4.5 YOLOv10n 기준 실제 Op 통계

Step 5b 완료 후 양자화 결과:

| Op | 전체 수 | i8 변환 | f32 잔존 | 잔존 이유 |
|---|---|---|---|---|
| `convolution` | 83 | **83 (100%)** | 0 | |
| `logistic` | 70 | **70 (100%)** | 0 | |
| `add` | 99 | **93 (94%)** | 6 | reduce body 4개 + 기타 2개 |
| `multiply` | 76 | **73 (96%)** | 3 | |
| DequantizeOp | 479 → | → | **17 잔존** | 96% 제거 |

### 4.6 칼리브레이션 데이터 요구사항

onnx_prepare.py가 양자화할 때 사용하는 calibration 데이터:

| 항목 | 기본값 | 설명 |
|---|---|---|
| **샘플 수** | 20회 | `--n-calibration` 옵션으로 조정 가능 |
| **입력 shape** | `[1, 3, 640, 640]` | YOLOv10n 기준. 다른 모델은 수정 필요 |
| **데이터 타입** | float32 | random 값 사용 (정확도보다 scale 범위 탐색 목적) |
| **생성 방식** | `np.random.rand(...)` | 실제 이미지 데이터 없어도 동작 |

> **주의**: calibration 데이터의 shape이 모델 입력 shape과 일치해야 한다. 다른 해상도 모델 사용 시 스크립트 내 shape 변경 필요.

---

## 5. Independent 파이프라인 — Step 0~5 (확정)

### 5.1 전체 단계 흐름

```
입력: fp32 ONNX 또는 QDQ int8 ONNX
       │
       ▼
[Step 0]  Python Importer
          QDQ ONNX → onnx2torch → torch_xla → StableHLO MLIR + 가중치 manifest
       │
       ▼
[Step 1]  --inline-constants
          manifest.json을 참조하여 MLIR 함수 인자를 stablehlo.constant로 교체
          미사용 인자 제거
       │
       ▼
[Step 2]  --canonicalize  --cse
          기본 fold/simplify 패턴 적용 + 공통 부분식 제거
          (MLIR upstream pass, 코드 구현 없음)
       │
       ▼
[Step 3]  --identity-elim  →  --canonicalize  --cse
          zero_point=0인 항등 연산 제거:
          subtract(x, 0) → x
          add(x, 0) → x
          multiply(x, 1) → x
          제거 후 Canonicalize/CSE 재적용 (새로운 fold 기회 발생)
       │
       ▼
[Step 4]  --math-to-qdq
          수학적으로 전개된 Q/DQ 표현을 명시적 op으로 변환:
          round(x / scale) + zp  →  uniform_quantize(x, scale, zp)
          (x - zp) * scale       →  uniform_dequantize(x, scale, zp)
       │
       ▼
[Step 5a]  --quantize-region  →  --canonicalize  --cse
          allowed op (conv 등)의 입력을 역추적:
          DQ → compute_op  →  compute_op(i8 입력)으로 변경
          중간 DQ가 불필요해지면 Canonicalize/CSE로 자동 제거
       │
       ▼
[Step 5b]  --quant-cleanup
          noop quantize (scale=1, zp=0) 제거
          f32 concat의 모든 입력이 DQ이면 → quant concat으로 fusion

출력: quantized StableHLO MLIR
```

### 5.2 Step 0: Python Importer 상세

#### 입력 경로 A (권장): fp32 ONNX가 있는 경우

```
fp32 ONNX
  │ onnx_prepare.py
  ├─ [Step A1] QDQ int8 symmetric 양자화 → quantized_int8sym.onnx
  ├─ [Step A2] Conv 노드별 metadata 추출 → conv_metadata.json + onnx_metadata/*.npy
  └─ [Step A3] fp32 ONNX → onnx2torch → PyTorch model
                (QDQ ONNX 아닌 fp32 기준 변환 → 안정적)

  → model_to_stablehlo.py
  ├─ conv_metadata.json 로드
  ├─ PyTorch model에 per-conv Fake QDQ wrapping 적용
  ├─ torch.export → torch_xla → StableHLO MLIR
  └─ input_locations 기반 가중치 manifest 생성
```

#### 입력 경로 B: QDQ ONNX만 있는 경우

```
QDQ ONNX
  │ onnx_prepare.py (--qdq-onnx 옵션)
  ├─ [Step A1 스킵]
  ├─ [Step A2] metadata 추출
  └─ [Step A3] QDQ ONNX → onnx2torch (custom DQ converter로 float 복원)
               → 경로 A보다 불안정할 수 있음
```

#### onnx_prepare.py 처리 내용

**Step 1 — QDQ 양자화 (fp32 ONNX → int8 QDQ ONNX):**
- `onnxruntime.quantization.quantize_static` 사용
- 설정: `QuantFormat.QDQ`, `QInt8`, `ActivationSymmetric=True`, `WeightSymmetric=True`, `per_channel=False`
- 20회 random calibration (shape: `[1, 3, 640, 640]`, 모델에 맞게 수정 필요)

**Step 2 — Conv metadata 추출:**
- ONNX 그래프를 순회하며 Conv 노드별로 다음 추출:
  - `input_scale`: Conv 입력 전 DequantizeLinear의 scale
  - `weight`: int8 weight 값 (`onnx_metadata/*.npy`)
  - `weight_scale`: weight DequantizeLinear의 scale
  - `bias`, `bias_scale`: bias 데이터 및 scale (있으면)
  - `output_scale`: Conv 출력 후 QuantizeLinear의 scale
- **ONNX node name 기반 파일 네이밍** (shape 기반 매칭 금지 → 중복 shape 문제 방지)

**Step 3 — ONNX → PyTorch 변환 (커스텀 converter 10개):**

| Converter | 처리 방식 |
|---|---|
| `QuantizeLinear` | `round(x/scale) + zp → int` (clamp [-128, 127]) |
| `DequantizeLinear` | `(x - zp) * scale → float` |
| `Split` | initializer에서 static split_sizes 추출 |
| `Reshape` | initializer에서 static target_shape 추출 |
| `Resize` | initializer에서 static scale_factor 추출 |
| `Slice` | initializer에서 static starts/ends/axes/steps 추출 |
| `TopK` | initializer에서 static k 추출 |
| `Tile` | initializer에서 static repeats 추출 |
| `Unsqueeze` | initializer에서 static axes 추출 |
| `Conv (functional)` | weight/bias가 initializer에 없으면 functional conv2d 사용 |

#### model_to_stablehlo.py 처리 내용

**Fake QDQ wrapping:**
- 각 Conv layer에 `PerConvFakeQDQ` wrapper 적용
- input: `fake_qdq(x, input_scale)` — float round-trip으로 양자화 효과 시뮬레이션
- weight: `fake_qdq(weight, weight_scale)`
- output: `fake_qdq(out, output_scale)`
- bias: dummy 유지 (C++ InlineConstantsPass에서 실제 값으로 교체)

**input_locations 기반 manifest 생성:**

torch_xla가 생성한 StableHLO의 `%arg_i`와 PyTorch parameter name이 `input_locations[i]`를 통해 1:1 매핑된다. 이 정보를 이용해 manifest를 생성하므로:
- bias 포함 가중치 82/82 ONNX 대비 오차 < 1e-5 달성
- FIFO 방식 대비 완벽한 매핑 보장

**출력:**
- `stablehlo.mlir`: Step 1 입력 (함수 인자가 가중치인 형태)
- `weights/manifest.json`: `{ "arg_idx": { "name", "file", "shape", "dtype" } }`
- `weights/*.npy`: 가중치 배열 파일들

### 5.3 Step 1~5: C++ Pass 상세

#### InlineConstantsPass (Step 1)

- manifest.json을 파싱하여 `arg_idx → .npy 파일` 매핑 로드
- MLIR 함수의 `%arg_i`를 `.npy` 값을 담은 `stablehlo.constant`로 교체
- 교체 후 더 이상 사용되지 않는 함수 인자 제거
- 가중치 정확도: weight 83/83, bias 82/82, ONNX 대비 오차 < 1e-5

#### IdentityElimPass (Step 3)

- IR 전체를 walk하며 ZP=0인 항등 연산 탐색
- `subtract(x, const(0))`, `add(x, const(0))`, `multiply(x, const(1))` → `x`로 교체
- ZP=0은 대칭 양자화에서 항상 성립하므로 대부분의 ZP 관련 연산 제거 가능

#### MathToQDQPass (Step 4)

StableHLO에서 수학적으로 표현된 Q/DQ 패턴을 인식하여 명시적 op으로 변환:

```
# Quantize 패턴 (float → int8)
round(x / scale) + zp  →  stablehlo.uniform_quantize(x)

# Dequantize 패턴 (int8 → float)
(cast(x) - zp) * scale  →  stablehlo.uniform_dequantize(x)
```

변환 후 `quant.uniform<i8:f32>` 타입이 명시적으로 IR에 표현된다.

#### QuantizeRegionPass (Step 5a) — backward 방식

일반 forward 패턴 매칭과 다른 **역방향 추적** 방식:

```
알고리즘:
1. compute_ops 목록(convolution, add 등)을 찾는다
2. 해당 op의 입력을 역방향으로 추적:
   input → uniform_dequantize → ...
3. DQ의 입력이 i8이면 compute_op을 quant type 버전으로 rewire
4. 중간 DQ가 사용처를 잃으면 Canonicalize/CSE가 자동 제거
5. transparent_ops(reshape, transpose 등)도 quant type을 전파받아 i8로 변환
```

**backward 방식이 필요한 이유:**
conv 직전의 DQ를 제거하면 그 앞의 concat, reshape 등 transparent op들도 연쇄적으로 i8 타입을 가질 수 있다. 이 연쇄를 forward로 추적하기 어려워 backward 방식을 사용한다.

**i8_allowed_ops.json으로 제어:**
- `compute_ops`: 직접 i8로 변환 (convolution, add, logistic, multiply, reduce_window)
- `transparent_ops`: quant type 전파만 함 (reshape, transpose, broadcast_in_dim, slice 등)
- 이 파일을 수정하면 코드 변경 없이 변환 범위 조정 가능

#### QuantCleanupPass (Step 5b)

- **noop quantize 제거**: scale=1, zp=0인 quantize op (의미 없는 재양자화) 제거
- **concat fusion**: f32 concat의 모든 입력이 `uniform_dequantize` 출력이면 → quant type concat으로 fusion하고 DQ 제거

### 5.4 ind-compile 옵션 정리

```bash
ind-compile input.mlir [옵션] -o output.mlir

주요 옵션:
  --manifest=<path>          가중치 manifest.json 경로 (Step 1 필수)
  --compile-to=<stage>       중간 단계에서 멈추기:
                               InlineConstants  → Step 1 완료 후
                               IdentityElim     → Step 3 완료 후
                               QDQFusion        → Step 4 완료 후
  --dump-after-each          각 Pass 후 MLIR 파일 저장
  --dump-dir=<dir>           dump 파일 저장 경로
  --print-stats              Pass별 op 통계 출력
  --mlir-elide-elementsattrs-if-larger=32
                             가중치 상수 출력 생략 (대형 모델 가독성)
```

---

## 6. Step 6+ Lowering 파이프라인 (실험 중)

Step 5 이후 quantized StableHLO를 NPU가 실행 가능한 memref 기반 affine IR로 변환하는 단계. ISA 확정 후 본격 개발 예정이며, 현재는 실험 브랜치(`experiment/fused-32x32-conv`)에서 진행 중.

### 6.1 실험 파이프라인 흐름

```
quantized StableHLO (Step 5 출력)
      │
      ▼ --conv-tile-32         L1: 32×32 spatial tiling
      ▼ --depthwise-split      L2: depthwise → per-channel 분리
      ▼ --distribute-elementwise  L3: elementwise per-tile 분배
      ▼ --fuse-conv-silu       L4: conv+bias+SiLU → fused.conv_silu op
      ▼ --strip-quant-types    L5: quant.uniform → i8/i32 raw type
      ▼ --tile-to-scf          L6: flat fused ops + concat → scf.forall
      ▼ --forall-to-affine     L6a: scf.forall → affine.for (iter_args)
      ▼ --absorb-pad           L6b: stablehlo.pad → fused.pad (attr 흡수)
      ▼ --affine-bufferize-fused  L7: tensor → memref (dest-passing style)
      ▼ --affine-loop-coalescing  L8: 2D nested → 1D coalesced (upstream)
      ▼ --affine-loop-tile="tile-size=12"  L8: 12-core strip-mining (upstream)

출력: affine.for + memref IR, 12-core 분산 완료
```

### 6.2 12-core 분산 결과 (실험 기준)

| 함수 | 원래 구조 | coalesce 후 | tile-12 후 |
|---|---|---|---|
| std_conv_silu | 5×5 nested | 25 (1D) | `for core=0 to 25 step 12 { for local ... }` |
| dw_conv_bias | 128 (1D) | 128 | `for core=0 to 128 step 12 { for local ... }` |
| pw_conv_silu | 2×2 nested | 4 (1D) | `for core=0 to 4 step 12 { for local ... }` |

12-core 분산은 **MLIR upstream pass 2개**(affine-loop-coalescing + affine-loop-tile)만으로 달성 가능하여 커스텀 Pass가 불필요했다.

---

## 7. IREE Plugin 파이프라인 (초기 설계, 대기 중)

IREE Plugin은 컴파일러의 **초기 설계 방향**이다. 현재는 구조 변경으로 대기 상태이나, 설계된 내용은 추후 Independent 파이프라인 하단부(ISA/EXP 계층)의 설계 참고로 활용될 수 있다.

### 7.1 설계된 전체 IR 변환 흐름

```
ONNX 모델
    │ iree-import-onnx
    ▼
linalg MLIR
    │ IREE to CAP PassPipeline  ← ✅ 구현 완료
    ▼
INF_CAP Layer  (HW 실행 가능한 추상 연산)
    │ CAP to ISA PassPipeline   ← ⏸️ ISA 대기
    ▼
INF_ISA Layer  (opcode + 스케줄링)
    │ ISA to EXP PassPipeline
    ▼
INF_EXP Layer  (runtime 제어 정보)
    │ EXP to HAL + Serialization
    ▼
out_bin.ihnn / plan.json / out.mlir
```

### 7.2 IREE to CAP Pass (구현 완료)

| 순서 | Pass | 수행하는 변환 |
|---|---|---|
| 1 | **QDQFusionPass** | quantize/dequantize 쌍 → inf_cap.quant / inf_cap.dequant (5 패턴) |
| 2 | **OpFusionPass** | conv2d, depthwise_conv2d, batch_matmul, pooling_max, softmax, sigmoid → 각 inf_cap op |
| 3 | **PostOpFusionPass** | bias 추가, requantize 후처리 |
| 4 | **MonoOpFusionPass** | add → inf_cap.add, mul → inf_cap.mul (1:1 변환) |
| 5 | **VailPadPass** | insert_slice padding 체인 → conv padding attribute로 이전 |
| 6 | **QOpFusionPass** | dequantize → op → quantize 체인 → quantized op으로 fusion |
| 7 | **GenericOpenPass** | 남은 linalg.generic 제거 |
| 8 | **ExitIREEPass** | IREE 전용 IR 제거 |

YOLOv10n 기준: inf_cap.conv2d 69개, depthwise 13개, batch_matmul 2개, pooling 3개, softmax 1개, sigmoid 70개

### 7.3 ISA Layer 스케줄링 알고리즘 (설계됨)

ISA가 확정되면 이 계층에서 컴파일러가 12코어에 연산을 분산 배치한다:

1. 각 opcode의 latency 기록
2. 전체 그래프에서 Critical Path 탐색
3. Critical Path를 우선 1개 Core에 배치
4. 나머지 연산을 latency 긴 것부터 남은 Core에 분산
5. Data locality 기반으로 Core 재배치 조정

---

## 8. 컴파일러가 수행하는 일 — Pipeline · Pass · Pattern

이 섹션은 컴파일러의 기능을 **실제 구현된 Pass와 Pattern 동작을 기준으로** 설명한다.

### 8.1 Pipeline — 변환 단계의 순서

Pipeline은 여러 Pass를 순서대로 적용하는 변환 체인이다. **순서가 중요하다** — 앞 단계가 만들어낸 IR 구조를 다음 단계가 인식하는 방식으로 의존 관계가 있다.

**Independent Step 1~5 Pipeline의 의존 관계:**

```
InlineConstants              Step2에서 fold할 상수를 만들어냄
    → Canonicalize/CSE       상수 folding 기회 활용
    → IdentityElim           ZP=0 항등식 제거 (InlineConstants가 ZP 값을 인라인해야 가능)
    → Canonicalize/CSE       IdentityElim 후 새 fold 기회 처리
    → MathToQDQ              항등식 제거 후 남은 Q/DQ 수학식을 명시적 op으로
    → QuantizeRegion         MathToQDQ가 uniform_quantize/dequantize를 만들어야 인식 가능
    → Canonicalize/CSE       죽은 DQ 제거
    → QuantCleanup           QuantizeRegion 이후 남은 noop 처리
```

**IREE to CAP Pipeline의 의존 관계:**

```
QDQFusion                    Q/DQ 쌍이 묶여야 OpFusion이 QDQ 없는 패턴 인식 가능
    → OpFusion               named op 변환 (QDQFusion 선행 필수)
    → PostOpFusion           OpFusion이 만든 inf_cap op에 bias/requant 추가
    → MonoOpFusion           위 Pass들이 처리 못한 잔여 op 1:1 변환
    → VailPadPass            inf_cap conv op이 생성된 후 padding 분리
    → QOpFusionPass          Q/DQ로 감싼 op을 quantized op으로 fusion
```

### 8.2 Pass — 하나의 변환 단위

Pass는 IR 전체를 순회하며 하나의 변환 목적을 수행한다. 두 가지 순회 방식을 사용한다:

**Walk 방식** (단순 제거/교체):

| Pass | Walk 대상 | 수행하는 일 |
|---|---|---|
| **InlineConstantsPass** | 함수 인자 | manifest의 arg_idx → .npy 로드 → constant로 교체 |
| **IdentityElimPass** | 모든 op | subtract/add/multiply에서 ZP=0 상수를 찾아 항등식 제거 |
| **MathToQDQPass** | 모든 op | `round(x/scale)+zp` 수식을 `uniform_quantize` op으로 교체 |
| **QuantizeRegionPass** | compute_ops | 각 op을 시작점으로 입력을 역방향 추적 → i8 rewire |
| **QuantCleanupPass** | quantize ops | noop Q 제거, concat의 DQ 입력 fusion |

**RewritePattern + applyPatternsGreedily 방식** (패턴 기반 반복 적용):

| Pass | 패턴 예시 | 수행하는 일 |
|---|---|---|
| **QDQFusionPass** | 5개 패턴 | linalg.generic의 Q/DQ 수학식 → inf_cap.quant/dequant |
| **OpFusionPass** | IREEToQConvPattern 등 | linalg.generic의 conv/sigmoid/pooling 패턴 → inf_cap.* |
| **VailPadPass** | VailPadConvPattern 등 | insert_slice→conv 체인 → padding attribute가 있는 conv |

### 8.3 Rewrite Pattern — 특정 IR 구조를 다른 구조로 교체

**IREEToQConvPattern (conv2d 변환):**
- **찾는 것**: indexing_map이 NCHW conv 구조인 linalg.generic
- **검사 항목**: loop 수(6개 또는 4개), iterator 종류(parallel/reduction), stride/dilation attribute
- **만드는 것**: `inf_cap.conv2d` (stride, dilation, padding 속성 포함)
- **1×1 conv**: loop 수 4개 분기로 별도 처리

**VailPadConvPattern (padding 이전):**
- **추적하는 체인**: tensor.insert_slice → tensor.expand_shape → transpose → linalg.generic → inf_cap.conv2d
- **하는 일**: insert_slice의 offset 값에서 실제 padding 크기 계산, conv2d의 padding attribute로 통합
- **제거**: 체인 전체를 제거하고 padding attribute가 있는 단순 conv2d로 교체

**IREEToDepthwiseConvPattern (depthwise conv):**
- **찾는 것**: `linalg.depthwise_conv_2d_nhwc_hwc` named op
- **만드는 것**: `inf_cap.depthwise_conv2d` (kernel shape에서 kernel_h/w 추출)

---

## 9. 모델 컴파일 → 런타임 실행 전체 구상

> 컴파일러 완성 시나리오 기준 (ISA 확정 후 구현 예정)

### 9.1 컴파일 단계 (오프라인)

```
[입력] fp32 ONNX 모델

[1단계] 모델 준비 (onnx_prepare.py)
        fp32 ONNX → QDQ int8 ONNX
        Conv metadata 추출 (scale, weight, bias)

[2단계] StableHLO 변환 (model_to_stablehlo.py)
        QDQ ONNX → StableHLO MLIR
        가중치 manifest 생성

[3단계] 컴파일 (ind-compile)
        Step 1~5: 양자화 IR 생성
        Step 6+:  Lowering (ISA 확정 후)
        → out_bin.ihnn: 명령어 큐 + 실행 메타데이터
        → plan.json:    메모리 배치 계획
        → out.mlir:     디버그/프로파일용 MLIR
```

### 9.2 런타임 실행 (온디바이스)

> 런타임은 Host CPU에서 실행되며, NPU와의 인터페이스를 담당한다. NPU 내부의 Layer 실행 순서/동기화는 Q에 적재된 명령어에 의해 NPU 자체적으로 수행된다.

```
[Host CPU] Runtime — 추론 1회 실행 흐름

[Step 1] 컴파일러 바이너리 파싱
         │ out_bin.ihnn을 읽어 다음을 분리:
         │   - Queue records: 각 pCore가 실행할 명령어 스트림
         │   - Weight 데이터: SM에 적재할 가중치
         │   - aCore 프로그램: aCore가 실행할 주소/제어 명령
         │   - 메모리 배치 정보: plan.json (주소 할당 맵)
         ▼
[Step 2] 데이터 적재 (추론 전, 1회)
         │ ① Weight → SM의 지정 주소에 쓰기
         │ ② Queue records → 각 pCore Q에 쓰기
         │ ③ aCore 프로그램 → 각 aCore에 쓰기
         │
         │ ※ 이 단계에서 모든 명령어와 가중치가 NPU에 로드됨
         │ ※ Q가 512 records를 초과하면 이 단계에서 실패
         ▼
[Step 3] 입력 데이터 적재
         │ 입력 이미지(예: 640×640×3 i8) → SM의 지정 주소에 쓰기
         │
         │ ※ plan.json이 지정하는 입력 주소 사용
         ▼
[Step 4] NPU 시작 신호
         │ Host CPU → NPU에 시작(kick) 신호 전송
         │
         │ ※ NPU가 시작되면 Q의 명령어를 순차 실행
         │ ※ Layer 간 동기화, 코어 간 통신은 NPU 내부에서 수행
         │ ※ 런타임은 이 시점부터 완료를 기다림
         ▼
[Step 5] NPU 완료 확인 및 결과 읽기
         │ NPU 완료 신호(인터럽트 또는 폴링) 확인
         │ SM의 지정 출력 주소에서 결과 텐서 읽기
         │
         │ ※ plan.json이 지정하는 출력 주소 사용
         ▼
[완료] 추론 결과 반환
```

**핵심 포인트**:
- 런타임은 **데이터를 쓰고(Step 2~3), 시작 신호를 주고(Step 4), 결과를 읽는다(Step 5)**
- NPU 내부의 Layer 실행 순서, 코어 간 동기화, DMA prefetch 등은 **Q에 적재된 명령어가 제어** (런타임이 Layer 단위로 개입하지 않음)
- 컴파일러가 생성하는 것: Q records + weight 배치 + 주소 할당 (plan.json)
- 런타임이 하는 것: 파싱 → 적재 → 시작 → 대기 → 읽기

### 9.3 런타임과 컴파일러의 역할 경계

| 단계 | 컴파일러 (오프라인) | 런타임 (온디바이스) |
|:---|:---|:---|
| 명령어 생성 | Q record 생성 (Layer/tile/DMA 명령어 포함) | Q record를 각 pCore에 적재 |
| Weight 배치 | SM 내 weight 주소 결정, 바이너리에 포함 | 지정 주소에 weight 데이터 쓰기 |
| aCore 프로그램 | aCore 명령어 생성 (주소 계산, 제어 흐름) | 각 aCore에 프로그램 적재 |
| 입출력 주소 | plan.json에 입력/출력 SM 주소 명시 | 입력 쓰기 / 출력 읽기 |
| 실행 제어 | Layer 순서, 동기화 정보를 Q에 인코딩 | 시작 신호 → 완료 대기 |
| 코어 할당 | 코어별 타일 분배를 Q record에 반영 | 관여하지 않음 |

### 9.4 동기화 메커니즘 (NPU 내부, 설계 중)

> 아래는 NPU 내부에서 Q 명령어에 의해 자율적으로 수행되는 동기화. 런타임은 관여하지 않는다.

- **Token/Epoch 기반**: Layer 간 실행 순서 추적
- **fence(Layer_end)**: Layer 완료 시 신호 발행 → 다음 Layer 트리거
- **NC Ring**: 12코어 간 데이터 통신 (aCore 링, pCore 링 각각)
- **Data Hazard 방지**: 컴파일러 생성 스케줄에 의존성 정보 포함
- **런타임 인터럽트**: 전체 추론 완료 시에만 Host CPU에 알림

---

## 10. 환경 설정

### 10.1 필수 소프트웨어

| 소프트웨어 | 버전 | 비고 |
|---|---|---|
| **OS** | Ubuntu 22.04 LTS | GLIBC 2.35 필요. **20.04 불가** (libssl 충돌) |
| **Python** | 3.10 이상 | |
| **torch** | 2.9.0 | |
| **torch_xla** | 2.9.0 | StableHLO 변환 (venv_xla29 환경) |
| **onnxruntime** | 최신 | `pip install onnxruntime` |
| **onnx2torch** | 최신 | `pip install onnx2torch` |
| **CMake** | 3.20 이상 | Ubuntu 22.04 호환 버전으로 설치 확인 |

### 10.2 Python 환경 설정

```bash
# venv_xla29 가상환경 활성화
source ~/venv_xla29/bin/activate

# 설치 확인
python -c "import torch, torch_xla; print(torch.__version__, torch_xla.__version__)"
# → 2.9.0  2.9.0
```

**알려진 이슈:**
- pyenv로 빌드한 Python은 Ubuntu 22.04에서 `libssl.so.1.1` 없음 → rebuild 필요
- `/usr/local/bin/cmake`가 Ubuntu 20.04 잔재이면 교체 필요

### 10.3 컴파일러 빌드

```bash
# Independent 컴파일러 빌드
cmake -B build_ind -G Ninja -DCMAKE_BUILD_TYPE=Release [옵션]
cmake --build build_ind --target ind-opt ind-compile -j$(nproc)

# IREE Plugin 빌드 (CAP Layer 검증 등 필요 시)
cmake -B build_inf -G Ninja -DCMAKE_BUILD_TYPE=Release [옵션]
cmake --build build_inf --target iree-opt iree-compile -j$(nproc)
```

**빌드 결과물:**

| 실행파일 | 역할 |
|---|---|
| `ind-opt` | Pass 단독 적용. 단계별 디버깅 시 사용 |
| `ind-compile` | 파이프라인 일괄 실행. `--compile-to`로 중간 단계 멈춤 가능 |
| `iree-opt` | IREE Pass 단독 적용 |
| `iree-compile` | IREE 전체 파이프라인 실행 |

---

## 11. 예제 스크립트

### 11.1 Step 0: 모델 준비 및 StableHLO 변환

```bash
# [선택] fp32 ONNX → QDQ int8 + Conv metadata 추출
python onnx_prepare.py \
  --onnx model_fp32.onnx \
  --output-dir ./work
# 출력: ./work/quantized_int8sym.onnx, ./work/conv_metadata.json

# StableHLO MLIR 생성
python model_to_stablehlo.py \
  --model-dir ./work \
  --onnx model_fp32.onnx
# 출력: ./work/stablehlo.mlir, ./work/weights/manifest.json, ./work/weights/*.npy
```

> **다른 해상도 모델 사용 시**: `onnx_prepare.py` 내 calibration shape `[1, 3, 640, 640]`을 모델 입력 shape에 맞게 수정해야 한다.

### 11.2 Step 1~5: ind-compile (권장)

```bash
# Step 1~4 일괄 실행
ind-compile stablehlo.mlir \
  --manifest=weights/manifest.json \
  --compile-to=QDQFusion \
  --dump-after-each \
  --dump-dir=./dumps \
  --print-stats \
  --mlir-elide-elementsattrs-if-larger=32 \
  -o step4_qdq.mlir

# Step 5a: 양자화 영역 확장
ind-opt --quantize-region \
  --quantize-region-config=config/i8_allowed_ops.json \
  step4_qdq.mlir -o step5a.mlir
ind-opt --canonicalize --cse step5a.mlir -o step5a_clean.mlir

# Step 5b: noop 정리
ind-opt --quant-cleanup step5a_clean.mlir -o step5b.mlir
```

### 11.3 Step 1~5: ind-opt (단계별 디버깅)

```bash
# Step 1: 가중치 인라인
ind-opt --inline-constants \
  --inline-constants-manifest=weights/manifest.json \
  stablehlo.mlir -o step1.mlir

# Step 2: 정규화
ind-opt --canonicalize --cse step1.mlir -o step2.mlir

# Step 3: 항등 연산 제거
ind-opt --identity-elim --canonicalize --cse step2.mlir -o step3.mlir

# Step 4: Q/DQ op 명시화
ind-opt --math-to-qdq step3.mlir -o step4.mlir

# Step 5a: i8 영역 확장
ind-opt --quantize-region \
  --quantize-region-config=config/i8_allowed_ops.json \
  step4.mlir -o step5a.mlir
ind-opt --canonicalize --cse step5a.mlir -o step5a_clean.mlir

# Step 5b: 잔여 정리
ind-opt --quant-cleanup step5a_clean.mlir -o step5b.mlir
```

### 11.4 IREE Plugin 검증

```bash
# ONNX → MLIR 변환
iree-import-onnx model_qdq.onnx --opset-version 17 -o model_qdq.onnx.mlir

# 개별 Pass 검증
iree-opt --qdq-fusion    model_qdq.onnx.mlir -o step1_qdq.mlir
iree-opt --op-fusion     step1_qdq.mlir      -o step2_op.mlir
iree-opt --vail-pad      step2_op.mlir       -o step3_vail.mlir

# 전체 파이프라인 검증
iree-compile model_qdq.onnx.mlir \
  --iree-hal-target-backends=infetron_v2 \
  --mlir-print-ir-after-all \
  --mlir-disable-threading \
  --compile-to=executable-targets \
  --iree-global-opt-experimental-disable-conv-generalization \
  --iree-opt-generalize-matmul=false \
  --mlir-elide-elementsattrs-if-larger=32 \
  -o dummy.vmfb > pipeline.mlir 2>&1
```

### 11.5 결과 분석 팁

```bash
# Op별 개수 확인
grep -c "inf_cap.conv2d"    output.mlir
grep -c "linalg.generic"    output.mlir
grep -c "uniform_quantize"  output.mlir
grep -c "uniform_dequantize" output.mlir

# 특정 함수 구조 확인 (가중치 출력 생략)
ind-opt --mlir-elide-elementsattrs-if-larger=32 output.mlir | less

# pipeline.mlir에서 특정 단계 찾기
grep -n "===== \[" pipeline.mlir
```

---

## 12. 장비 사양 참고

### 12.1 개발 서버 권장 사양

| 항목 | 최소 | 권장 | 비고 |
|---|---|---|---|
| **OS** | Ubuntu 22.04 | Ubuntu 22.04 LTS | 20.04는 라이브러리 이슈 |
| **RAM** | 16GB | 32GB 이상 | LLVM/IREE 빌드 메모리 집약적 |
| **CPU** | 4코어 | 16코어 이상 | 빌드 병렬화 기준 |
| **디스크** | 50GB | 150GB 이상 | IREE + LLVM 빌드 결과물 크기 |

### 12.2 CPU 성능에 따른 빌드 시간 기대치

| 환경 | IREE 전체 빌드 | Independent 빌드 | Pass 1개 수정 후 증분 빌드 |
|---|---|---|---|
| 8코어 | 30~60분 | 10~20분 | 1~3분 |
| 16코어 이상 | 15~30분 | 5~10분 | 30초~1분 |

> 처음 빌드 시 LLVM/MLIR 전체가 컴파일되어 오래 걸린다. 이후 증분 빌드는 변경 파일만 재컴파일.

---

## 13. Git 저장소 정리

### 13.1 주요 저장소

| 저장소 | 내용 |
|---|---|
| **keti_publish_hardware_bundle** | 컴파일러 메인 (IREE Plugin + Independent) |
| **keti_hardware_IR_bundle** | HW IR Dialect/Pass (IREE 독립, pubH의 submodule) |
| **MyDocs** | 설계 문서, 보고서, 실험 기록 |

### 13.2 keti_publish_hardware_bundle 브랜치

| 브랜치 | 내용 |
|---|---|
| `main` | 안정 버전 (Step 1~4 확정 포함) |
| `step5-quantize-region` | Step 5 QuantizeRegionPass + QuantCleanupPass 확정 |
| `qdq-direct-conv` | QDQ Direct 파이프라인 (현재 main과 동기화) |
| `experiment/fused-32x32-conv` | **Lowering Pipeline L1~L8 실험** (Step 6+, 12-core 분산) |
| `affine-test` | affine 기반 bufferize + MemoryMapping 실험 |

### 13.3 keti_hardware_IR_bundle 브랜치

| 브랜치 | 내용 |
|---|---|
| `main` | INF_CAP Dialect + 8개 Pass 안정 버전 |
| `affine-test` | AffineFusionPass + MemoryMappingPass 실험 |
| `independent-test` | Independent 파이프라인 연동 실험 |

### 13.4 현재 활성 브랜치 (2026-04-02)

```
keti_publish_hardware_bundle : experiment/fused-32x32-conv
keti_hardware_IR_bundle      : independent-test
```

---

*작성: 2026-04-02*  
*기반 자료: ArchitectureSpecification v1.3, QDQ Direct 파이프라인.md, i8 허용 Op 목록.md, 소스코드 (pubH/independent/)*
