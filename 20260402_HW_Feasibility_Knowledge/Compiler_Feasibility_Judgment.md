# Infetron-V2 — 컴파일러 지원 가능 여부 판단 지식 문서

> **용도**: Claude Team Project 지식 문서  
> **대상 독자**: HW/ISA/런타임 개발자가 새로운 하드웨어 가정을 제시할 때, 컴파일러 측에서 지원 가능 여부를 판단하기 위한 참조 문서  
> **목적**: "이런 하드웨어/ISA를 만들면 컴파일러가 대응 가능한가?"에 대해 정확하고 일관된 판단 근거를 제공  
> **작성일**: 2026-04-02

---

## 목차

1. [이 문서의 사용 방법](#1-이-문서의-사용-방법)
2. [컴파일러 현재 상태 요약](#2-컴파일러-현재-상태-요약)
3. [판단 기준 프레임워크](#3-판단-기준-프레임워크)
4. [영역별 지원 범위 및 제약](#4-영역별-지원-범위-및-제약)
5. [판단 결과 출력 포맷](#5-판단-결과-출력-포맷)
6. [자주 발생하는 가정 유형별 판단 가이드](#6-자주-발생하는-가정-유형별-판단-가이드)

---

## 1. 이 문서의 사용 방법

### 1.1 사용 시나리오

HW/ISA/런타임 개발자가 다음과 같은 가정을 제시할 때:

- "UB를 512 KiB로 늘리면 컴파일러가 대응 가능한가?"
- "fp16 연산을 추가하면 컴파일러 수정이 필요한가?"
- "NHWC 레이아웃으로 바꾸면 어떻게 되나?"
- "코어를 16개로 늘리면?"
- "Weight streaming을 ISA로 지원하면?"
- "Depthwise conv를 HW에서 직접 지원하면?"

이 문서를 참조하여 아래 3단계 판정 중 하나로 답변한다.

### 1.2 판정 등급

| 등급 | 의미 | 컴파일러 작업 |
|:---|:---|:---|
| **A. 즉시 가능** | 현재 코드 그대로 지원 가능. 설정값만 변경. | 없음 (config 변경만) |
| **B. 수정 후 가능** | 코드 수정이 필요하지만 기존 아키텍처 내에서 해결 가능. | Pass 수정 / 신규 Pass 추가 |
| **C. 재설계 필요** | 파이프라인 구조 변경이 필요. 상당한 개발 기간 소요. | 파이프라인 단계 추가/변경 |
| **D. 현재 불가** | 근본적 제약으로 대응 불가. 외부 프레임워크 변경 또는 전면 재작성 필요. | MLIR/StableHLO 자체 한계 |

---

## 2. 컴파일러 현재 상태 요약

### 2.1 파이프라인 구조

```
[Python Importer]   ONNX → torch_xla → StableHLO MLIR + 가중치 manifest
        │
[Step 1~5]          quantized StableHLO 생성 (확정, 검증 완료)
        │
[Step 6+ Lowering]  fused ops → affine.for + memref → 12-core 분산 (실험 중)
```

### 2.2 현재 지원 범위 한 줄 요약

| 항목 | 현재 상태 |
|:---|:---|
| 입력 형식 | ONNX (fp32 또는 QDQ int8) |
| 양자화 | int8 대칭, per-tensor, ZP=0 |
| 레이아웃 | NCHW (standard conv), NHWC (depthwise) |
| 연산 | Conv2D, 1x1 Conv, Depthwise Conv, Add, Mul, Sigmoid, MaxPool |
| 타일링 | 32×32 고정 |
| 코어 분산 | 12-core, 2D block |
| 메모리 모델 | contiguous memref, zero-init alloc |
| 활성화 퓨전 | SiLU (sigmoid×x) 만 |
| 검증 모델 | YOLOv10n |

---

## 3. 판단 기준 프레임워크

가정이 주어지면 아래 체크리스트를 순서대로 확인한다.

### 3.1 판단 순서

```
Step 1. 가정 분류 → 어떤 영역의 변경인가? (§4 참조)
Step 2. 영향 범위 파악 → 어떤 Pass/단계에 영향을 주는가?
Step 3. 코드 변경 수준 → config / Pass 수정 / 파이프라인 변경 / 불가
Step 4. 부수 영향 → 다른 Pass에 연쇄 영향이 있는가?
Step 5. 판정 및 근거 → A/B/C/D + 구체적 이유
```

### 3.2 영향 전파 맵

하나의 변경이 어디까지 파급되는지를 보여주는 맵이다. 가정을 평가할 때 이 맵에서 영향받는 모든 노드를 확인해야 한다.

```
[데이터 타입 변경]
  → Step 0 (Importer: onnx_prepare quantization 설정)
  → Step 3 (IdentityElim: ZP=0 가정)
  → Step 4 (MathToQDQ: scale/zp 패턴)
  → Step 5 (QuantizeRegion: i8_allowed_ops)
  → Step 6 (StripQuantTypes: quant→plain 변환)
  → Lowering 전체 (타일 크기 계산에서 D_BYTES 변경)
  → Feasibility (UB/SM 사용량 전체 재계산)

[레이아웃 변경]
  → Step 0 (Importer: ONNX → StableHLO 변환)
  → Step 6+ (ConvTile32: dimension index 하드코딩)
  → Step 6+ (DepthwiseSplit: NHWC 가정)
  → Step 6+ (DistributeElementwise: concat axis)
  → Step 6+ (AffineBufferize: memref layout)

[메모리 크기 변경]
  → Feasibility 분석만 (config.json 변경)
  → 컴파일러 코드 영향 없음

[코어 수 변경]
  → CoreDistributePass (--num-cores 옵션)
  → affine-loop-tile (tile-size 파라미터)

[타일 크기 변경]
  → ConvTile32Pass (TILE_SIZE 상수)
  → TileToSCFPass (타일 경계 처리)
  → ForallToAffinePass (affine_map 계수)
  → AffineBufferizeFusedPass (alloc 크기)
  → Feasibility (UB_need 재계산)

[새 연산 추가]
  → i8_allowed_ops.json (compute/transparent/never 분류)
  → QuantizeRegionPass (i8 변환 범위)
  → FuseConvSiluPass (퓨전 패턴 추가 시)
  → Lowering 각 Pass (새 op 처리 분기)
```

---

## 4. 영역별 지원 범위 및 제약

### 4.1 메모리 크기 (Q / UB / SM)

| 가정 | 판정 | 근거 |
|:---|:---|:---|
| Q 크기 변경 | **A. 즉시 가능** | config.json의 `q_size_bytes`, `q_record_size` 변경만으로 분석 가능. 컴파일러 코드 무관. |
| UB 크기 변경 | **A. 즉시 가능** | config.json의 `ub_size_bytes` 변경. 컴파일러의 타일 크기는 별도 설정. |
| SM 크기 변경 | **A. 즉시 가능** | config.json의 `sm_size_bytes` 변경. |
| UB ping-pong on/off | **A. 즉시 가능** | config.json의 `ub_half_bytes` 변경. |
| 메모리 뱅크 수 변경 | **A. 즉시 가능** | 현재 컴파일러는 뱅크 구조를 고려하지 않음. 뱅크 충돌 최적화가 필요해지면 **B**. |

### 4.2 코어 구성

| 가정 | 판정 | 근거 |
|:---|:---|:---|
| 코어 수 변경 (예: 8, 16, 24) | **A. 즉시 가능** | `--num-cores` CLI 옵션으로 변경 가능. `affine-loop-tile="tile-size=N"`도 변경. |
| aCore + pCore 동시 사용 | **C. 재설계 필요** | 현재 pCore만 가정. aCore용 명령어 생성, 이종 코어 스케줄링 파이프라인 필요. |
| 코어 간 통신 토폴로지 변경 (Ring → Mesh) | **B. 수정 후 가능** | CoreDistributePass의 2D block 할당 로직 수정. 데이터 이동 패턴만 변경. |
| 클러스터 단위 할당 (3×4 → 4×4) | **A. 즉시 가능** | `--num-cores=16`. 2D block 할당은 자동 조정. |

### 4.3 데이터 타입

| 가정 | 판정 | 근거 |
|:---|:---|:---|
| i8 유지 | **A. 즉시 가능** | 현재 기본. |
| i4 (4-bit 양자화) | **C. 재설계 필요** | StableHLO가 i4를 제한적 지원. Importer, quantization, 모든 lowering pass에서 i4 경로 필요. MLIR quant dialect에서 i4 uniform type 지원 여부 확인 필요. |
| i16 | **B. 수정 후 가능** | i8_allowed_ops.json의 type 확장 + StripQuantTypesPass에서 i16 처리 + Feasibility D_BYTES=2. 각 Pass에서 i8 하드코딩 부분 수정. |
| fp16 (half precision) | **C. 재설계 필요** | 양자화 파이프라인(Step 0~5) 전면 재설계. fake QDQ wrapping 불필요. StableHLO fp16은 지원하지만 quantize/dequantize 경로가 완전히 다름. |
| fp32 (float 그대로) | **B. 수정 후 가능** | Step 0~5 스킵, Step 6+만 사용. 단, 현재 Lowering pass들이 quant type 가정하는 부분 수정 필요. Feasibility D_BYTES=4로 재계산 시 UB/SM 매우 빡빡. |
| 혼합 정밀도 (i8 conv + fp16 activation) | **C. 재설계 필요** | 타입 전환 지점에 cast op 삽입 로직 필요. 퓨전 패턴 전면 재검토. |

### 4.4 연산 (Op) 지원

| 가정 | 판정 | 근거 |
|:---|:---|:---|
| 기존 compute_ops (conv, add, mul, sigmoid, pooling) | **A. 즉시 가능** | 현재 지원 중. |
| ReLU 추가 | **B. 수정 후 가능** | `stablehlo.clamp` 또는 `stablehlo.maximum` 패턴. i8_allowed_ops에 추가 + Lowering에서 fused.conv_relu 패턴 추가. |
| GELU / Swish / Mish 추가 | **B. 수정 후 가능** | FuseConvSiluPass와 유사한 퓨전 패턴 작성. 단, 정확한 수학적 분해가 StableHLO에서 어떻게 표현되는지 분석 필요. |
| Softmax HW 가속 | **B. 수정 후 가능** | `stablehlo.reduce` + `stablehlo.exponential` 패턴 매칭. 단, exponential은 현재 never_allowed → allowed로 이동 또는 별도 fused op 필요. |
| Attention / MatMul HW 가속 | **B. 수정 후 가능** | `stablehlo.dot_general` 매칭. 현재 batch_matmul은 IREE Plugin에서만 지원. Independent 파이프라인에 추가 필요. |
| Deformable Conv | **C. 재설계 필요** | ONNX → StableHLO 변환에서 offset grid 처리 필요. 동적 인덱싱이라 정적 분석 어려움. |
| Dynamic shape 연산 | **D. 현재 불가** | 전 파이프라인이 정적 shape 가정. `torch.export`의 dynamic_shapes 지원 필요. affine loop 경계가 상수여야 하는 제약. |

### 4.5 레이아웃

| 가정 | 판정 | 근거 |
|:---|:---|:---|
| NCHW (현재) | **A. 즉시 가능** | 기본. |
| NHWC 전면 전환 | **C. 재설계 필요** | ConvTile32Pass, TileToSCFPass, DistributeElementwisePass, AffineBufferizeFusedPass 모두 dimension index 하드코딩. 예: `outH = resultType.getDimSize(2)` → `getDimSize(1)`. 모든 Pass 동시 수정 + 테스트 필요. |
| HW가 NHWC를 요구하되, 컴파일러는 내부적으로 NCHW 사용 | **B. 수정 후 가능** | 최종 출력 직전에 transpose 삽입하는 Pass 1개 추가. |
| 채널 방향 interleave (예: C를 8의 배수로 패딩) | **B. 수정 후 가능** | 채널 패딩 + 재배열 Pass 추가. Feasibility에서 실효 채널 수 변경 반영. |

### 4.6 타일링

| 가정 | 판정 | 근거 |
|:---|:---|:---|
| 32×32 유지 | **A. 즉시 가능** | 현재 기본. |
| 다른 고정 크기 (16, 64, 등) | **B. 수정 후 가능** | `ConvTile32Pass.cpp`의 `TILE_SIZE` 상수 변경. 단, CLI 옵션화가 바람직 → 코드 1줄 수정. 연관 Pass 테스트 필요. |
| 동적 타일 크기 (layer별 최적 타일) | **B. 수정 후 가능** | ConvTile32Pass에 per-layer adaptive 로직 추가. Feasibility의 adaptive tiling 결과를 컴파일러에 피드백하는 구조 필요. |
| 비정방 타일 (예: 32×64) | **B. 수정 후 가능** | TILE_H, TILE_W 분리. ConvTile32 + TileToSCF + ForallToAffine에서 2차원 독립 처리. |
| 3D 타일 (spatial + channel) | **C. 재설계 필요** | 현재 spatial만 타일링. 채널 방향 타일링은 weight splitting 로직 + partial sum accumulation 필요. |

### 4.7 메모리 접근 모델

| 가정 | 판정 | 근거 |
|:---|:---|:---|
| Weight를 SM에 전량 적재 | **A. 즉시 가능** | 현재 기본. |
| Weight streaming (SM → UB slice 전송) | **B. 수정 후 가능** | AffineBufferizeFusedPass에서 weight alloc을 DMA descriptor로 변환하는 로직 추가. ISA에 DMA 명령어 지원 필수. |
| UB double buffering (HW 지원) | **A. 즉시 가능** | CoreDistributePass가 이미 prefetch annotation을 생성. HW가 이를 해석하면 됨. |
| SM에서 직접 연산 (SM read port) | **B. 수정 후 가능** | weight를 UB에 복사하지 않고 SM에서 직접 읽는 경우. Bufferize에서 weight memref를 SM address로 매핑하는 Pass 필요. Feasibility의 UB 계산에서 weight 제외. |
| Scratchpad 메모리 추가 | **B. 수정 후 가능** | 새로운 memref address space 정의 + alloc 할당 로직 추가. |
| 외부 DRAM 접근 | **B. 수정 후 가능** | SM에 들어가지 않는 weight를 DRAM에서 layer 단위로 로드하는 DMA pass 추가. |

### 4.8 양자화 정책

| 가정 | 판정 | 근거 |
|:---|:---|:---|
| Per-tensor symmetric (현재) | **A. 즉시 가능** | 기본. |
| Per-channel quantization | **B. 수정 후 가능** | onnx_prepare.py에서 `per_channel=True`. InlineConstantsPass에서 채널별 scale 처리. IdentityElimPass에서 ZP=0 조건 유지 시 영향 적음. MathToQDQPass에서 per-channel scale 패턴 매칭. |
| Asymmetric quantization (ZP≠0) | **B. 수정 후 가능** | IdentityElimPass가 ZP=0 가정 → 수정 필요. MathToQDQPass의 `(x-zp)*scale` 패턴은 이미 ZP를 포함. 연산 시 ZP correction 추가 비용 발생. |
| Mixed precision (layer별 다른 bit width) | **C. 재설계 필요** | 양자화 파이프라인에서 layer별 precision 결정 로직 + cast 삽입 + 각 lowering pass에서 다중 타입 처리. |

### 4.9 ISA 명령어

| 가정 | 판정 | 근거 |
|:---|:---|:---|
| VLIW 명령어 생성 | **C. 재설계 필요** | 현재 Pipeline.cpp의 CodeGen phase가 미구현. ISA 확정 후 CodeGen Pass 작성 필요. |
| 특수 명령어 (Fused MAC, LUT sigmoid) | **B. 수정 후 가능** | fused dialect에 대응 op 추가 + lowering pass에서 fused op → 특수 명령어 매핑. |
| SIMD 폭 변경 | **B. 수정 후 가능** | 타일링과 코어 분배에서 SIMD 폭 반영. 현재는 element 단위 처리. |
| 가변 길이 명령어 | **C. 재설계 필요** | Q record 크기(256B) 고정 가정. 가변이면 Q 분석 전면 수정 + 명령어 패킹 로직. |

### 4.10 런타임 인터페이스 (바이너리 포맷 / 적재 절차)

> 런타임의 역할: 컴파일러 바이너리 파싱 → weight/Q/aCore 적재 → 입력 쓰기 → NPU 시작 → 완료 대기 → 결과 읽기

| 가정 | 판정 | 근거 |
|:---|:---|:---|
| 바이너리 포맷 변경 (ihnn → 다른 포맷) | **B. 수정 후 가능** | CodeGen(미구현) 단계에서 직렬화 포맷 결정. 런타임 파서와 합의 필요. |
| Q record를 런타임이 동적으로 생성 | **D. 현재 불가** | 현재 설계는 컴파일러가 Q record를 전부 생성하고 런타임은 그대로 적재. 런타임 동적 생성은 컴파일러 범위 밖. |
| Weight를 런타임이 SM 주소를 자유롭게 결정 | **B. 수정 후 가능** | 현재는 컴파일러가 plan.json에 주소를 확정. 런타임 자유 배치를 허용하려면 Q record의 주소 참조를 재배치 가능(relocatable)하게 생성해야 함. |
| 입력 주소를 런타임이 결정 | **A. 즉시 가능** | plan.json에서 입력 주소를 런타임이 덮어쓸 수 있게 약속하면 됨. 컴파일러는 오프셋 기반 주소만 생성. |
| 출력 주소를 런타임이 결정 | **A. 즉시 가능** | 위와 동일. |
| 추론 중 Q 재로딩 (Q 크기 부족 시) | **C. 재설계 필요** | 현재 Q는 1회 적재 가정. 재로딩 허용 시: (1) 컴파일러가 Q를 chunk 단위로 분할 생성, (2) 런타임이 chunk 완료 시 다음 chunk 적재, (3) NPU가 Q 소진 시 런타임에 인터럽트. Feasibility의 Q 제약이 사실상 해소. |
| aCore 프로그램 변경 (제어 흐름 변경) | **C. 재설계 필요** | 현재 aCore 프로그램 생성 미구현. ISA 확정 후 aCore CodeGen 작성 필요. |
| 런타임이 Layer 단위로 NPU를 제어 | **C. 재설계 필요** | 현재 설계: 전체 추론을 1회 시작/완료로 처리. Layer 단위 제어는 Q 구조 전면 변경 + 런타임-NPU 간 인터럽트 프로토콜 필요. |
| 멀티 모델 동시 실행 | **C. 재설계 필요** | SM/Q 자원 분할 + 코어 파티셔닝 + 런타임 스케줄러. 컴파일러가 자원 사용량을 쿼리 가능한 메타데이터 생성 필요. |

**런타임 개발자를 위한 핵심 포인트**:

| 구분 | 컴파일러가 결정 | 런타임이 수행 |
|:---|:---|:---|
| **Q** | record 내용, 개수, 코어별 배분 | 바이너리에서 파싱 → 각 pCore Q에 쓰기 |
| **Weight** | SM 내 배치 주소, 데이터 | 바이너리에서 파싱 → SM 지정 주소에 쓰기 |
| **aCore** | 프로그램 코드 | 바이너리에서 파싱 → 각 aCore에 쓰기 |
| **입력** | SM 내 입력 주소 (plan.json) | 입력 데이터를 지정 주소에 쓰기 |
| **실행** | Layer 순서, 동기화 (Q에 인코딩) | 시작 신호 → 완료 대기 |
| **출력** | SM 내 출력 주소 (plan.json) | 지정 주소에서 결과 읽기 |

---

## 5. 판단 결과 출력 포맷

가정에 대한 판단 결과는 반드시 아래 형식으로 반환한다.

### 5.1 포맷 구조

```
## 컴파일러 지원 가능 여부 판단

### 요청 가정
{HW/ISA 개발자가 제시한 가정을 원문 그대로 인용}

### 판정 결과

| 항목 | 내용 |
|:---|:---|
| **판정** | {A / B / C / D} — {즉시 가능 / 수정 후 가능 / 재설계 필요 / 현재 불가} |
| **영향 영역** | {영향받는 Pass/단계 목록} |
| **코드 변경 규모** | {없음 / config만 / Pass N개 수정 / 신규 Pass N개 / 파이프라인 단계 추가} |
| **선행 조건** | {이 가정이 성립하려면 다른 어떤 것이 먼저 확정되어야 하는지} |
| **Feasibility 영향** | {Q/UB/SM 계산에 미치는 영향} |

### 판단 근거

**현재 컴파일러 상태**:
- {관련 Pass/코드의 현재 동작 설명}
- {현재 하드코딩된 값이나 가정}

**가정 적용 시 변경점**:
1. {변경이 필요한 코드/설정 ①}
   - 파일: {파일 경로}
   - 현재: {현재 동작}
   - 변경: {필요한 변경}
2. {변경이 필요한 코드/설정 ②}
   - ...

**부수 영향**:
- {다른 Pass/단계에 미치는 연쇄 영향}
- {Feasibility 수치 변화 예측}

**위험 요소**:
- {변경 시 발생 가능한 문제}
- {테스트/검증이 필요한 항목}

### 결론 및 권고

{1~3문장으로 결론. 가능하면 구체적 action item 포함.}
```

### 5.2 복수 가정이 동시에 제시된 경우

각 가정을 독립적으로 판정한 뒤, **조합 판정**을 추가한다.

```
### 조합 판정

| 가정 | 개별 판정 | 조합 시 추가 영향 |
|:---|:---|:---|
| {가정 1} | {A/B/C/D} | {가정 2와 결합 시 추가 고려사항} |
| {가정 2} | {A/B/C/D} | {가정 1과 결합 시 추가 고려사항} |

**종합 판정**: {가장 높은(=어려운) 등급}
**이유**: {조합으로 인해 추가로 발생하는 복잡성 설명}
```

### 5.3 포맷 적용 규칙

1. **판정 등급은 반드시 A/B/C/D 중 하나**. 모호한 표현 ("아마 가능", "될 수도") 금지.
2. **근거에 파일 경로 포함**. 어떤 코드를 왜 수정해야 하는지 구체적으로.
3. **Feasibility 수치 영향 포함**. 메모리 관련 가정이면 UB/SM/Q 수치가 어떻게 바뀌는지.
4. **부수 영향 반드시 기술**. 하나의 변경이 다른 Pass에 연쇄 영향을 줄 수 있다.
5. **선행 조건 명시**. ISA 확정, HW 스펙 확정 등 이 판단이 유효하기 위한 전제.

---

## 6. 자주 발생하는 가정 유형별 판단 가이드

### 6.1 "메모리 크기를 바꾸면?"

**이것은 컴파일러 문제가 아니라 Feasibility 문제다.**

→ `HW_Feasibility_Knowledge.md` 참조. config.json 변경 후 calculate.py 실행.
→ 컴파일러 코드 수정 불필요 (판정: **A**).

다만, 메모리 크기 변경에 따라 **최적 타일 크기**가 달라질 수 있다. 이 경우:
- Feasibility의 adaptive tiling 결과에 맞춰 ConvTile32Pass의 TILE_SIZE를 조정하면 최적 (판정: **B**, 코드 1줄).

### 6.2 "새로운 연산을 HW로 지원하면?"

```
질문: "HW에서 {Op}을 직접 지원한다. 컴파일러가 이것을 활용할 수 있나?"

판단 순서:
1. 해당 Op이 StableHLO에 존재하는가?
   → 없으면: ONNX → StableHLO 변환 단계에서 커스텀 처리 필요 (B 이상)
   → 있으면: 다음 단계

2. 해당 Op이 i8_allowed_ops.json에서 어디에 속하는가?
   → compute_ops: i8 변환 이미 지원 (A)
   → transparent_ops: 타입 전파만 (A)
   → never_allowed: i8 변환 불가, f32로 유지 (B: 별도 처리 경로 필요)
   → 목록에 없음: 추가 필요 (B)

3. Lowering에서 해당 Op을 어떻게 처리하는가?
   → 기존 fused op으로 매핑 가능: FuseConvSiluPass 류 패턴 추가 (B)
   → 새로운 fused op 정의 필요: fused dialect 확장 + lowering Pass 추가 (B)
   → 기존 패턴에 맞지 않음: (C)
```

### 6.3 "데이터 타입을 바꾸면?"

```
질문: "HW에서 {dtype}을 지원한다. 컴파일러에서 사용 가능한가?"

영향 범위 체크리스트:
□ Step 0 (Importer): 양자화 설정 변경 필요?
  - i8 외 타입 → onnx_prepare.py의 quantize 설정 변경
  - fp16/fp32 → 양자화 파이프라인 자체를 스킵

□ Step 3 (IdentityElim): ZP=0 가정이 유효한가?
  - 대칭 양자화 유지 → 유효
  - 비대칭 → IdentityElimPass 수정

□ Step 4 (MathToQDQ): scale/zp 패턴이 동일한가?
  - 동일한 Q/DQ 패턴 → 유효
  - 다른 양자화 방식 → MathToQDQPass 수정

□ Step 5 (QuantizeRegion): i8_allowed_ops 타입 변경
  - input_type / output_type 필드 수정

□ Step 6 (StripQuantTypes): quant → plain 변환 타입 변경
  - 현재 quant.uniform<i8:f32> → i8
  - i16이면 quant.uniform<i16:f32> → i16

□ Lowering 전체: D_BYTES 변경
  - 타일 크기 계산, memref 타입, alloc 크기 모두 영향

□ Feasibility: D_BYTES 변경
  - UB_need, SM peak 전부 재계산
  - 2byte dtype → 메모리 사용량 2배 → FAIL 증가 예상
```

### 6.4 "코어 토폴로지를 바꾸면?"

```
질문: "코어를 {N}개, {topology} 구조로 바꾼다."

코어 수만 변경:
  → --num-cores=N + affine-loop-tile="tile-size=N" (판정: A)

토폴로지 변경 (예: Ring → 2D Mesh → NoC):
  → CoreDistributePass의 2D block 할당 로직 수정 (판정: B)
  → 데이터 이동 최적화가 필요하면 DMA 스케줄링 Pass 추가 (판정: B~C)

이종 코어 (aCore + pCore 동시 사용):
  → 코어별 역할 분담 + 이종 스케줄링 (판정: C)
  → ISA가 확정되어야 구체적 판단 가능
```

### 6.5 "Weight streaming을 지원하면?"

```
이 가정은 Feasibility + 컴파일러 양쪽에 영향:

Feasibility 측:
  → config.json의 weight_in_ub: false
  → UB 계산에서 weight 제외 → UB FAIL 대폭 감소
  → calculate.py 재실행으로 즉시 확인 가능 (A)

컴파일러 측:
  → AffineBufferizeFusedPass에서 weight를 UB에 alloc하지 않고
    SM address를 직접 참조하도록 변경 (B)
  → ISA에서 SM direct read 또는 DMA streaming 명령어 필요 (선행 조건)
  → 전제: ISA가 SM → 연산 유닛 직접 경로 지원

종합 판정: B (컴파일러 수정 필요, 단 ISA 확정 선행)
```

---

## 부록. 컴파일러 코드 참조 테이블

판단 시 참조해야 하는 핵심 파일과 하드코딩된 가정:

| 파일 | 하드코딩 가정 | 변경 시 영향 |
|:---|:---|:---|
| `ConvTile32Pass.cpp:26` | `TILE_SIZE = 32` | 타일 크기 전체 변경 |
| `CoreDistributePass.cpp:33` | `init(12)` (기본 코어 수) | CLI로 변경 가능, 코드 수정 불필요 |
| `AffineBufferizeFusedPass.cpp` | contiguous memref, zero-init | 메모리 모델 변경 시 수정 |
| `ForallToAffinePass.cpp` | `bodyBuilder` 콜백 사용 | affine.for 생성 방식 변경 시 |
| `StripQuantTypesPass.cpp` | `quant.uniform<i8:f32>` → `i8` | 데이터 타입 변경 시 |
| `FuseConvSiluPass.cpp` | SiLU = sigmoid × x 패턴 | 다른 활성화 퓨전 추가 시 |
| `i8_allowed_ops.json` | 5 compute + 6 transparent + 9 never | 연산 추가/제거 시 |
| `config.json` | Q/UB/SM 크기, 코어 수, 정책 | Feasibility 재분석 시 |
| `model_data.json` | YOLOv10n/s layer 데이터 | 새 모델 분석 시 |
| `onnx_prepare.py` | per-tensor, symmetric, ZP=0, opset 17 | 양자화 정책 변경 시 |
| `model_to_stablehlo.py` | input_locations 기반 manifest | Importer 변경 시 |
| `Pipeline.cpp:90,100` | HLOOpt, CodeGen = 미구현 | ISA 확정 후 구현 대상 |
