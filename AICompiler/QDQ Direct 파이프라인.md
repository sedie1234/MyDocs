# QDQ Direct 파이프라인

**최종 갱신**: 2026-03-30
**브랜치**: `qdq-direct-conv` (main), `step5-quantize-region` (Step5 확정)

---

## 개요

QDQ ONNX 모델을 직접 onnx2torch로 변환하여 StableHLO를 생성하고, i8 연산으로 변환하는 파이프라인.

이전 [[Quantized StableHLO 파이프라인]]의 fake QDQ 방식을 대체.

주요 개선:
- fake QDQ wrapping 불필요 (QDQ ONNX 직접 변환)
- [[input_locations 기반 InlineConstantsPass|input_locations]]로 bias 매칭 100% 정확
- conv 83/83 i8 변환 완료

---

## 확정 파이프라인 (Step0~5)

```
[Step 0]  [Python] QDQ ONNX → onnx2torch → torch_xla → StableHLO + manifest
[Step 1]  --inline-constants        manifest arg_idx → constants inline
[Step 2]  --canonicalize --cse      공통 부분식 제거
[Step 3]  --identity-elim + canon   zp=0 항등 연산 제거
[Step 4]  --math-to-qdq             explicit math → uniform_quantize/uniform_dequantize
[Step 5a] --quantize-region         allowed op의 input을 DQ output → DQ input으로 rewire
          --canonicalize --cse      dead DQ 자동 제거
[Step 5b] --quant-cleanup           noop Q 제거 + f32 concat → quant concat fusion
```

### 각 Step 상세

- [[Step0 모델 준비]] — ONNX → onnx2torch → StableHLO + manifest
- [[Step1 InlineConstants]] — manifest → constants inline
- [[Step2 Canonicalize + CSE]] — MLIR 내장 pass
- [[Step3 IdentityElim]] — zp=0 항등 연산 제거
- [[Step4 MathToQDQ]] — explicit math → uniform Q/DQ
- [[Step5 QuantizeRegion]] — i8 영역 변환 + cleanup

---

## 실행 명령

### Step0: StableHLO 생성

```bash
python model_to_stablehlo.py \
  --model-dir ./work --onnx model_fp32.onnx --output-dir ./work
```

### Step1~4: ind-compile

```bash
ind-compile stablehlo.mlir \
  --manifest=weights/manifest.json \
  --compile-to=QDQFusion \
  --dump-after-each --dump-dir=./dumps --print-stats \
  -o step4_qdq.mlir
```

### Step5: ind-opt

```bash
ind-opt --quantize-region --quantize-region-config=config/i8_allowed_ops.json \
  step4_qdq.mlir -o step5a_qregion.mlir
ind-opt --canonicalize --cse step5a_qregion.mlir -o step5a2_canon.mlir
ind-opt --quant-cleanup step5a2_canon.mlir -o step5b_cleanup.mlir
```

---

## 검증 결과 (YOLOv10n)

| Step | Q | DQ | conv quant | lines |
|---|---|---|---|---|
| Step4 | 311 | 479 | 0/83 | 1752 |
| Step5a | 311 | ~80 | 83/83 | ~1340 |
| **Step5b** | **276** | **17** | **83/83** | **1255** |

conv 83/83 i8 변환, logistic 70/70, add 93/99, DQ 96% 제거.

| Op | 전체 | quant | f32 |
|---|---|---|---|
| convolution | 83 | **83** | 0 |
| logistic | 70 | **70** | 0 |
| add | 99 | **93** | 4 (reduce body) + 2 |
| multiply | 76 | **73** | 3 |

---

## 파일 구성

```
independent/
├── config/i8_allowed_ops.json       ← [[i8 허용 Op 목록]]
├── lib/Transforms/
│   ├── InlineConstantsPass.cpp      ← Step1
│   ├── IdentityElimPass.cpp         ← Step3
│   ├── MathToQDQPass.cpp            ← Step4
│   ├── QuantizeRegionPass.cpp       ← Step5a
│   └── QuantCleanupPass.cpp         ← Step5b
├── scripts/
│   ├── onnx_prepare.py              ← converter 등록
│   └── model_to_stablehlo.py        ← StableHLO 생성
└── test/rework/qdq_direct/          ← 결과물
    ├── step1~step5b .mlir + _view.mlir
    └── weights/manifest.json
```

---

## 관련 문서

- [[컴파일러 개요]] — Independent / IREE 분리 구조
- [[Quantized StableHLO 파이프라인]] — 이전 방식 (fake QDQ)
- [[input_locations 기반 InlineConstantsPass]] — bias 매칭 해결
- [[i8 허용 Op 목록]] — compute_ops / transparent_ops 분류
- [[Step5 QuantizeRegion 변환 규칙]] — DQ rewire 알고리즘
- [[Step5 fp32 잔여 분석]] — 변환 불가 영역
- [[Independent Compiler 파이프라인]] — 파이프라인 상세
