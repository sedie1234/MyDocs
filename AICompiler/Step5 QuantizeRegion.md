# Step5 QuantizeRegion

**Pass**: `--quantize-region` + `--canonicalize --cse` + `--quant-cleanup`
**파일**: `QuantizeRegionPass.cpp`, `QuantCleanupPass.cpp`
**Config**: `independent/config/i8_allowed_ops.json`

---

## 역할

i8 연산 가능한 영역의 Q→DQ 경계를 제거하여 quantized 연산으로 변환.

## 하위 단계

### Step5a: QuantizeRegion

DQ를 직접 제거하지 않고, **allowed op의 input을 DQ output → DQ input(quant type)으로 rewire**.
DQ는 dead code가 되면 다음 canonicalize에서 자동 제거.

→ 상세: [[Step5 QuantizeRegion 변환 규칙]]

### Step5a→canon: Canonicalize + CSE

dead DQ 자동 제거 + 중복 정리.

### Step5b: QuantCleanup

1. **NoopQuantize 제거**: same-scale `uniform_quantize(quant<i8,S> → quant<i8,S>)` → identity
2. **ConcatFusion**: f32 concat의 모든 input이 DQ + 모든 user가 Q → quant concat

## Op 분류

→ 상세: [[i8 허용 Op 목록]]

| 분류 | 역할 | 예시 |
|---|---|---|
| compute_ops | i8 연산 변환 대상 | conv, add, logistic, multiply |
| transparent_ops | 자동 type 전파 | broadcast, reshape, transpose, slice |
| never_allowed | f32 필수 | exp, divide, log |
| unknown | 경계 유지 (안전) | dot_general, concatenate |

## 결과 (yolov10n)

| | Step4 | Step5b |
|---|---|---|
| uniform_dequantize | 479 | **17** (-96%) |
| uniform_quantize | 311 | **276** |
| conv quant | 0/83 | **83/83** |
| logistic quant | 0/70 | **70/70** |
| add quant | 0/99 | **93/99** |
| multiply quant | 0/76 | **73/76** |
| lines | 1752 | **1255** |

## 잔여 f32 영역

→ 상세: [[Step5 fp32 잔여 분석]]

| 카테고리 | 변환 | 이유 |
|---|---|---|
| Conv+SiLU | ✅ 완료 | 핵심 경로 100% |
| Attention | ❌ f32 | softmax (exp, divide) |
| Max Pooling | ⚠️ 가능 | pad 추가 필요 |
| Resize | ❌ f32 | interpolation |
| Detection Head | ❌ f32 | 후처리 (TopK, NMS) |

## 실행

```bash
ind-opt --quantize-region --quantize-region-config=config/i8_allowed_ops.json \
  step4_qdq.mlir -o step5a_qregion.mlir
ind-opt --canonicalize --cse step5a_qregion.mlir -o step5a2_canon.mlir
ind-opt --quant-cleanup step5a2_canon.mlir -o step5b_cleanup.mlir
```

## 관련

- [[Step4 MathToQDQ]] — 이전 단계
- [[Step5 QuantizeRegion 변환 규칙]] — DQ rewire 알고리즘
- [[Step5 fp32 잔여 분석]] — 변환 불가 영역
- [[i8 허용 Op 목록]] — op 분류 config
- [[QDQ Direct 파이프라인]] — 전체 파이프라인
