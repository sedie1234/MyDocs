# Step5 fp32 잔여 분석

**최종 갱신**: 2026-03-30
**입력**: `step5b_cleanup.mlir`
**모델**: yolov10n

---

## 현황

Conv+SiLU 핵심 경로: **100% quant 변환 완료**
잔여 f32: attention, pooling, resize, detection head

## 카테고리별 분석

### Attention (L555~585) — ❌ f32 유지

PSA (Partial Self-Attention) 모듈.

| Op | 설명 |
|---|---|
| `dot_general` (4) | Q·K matmul, score·V matmul |
| `exponential` (2) | softmax exp |
| `divide` (2) | softmax normalize |
| `reduce(add/max)` (3) | softmax sum/max |
| `subtract` (1) | x - max |

softmax 내부의 exp, divide가 f32 필수.

### Max Pooling (L493~510) — ⚠️ 변환 가능

SPP 모듈의 5x5, 9x9, 13x13 max pooling.

| Op | 설명 |
|---|---|
| `pad` (3) | pooling 전 padding |
| `reduce_window(max)` (3) | max pooling |

`pad`를 transparent_ops에 추가하면 변환 가능.
`reduce_window`는 이미 compute_ops에 있음.

### Resize/Deformable (L630~710) — ❌ f32 유지

Bilinear interpolation (2x upsample).

| Op | 설명 |
|---|---|
| `gather` (10) | grid sampling |
| `dot_general` (2) | interpolation weight |
| `floor`, `iota` | grid 좌표 |
| `concatenate` (2) | grid xy 결합 |

grid 좌표가 부동소수점이므로 f32 필수.

### Detection Head (L1130~) — ❌ f32 유지

DFL + TopK + NMS.

| Op | 설명 |
|---|---|
| `exponential`, `divide`, `reduce` | DFL softmax |
| `gather` (5) | TopK 결과 수집 |
| `sort` | TopK |

후처리는 CPU/호스트 실행이 일반적.

## 요약

| 카테고리 | f32 op 수 | i8 변환 | 비고 |
|---|---|---|---|
| Conv+SiLU | 0 | ✅ 완료 | 83 conv, 70 logistic |
| Attention | ~10 | ❌ | softmax 필수 |
| Max Pooling | 6 | ⚠️ 가능 | pad 추가 필요 |
| Resize | ~15 | ❌ | interpolation |
| Detection Head | ~15 | ❌ | 후처리 |

## 관련

- [[QDQ Direct 파이프라인]] — 전체 파이프라인
- [[Step5 QuantizeRegion 변환 규칙]] — 변환 알고리즘
- [[i8 허용 Op 목록]] — op 분류
