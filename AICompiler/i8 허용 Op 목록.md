# i8 허용 Op 목록

**최종 갱신**: 2026-03-30
**파일**: `independent/config/i8_allowed_ops.json`

---

## Op 분류 체계

[[Step5 QuantizeRegion 변환 규칙]]에서 사용하는 3+1 분류:

### compute_ops — i8 연산 변환 대상

| Op | 입력 | 출력 | 설명 |
|---|---|---|---|
| `stablehlo.convolution` | i8 | i32 | i8×i8 convolution |
| `stablehlo.add` | i8 | i8 | element-wise 덧셈 |
| `stablehlo.logistic` | i8 | i8 | sigmoid (NPU LUT) |
| `stablehlo.multiply` | i8 | i8 | element-wise 곱셈 |
| `stablehlo.reduce_window` | i8 | i8 | max pooling 등 |

### transparent_ops — 데이터 전달 (자동 전파)

인접 op이 quant type이면 자동으로 따라감.

| Op | 설명 |
|---|---|
| `stablehlo.broadcast_in_dim` | tensor broadcast |
| `stablehlo.reshape` | shape 변환 |
| `stablehlo.transpose` | 축 순서 변환 |
| `stablehlo.bitcast_convert` | type 재해석 |
| `stablehlo.slice` | tensor 슬라이싱 |
| `stablehlo.convert` | type 변환 |

### never_allowed — f32 필수

| Op | 이유 |
|---|---|
| `divide`, `power` | 비선형 산술 |
| `exponential`, `log` | 초월함수 |
| `sqrt`, `rsqrt` | 제곱근 |
| `sine`, `cosine`, `tanh` | 삼각/쌍곡선 |

### unknown — 목록에 없음 → 경계 유지

| Op | 이유 | 향후 |
|---|---|---|
| `concatenate` | operand scale 불일치 가능 | QuantCleanup에서 별도 처리 |
| `dot_general` | attention에서 양쪽 activation | 경우에 따라 가능 |
| `pad` | pooling padding, quant 호환 필요 | transparent로 이동 가능 |
| `gather` | interpolation index | f32 유지 |
| `sort` | TopK | f32 유지 |
| `reduce` | softmax body | f32 유지 |

## 수정 방법

`i8_allowed_ops.json`을 수정하면 코드 변경 없이 변환 범위 조정 가능.

## 관련

- [[QDQ Direct 파이프라인]] — 이 목록을 사용하는 파이프라인
- [[Step5 QuantizeRegion 변환 규칙]] — 변환 알고리즘
- [[Step5 fp32 잔여 분석]] — 각 카테고리별 잔여 현황
