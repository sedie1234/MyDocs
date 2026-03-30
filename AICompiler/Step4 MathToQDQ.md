# Step4 MathToQDQ

**Pass**: `--math-to-qdq`
**파일**: `MathToQDQPass.cpp`

---

## 역할

Explicit math로 표현된 Q/DQ 패턴을 StableHLO의 `uniform_quantize` / `uniform_dequantize` op으로 uppering.

## 패턴

### Dequantize (weight/bias)

```mlir
// 변환 전
%0 = convert %c_i8 : (tensor<16x3x3x3xi8>) → tensor<16x3x3x3xf32>
%1 = broadcast_in_dim %scale → tensor<16x3x3x3xf32>
%2 = multiply %0, %1

// 변환 후
%0 = bitcast_convert %c_i8 : (i8) → !quant.uniform<i8:f32, scale>
%1 = uniform_dequantize %0 → f32
```

`convert→multiply` → `bitcast_convert + uniform_dequantize`

### Quantize (activation)

```mlir
// 변환 전
%0 = divide %x, %scale
%1 = round_nearest_even %0
%2 = clamp %min128, %1, %max127
%3 = multiply %2, %scale

// 변환 후
%0 = uniform_quantize %x : (f32) → !quant.uniform<i8:f32, scale>
%1 = uniform_dequantize %0 → f32
```

`divide→round→clamp→multiply` → `uniform_quantize + uniform_dequantize`

### 적용 순서

**Pattern 2 (Q+DQ) 먼저** 적용하여 `clamp→multiply`를 소비.
**Pattern 1 (DQ) 나중에** 적용하여 `convert→multiply`를 소비.
순서 중요: 둘 다 multiply를 최종 op으로 매칭하므로 오매칭 방지.

## 결과 (yolov10n)

| | Step3 | Step4 |
|---|---|---|
| lines | 3334 | **1752** (-47%) |
| uniform_quantize | 0 | **311** |
| uniform_dequantize | 0 | **479** |
| divide | 316 | **5** |
| round | 311 | **0** |
| clamp | 311 | **0** |
| multiply | 555 | **76** |

## 관련

- [[Step3 IdentityElim]] — 이전 단계
- [[Step5 QuantizeRegion]] — 다음 단계
