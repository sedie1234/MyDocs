# Step3 IdentityElim

**Pass**: `--identity-elim --canonicalize --cse`
**파일**: `IdentityElimPass.cpp`

---

## 역할

int8 symmetric quantization의 zero_point=0에 의한 항등 연산 제거.

## 패턴

| 패턴 | 변환 | 이유 |
|---|---|---|
| `subtract(x, broadcast(0))` | → `x` | `(x - zp) * scale`, zp=0 |
| `add(x, broadcast(0))` | → `x` | `round(x/scale) + zp`, zp=0 |
| `multiply(x, broadcast(1))` | → `x` | scale=1인 경우 |

## 동작

```mlir
// 변환 전 (DequantizeLinear: (x - zp) * scale, zp=0)
%0 = broadcast_in_dim %cst_zero → tensor<16x3x3x3xf32>  // broadcast(0)
%1 = subtract %convert_result, %0                         // x - 0
%2 = broadcast_in_dim %cst_scale → tensor<16x3x3x3xf32>
%3 = multiply %1, %2                                      // x * scale

// 변환 후
%2 = broadcast_in_dim %cst_scale → tensor<16x3x3x3xf32>
%3 = multiply %convert_result, %2                          // subtract 제거됨
```

## 결과 (yolov10n)

| | Step2 | Step3 |
|---|---|---|
| lines | 4215 | **3334** (-21%) |
| subtract | 482 | **3** (-479) |
| add | 410 | **99** (-311) |

## 관련

- [[Step2 Canonicalize + CSE]] — 이전 단계
- [[Step4 MathToQDQ]] — 다음 단계
