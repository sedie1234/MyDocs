- 각종 연산은 arith와 math, linalg등의 Dialect를 이용한 IREE의 정규식으로 식이 펼쳐져 있다. 이를 다시 묶어낸다.

# 1. Patterns

같은 정규식이어도, 최적화 수준에 따라 다르게 표현되어 있으므로 각각의 모든 경우에 대해 변환패턴을 등록한다.

## 1.1 Conv

### 1.1.1 IREEToQConvPattern
- generic을 잡고, yield부터 back tracking
- Asymmetric(subi 존재, ZP≠0)과 Symmetric(subi 없음, ZP=0)을 하나의 패턴에서 모두 처리
- 6D(3×3+ conv, [P,P,P,R,R,R])와 4D(1×1 conv, [P,P,P,R])를 모두 매칭
- 6D: indexing_maps input(d3, d1\*s+d4, d2\*s+d5), weight(d0,d3,d4,d5), output(d0,d1,d2). stride는 affine expr에서 추출.
- 4D: indexing_maps input(d3,d1,d2), weight(d0,d3), output(d0,d1,d2). stride는 항상 1 (kernel 1×1이므로 affine expr에 stride 항 없음). 4D를 매칭하지 않으면 PostOpFusion의 conv_bias 매칭이 연쇄 실패.
- body 패턴 (6D/4D 공통): addi←{out, muli←{subi(extsi, zp), extsi}} (asymmetric) 또는 addi←{out, muli←{extsi, extsi}} (symmetric)
- benefit = 10

```
[변환 전 — Asymmetric 3×3 conv (6D)]
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 * 2 + d4, d2 * 2 + d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%inserted_slice, %cst_0 : tensor<3x642x642xi8>, tensor<16x3x3x3xi8>) outs(%10 : tensor<16x320x320xi32>) {
    ^bb0(%in: i8, %in_10: i8, %out: i32):
      %14 = arith.extsi %in : i8 to i32
      %15 = arith.subi %14, %c-128_i32 : i32
      %16 = arith.extsi %in_10 : i8 to i32
      %17 = arith.muli %15, %16 : i32
      %18 = arith.addi %out, %17 : i32
      linalg.yield %18 : i32
    } -> tensor<16x320x320xi32>
```
```
[변환 후 — 3×3 conv]
%12 = inf_cap.conv2d %inserted_slice, %cst_0 { dilation_h = 1 : i64, dilation_w = 1 : i64, stride_h = 2 : i64, stride_w = 2 : i64, pad_h = 0 : i64, pad_w = 0 : i64, quant_info = #inf_cap.quant<scale = 1.0, zero_point = -128>, layout_hint = #inf_cap.layout<NCHW> } : (tensor<3x642x642xi8>, tensor<16x3x3x3xi8>) -> tensor<16x320x320xi32>
```

```
[변환 전 — Symmetric 3×3 conv (6D)]
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 * 2 + d4, d2 * 2 + d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%inserted_slice, %cst_0 : tensor<3x642x642xi8>, tensor<16x3x3x3xi8>) outs(%10 : tensor<16x320x320xi32>) {
    ^bb0(%in: i8, %in_10: i8, %out: i32):
      %14 = arith.extsi %in : i8 to i32
      %15 = arith.extsi %in_10 : i8 to i32
      %16 = arith.muli %14, %15 : i32
      %17 = arith.addi %out, %16 : i32
      linalg.yield %17 : i32
    } -> tensor<16x320x320xi32>
```

```
[변환 전 — Asymmetric 1×1 conv (4D)]
    %240 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%238, %cst_16 : tensor<32x160x160xi8>, tensor<32x32xi8>) outs(%235 : tensor<32x160x160xi32>) {
    ^bb0(%in: i8, %in_701: i8, %out: i32):
      %770 = arith.extsi %in : i8 to i32
      %771 = arith.subi %770, %c-127_i32 : i32
      %772 = arith.extsi %in_701 : i8 to i32
      %773 = arith.muli %771, %772 : i32
      %774 = arith.addi %out, %773 : i32
      linalg.yield %774 : i32
    } -> tensor<32x160x160xi32>
```
```
[변환 후 — 1×1 conv]
%240 = inf_cap.conv2d %238, %cst_16 { dilation_h = 1 : i64, dilation_w = 1 : i64, stride_h = 1 : i64, stride_w = 1 : i64, pad_h = 0 : i64, pad_w = 0 : i64, quant_info = #inf_cap.quant<scale = 1.0, zero_point = -127>, layout_hint = #inf_cap.layout<NCHW> } : (tensor<32x160x160xi8>, tensor<32x32xi8>) -> tensor<32x160x160xi32>
```

## 1.2 Sigmoid

### 1.2.1 IREEToSigmoidPattern
- benefit = 10
- 아래 flow를 inf_cap.sigmoid로 고친다.
```
[변환 전]
      %27 = arith.negf %26 : f32
      %28 = math.exp %27 : f32
      %29 = arith.addf %28, %cst_6 : f32
      %30 = arith.divf %cst_6, %29 : f32
```
```
[변환 후]
      %30 = inf_cap.sigmoid %29 : f32
```



## 1.3 SiLU
- SiLU는 여기서 구현하지 않는다. IREEToCapPass에서 Sigmoid와 Mul을 찾아 변환한다.


## 1.4 IREEToQBatchMatMulPattern

- `linalg::BatchMatmulOp`를 타겟으로 잡는다.
- benefit = 10
- i8 × i8 → i32 타입인 경우에만 매칭
- indexing_maps를 분석하여 transpose_lhs, transpose_rhs를 판별한 후 `inf_cap.batch_matmul`로 변환한다.
- outs(zero-init 누적 버퍼)는 항상 0 초기화이므로 신규 op에서 생략 → 기존 `linalg.fill`은 DCE로 자동 소멸

### transpose 판별 로직

4차원 루프 `(d0=batch, d1=m, d2=n, d3=k_reduction)` 기준:

| 경우 | 조건 | 결과 |
|---|---|---|
| LHS 전치 | lhsMap.getResult(1) == d3 (reduction dim) | transpose_lhs = 1 |
| LHS 비전치 | lhsMap.getResult(1) == d1 (parallel dim) | transpose_lhs = 0 |
| RHS 전치 | rhsMap.getResult(1) ≠ d3 (reduction dim) | transpose_rhs = 1 |
| RHS 비전치 | rhsMap.getResult(1) == d3 (reduction dim) | transpose_rhs = 0 |

### YOLOv10 Attention 블록 적용 결과

```
[변환 전 — 인스턴스 1: Q·Kᵀ (L1563)]
linalg.batch_matmul indexing_maps = [#map11, #map12, #map4]
    ins(%434, %435 : tensor<2x32x400xi8>, tensor<2x32x400xi8>)
    outs(%437 : tensor<2x400x400xi32>) -> tensor<2x400x400xi32>

[변환 후]
"inf_cap.batch_matmul"(%434, %435)
    <{transpose_lhs = 1 : i64, transpose_rhs = 0 : i64}>
    : (tensor<2x32x400xi8>, tensor<2x32x400xi8>) -> tensor<2x400x400xi32>
```

```
[변환 전 — 인스턴스 2: Attn·V (L1630)]
linalg.batch_matmul indexing_maps = [#map15, #map16, #map4]
    ins(%456, %455 : tensor<2x64x400xi8>, tensor<2x400x400xi8>)
    outs(%458 : tensor<2x64x400xi32>) -> tensor<2x64x400xi32>

[변환 후]
"inf_cap.batch_matmul"(%456, %455)
    <{transpose_lhs = 0 : i64, transpose_rhs = 1 : i64}>
    : (tensor<2x64x400xi8>, tensor<2x400x400xi8>) -> tensor<2x64x400xi32>
```

검증 결과 (yolov10_step3_post.mlir 기준):
- `linalg.batch_matmul` 잔존: 0개
- `inf_cap.batch_matmul` 생성: 2개
- MonoOpFusion 충돌 없음 (OpFusion이 먼저 제거하므로)


## 1.5 IREEToDepthwiseConvPattern

- `linalg::DepthwiseConv2DNhwcHwcOp`를 타겟으로 잡는다.
- benefit = 10
- i8 × i8 타입인 경우에만 매칭
- `op.getStrides()`, `op.getDilations()`에서 stride/dilation 추출
- padding은 0으로 초기화 — VailPadPass에서 이후 처리

```
[변환 전]
linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1>, strides = dense<2>}
    ins(%input : tensor<1x82x82x128xi8>, %weight : tensor<3x3x128xi8>)
    outs(%init : tensor<1x40x40x128xi32>)

[변환 후]
"inf_cap.depthwise_conv2d"(%input, %weight)
    <{dilation_h = 1, dilation_w = 1, stride_h = 2, stride_w = 2,
      pad_h = 0, pad_w = 0}>
    : (tensor<1x82x82x128xi8>, tensor<3x3x128xi8>) -> tensor<1x40x40x128xi32>
```

YOLOv10 검증 결과: `linalg.depthwise_conv_2d_nhwc_hwc` 13개 → `inf_cap.depthwise_conv2d` 13개 ✅

---

## 1.6 IREEToMaxPoolingPattern

- `linalg::PoolingNchwMaxOp`를 타겟으로 잡는다.
- benefit = 10
- kernel_window 텐서(inputs[1])의 shape에서 kernel_h, kernel_w 추출
- `op.getStrides()`에서 stride 추출
- padding은 0으로 초기화 — VailPadPass에서 이후 처리

```
[변환 전]
linalg.pooling_nchw_max {dilations = dense<1>, strides = dense<1>}
    ins(%input : tensor<1x128x24x24xf32>, %kernel_window : tensor<5x5xf32>)
    outs(%init : tensor<1x128x20x20xf32>)

[변환 후]
"inf_cap.pooling_max"(%input)
    <{kernel_h = 5, kernel_w = 5,
      stride_h = 1, stride_w = 1,
      pad_h = 0, pad_w = 0}>
    : (tensor<1x128x24x24xf32>) -> tensor<1x128x20x20xf32>
```

YOLOv10 검증 결과: `linalg.pooling_nchw_max` 3개 → `inf_cap.pooling_max` 3개 ✅

---

## 1.7 IREEToSoftmaxPattern2

- `linalg::SoftmaxOp`를 타겟으로 잡는다.
- benefit = 10
- `op.getInput()` (단수형, `getInputs()` 아님)으로 입력 추출
- `op->getResult(0).getType()` (기반 Op* 포인터)으로 결과 타입 추출
- `op.getDimension()`으로 softmax 차원 추출

```
[변환 전]
linalg.softmax dimension(2)
    ins(%input : tensor<2x400x400xf32>)
    outs(%init : tensor<2x400x400xf32>)

[변환 후]
"inf_cap.softmax"(%input) <{dimension = 2}>
    : (tensor<2x400x400xf32>) -> tensor<2x400x400xf32>
```

> **API 주의**: `linalg::SoftmaxOp`는 `getInputs()` 멤버가 없고 `getInput()` 단수형만 존재.
> 결과 타입은 `op->getResult(0).getType()` 형태로 기반 Op 포인터를 통해 접근.

YOLOv10 검증 결과: `linalg.softmax` 1개 → `inf_cap.softmax` 1개 ✅

---

## 1.8 IREEToMulPattern