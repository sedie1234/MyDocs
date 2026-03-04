- 각종 연산은 arith와 math, linalg등의 Dialect를 이용한 IREE의 정규식으로 식이 펼쳐져 있다. 이를 다시 묶어낸다.

# 1. Patterns

같은 정규식이어도, 최적화 수준에 따라 다르게 표현되어 있으므로 각각의 모든 경우에 대해 변환패턴을 등록한다.

## 1.1 Conv

### 1.1.1 IREEToUnsymetricQConvPattern
- generic을 잡고, yield부터 back tracking
- IREEToSymetricQConvPattern과 결합 = IREEToQConvPattern
- benefit = 10

```
[변환 전]
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
[변환 후]
%12 = inf_cap.conv2d %inserted_slice, %cst_0 { dilation_h = 1 : i64, dilation_w = 1 : i64, stride_h = 2 : i64, // indexing_map의 'd1 * 2'에서 추출 stride_w = 2 : i64, // indexing_map의 'd2 * 2'에서 추출 pad_h = 0 : i64, pad_w = 0 : i64, quant_info = #inf_cap.quant<scale = 1.0, zero_point = -128>, // subi에서 추출 layout_hint = #inf_cap.layout<NCHW> // 필요시 추가 } : (tensor<3x642x642xi8>, tensor<16x3x3x3xi8>) -> tensor<16x320x320xi32>
```

### 1.1.2 IREEToSymetricQConvPattern
- generic을 잡고, yield부터 back tracking
- IREEToUnsymetricQConvPattern과 결합 = IREEToQConvPattern
- benefit = 10
```
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 * 2 + d4, d2 * 2 + d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%inserted_slice, %cst_0 : tensor<3x642x642xi8>, tensor<16x3x3x3xi8>) outs(%10 : tensor<16x320x320xi32>) {
    ^bb0(%in: i8, %in_10: i8, %out: i32):
      %14 = arith.extsi %in : i8 to i32
      %15 = arith.extsi %in_10 : i8 to i32
      %16 = arith.muli %14, %15 : i32
      %17 = arith.addi %out, %16 : i32
      linalg.yield %17 : i32
    } -> tensor<16x320x320xi32>
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


## 1.4 IREEToMulPattern