
- Quantize와 Dequantize는 arith와 math Dialect를 이용한 IREE의 정규식으로 식이 펼쳐져 있다. 이를 QuantOp와 DequantOp로 다시 묶어낸다.

# 1. Patterns

같은 정규식이어도, 최적화 수준에 따라 다르게 표현되어 있으므로 각각의 모든 경우에 대해 변환패턴을 등록한다.

## 1.1 Quant1Pattern
- back tracking 방식으로 찾는다.
- 안전장치로, 각 Op마다 user가 1개인지 검사한다.
- 아래 식을 QuantOp로 변환한다.
```
[변경 전]
      %1 = arith.divf %in, %scale : f32
      %2 = math.roundeven %1 : f32
      %3 = arith.addf %2, %zeropoint : f32
      %4 = arith.maximumf %3, %min : f32
      %5 = arith.minimumf %4, %max : f32
      %6 = arith.fptoui %5 : f32 to i8
```
```
[변경 후]
      %6 = inf_cap.quantize %in : f32 to i8
```

## 1.2 Dequant1Pattern
- back tracking 방식으로 찾는다.
- 안전장치로, 각 Op마다 user가 1개인지 검사한다.
- 아래 식을 DequantOp로 변환한다.
```
[변경 전]
      %1 = arith.extui %in : i8 to i32
      %2 = arith.subi %1, %zeropoint : i32
      %3 = arith.sitofp %2 : i32 to f32
      %4 = arith.mulf %3, %scale : f32
```
```
[변경 후]
      %4 = inf_cap.dequantize %in : f32 to i8
```
## 1.3 Dequant2Pattern
- back tracking 방식으로 찾는다.
- 안전장치로, 각 Op마다 user가 1개인지 검사한다.
- zeropoint가 0인 경우, 최적화가 적용되어 아래와 같이 나타날 수 있다.
- 아래 식을 DequantOp로 변환한다.
```
[변경 전]
      %1 = arith.extui %in : i8 to i32
      %2 = arith.sitofp %1 : i32 to f32
      %3 = arith.mulf %2, %scale : f32
```
```
[변경 후]
      %3 = inf_cap.dequantize %in : f32 to i8
```

## 1.4 AddZeropointFusionPattern
- quantize 다음에 add가 오는 경우, zeropoint에 반영하여 하나의 quantize로 바꾼다.
```
[before]
      %12 = "inf_cap.quantize"(%in) <{quant_info = #inf_cap.quant<0.0039215646699999997, 0>}> : (f32) -> i8
      %13 = arith.addi %12, %c-128_i8 : i8

[after]
%12 = "inf_cap.quantize"(%in) <{quant_info = #inf_cap.quant<0.0039215646699999997, %c-128_i8>}> : (f32) -> i8
```
## 1.5 SubZeropointFusionPattern
- quantize 다음에 sub가 온느 경우, zeropoint에 반영하여 하나의 quantize로 바꾼다. 