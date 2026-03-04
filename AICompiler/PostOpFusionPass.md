- OpFusionPass의 뒷처리를 한다. : conv를 묶고 남은 bias나, requantize가 있다. 

# 1. Patterns



## 1.1 Post - Conv

- Conv와 Bias, Requantize는 다른 generic으로 분리되어 있음. 
- Conv와 Bias, Requantize는 따로 관리하기 위해 별도의 operation으로 관리한다.
### 1.1.1 FindConvBiasPattern
- addi를 만나면 Bias로 바꾼다. addi의 경우 Bias가 아닌 add일 수도 있으므로 엄격한 규칙을 적용한다.
- Bias가 아닌 addi 보다 우선순위를 높히기 위해 benefit을 활용한다.
- 규칙 1 : generic의 첫 operation으로 존재하는 addi 에 대해서만 규칙을 적용한다.
- 규칙 2 : addi가 첫번째로 존재하는 generic의 definingOp를 찾아 Conv인 경우만 규칙을 적용한다.
- 규칙 3 : addi의 basic block 및 generic의 affine map을 조합하여 그 shape이 Conv의 result와 일치하는 경우에만 규칙을 적용한다.
- Benefit=5
- 예시
```
=============================================================//
FindConvBias Pattern
=============================================================//
This pattern fuses the 'arith.addi' operation following a convolution
into a single hardware-specific bias operation.
It validates the bias broadcasting pattern via Affine Maps to ensure
the addition occurs across the channel dimension.

[before]
%0 = "inf_cap.conv2d"(%input, %weight) : ... -> tensor<16x160x160xi32
%1 = linalg.generic 
  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, // conv result
    affine_map<(d0, d1, d2) -> (d0)>,         // bias (1D)
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>] // output
      } ins(%0, %bias : ...) {
        ^bb0(%in: i32, %in_bias: i32, %out: i8):
        %2 = arith.addi %in, %in_bias : i32  <-- Target for Fusion
        ...
}
  
[after]
        ^bb0(%in: i32, %in_bias: i32, %out: i8):
        %2 = "inf_cap.bias"(%in, %in_bias) : (i32, i32) -> i32
        ...
```

### 1.1.2 FindRequantPattern
- conv - bias 이후에 오는 requant를 변환한다. 이 requant를 찾아 inf_cap.requant로 바꾼다. 
- 이 패턴은 zeropoint가 0인 패턴으로, 최적화에 의해 사라진 버전이다.
- 규칙 1 : bias 이후에 오는 값이어야 한다.
- 규칙 2 : arith.sitofp -> arith.mulf로 이어지는 패턴을 찾아 requant로 변환한다.
- benefit=5
- 예시
```
[before]
          ^bb0(%in: i32, %in_bias: i32, %out: f32):
            %0 = "inf_cap.conv_bias"(%in, %in_bias) : (i32, i32) -> i32
            %1 = arith.sitofp %0 : i32 to f32
            %2 = arith.mulf %1, %scale : f32

[after]
          ^bb0(%in: i32, %in_bias: i32, %out: f32):
            %0 = "inf_cap.conv_bias"(%in, %in_bias) : (i32, i32) -> i32
            %1 = "inf_cap.requant"(%0, %scale) : (i32, f32) -> f32
```

### 1.1.3 FindRequant2Pattern
- conv-bias 이후에 오는 requant를 변환한다.
- 이 패턴은 zeropoint가 0이 아닌 패턴이다. (TODO : 아직 실제로 requant의 zeropoint가 0이 아닌 버전을 보지 못해서 검증을 하지 못했다. 해당 패턴을 발견하고 실험하여 적용되는지 검증할 필요가 있다.)
- 규칙 1 : bias 이후에 오는 값이어야 한다.
- 규칙 2 : 아래 패턴을 매칭한다.
- benefit=4
- 예시
```
[before]

          %0 = "inf_cap.conv_bias"(%in, %in_bias) : (i32, i32) -> i32
          %1 = arith.sitofp %0 : i32 to f32
          %2 = arith.mulf %1, %scale : f32
          %3 = arith.addf %2, %zp_f32 : f32 

[after]
          %4 = "inf_cap.requant"(%0, %scale, %zp_i32) : (i32, f32, i32) -> i8
```

