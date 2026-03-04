- ver0.10 이후로 IREE의 VM bytecode, runtime, HAL Ops를 사용하지 않는 것으로 결정되며, 단일 variant 내부에 IREE관련 식들은 필요없어졌다.
- 단일 variant 내부에 있는 IREE 정의의 IR들을 제거한다.

# 1. Patterns

## 1.1 RemoveCollapsePattern
- tensor.collapse_shape를 제거한다.
- tensor.collapse_shape는 axis를 축약하는 반면, tensor의 의미를 축소시키는 역할을 한다.
- collapse를 제거하여 axis를 축약하지 않고, tensor의 의미를 그대로 살려 다음 계층으로 전달한다.
- collapse의 result가 input으로 사용되는 Ops를 찾아 collapse의 result를 collapse의 input으로 대체한다.
- 예시
```
[Before]
%collapsed = tensor.collapse_shape %3 [[0, 1], [2], [3]] : tensor<1x16x320x320xf32> into tensor<16x320x320xf32>
%5 = "inf_cap.quantize"(%collapsed) <{quant_info = #inf_cap.quant<0.00392156234, -128>}> : (tensor<16x320x320xf32>) -> tensor<16x320x320xi8>

[After]
%5 = "inf_cap.quantize"(%3) <{quant_info = #inf_cap.quant<0.00392156234, -128>}> : (tensor<1x16x320x320xf32>) -> tensor<1x16x320x320xi8>
```

## 1.2 