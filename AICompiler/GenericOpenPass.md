- IREE의 정규식은 linalg.generic을 통해 병렬성적인 연산과 반복연산을 표현하고 있다.
- 이것은 NPU에 최적화하기에는 너무 잘게 분해되어 있어, 더 큰 덩어리로 결합한다.
- 큰 덩어리는 NPU에 최적화 된 방식으로 뒤에서 새로이 분할한다.

# 1. Patterns

## 1.1 GenericOpenPattern
- generic을 제거하고, 내부에 있는 basic block의 단일 타입 연산을 tensor 단위 연산으로 바꾼다.

- 예시
```
[Before]
%0 = linalg.generic ins(%in) outs(%out) {
^bb0(%s_in: f32, %s_out: i8):
  %s_1 = "inf_cap.sigmoid"(%s_in) : (f32) -> f32
  linalg.yield %s_1 : f32
}
  
[After]
%0 = "inf_cap.sigmoid"(%in) : (tensor<f32>) -> tensor<f32>
```