- quantize 모델을 사용하는 경우, QLinear로 시작하는 quantized operation을 IREE는 지원하지 않는다. 
- quantize 모델을 사용하기 위해 qdq로 quantize를 실행하고, 컴파일러가 q - dq와 operation을 합성해, quantized operation으로 만들어 낸다. 
- 1. 먼저 dequantize를 찾고 backward tracking으로 연결된 operation을 찾아낸다. 
- 2. 그 앞에 붙은 quantize들을 모아서 새로운 arguments들을 만든다. 
- 3. 이 arguments를 넣는 quantized operation을 만들어 낸다.
- 요약 : dequant + operation = quantized operation // quant + arguments = quantized arguments

# 1. Patterns

## 1.1 QdqToQopPattern
- **개요** : Q-Dq 사이의 Op를 QuantOp로 변경한다. (Op이름은 그대로이고, Attribute를 추가하는 것이 대부분)
- **Trigger : QuantOp**
- 1. Quant에서부터 back tracking방식으로 Op를 발견하고 QuantOp를 생성, Op의 설정들을 QuantOp로 옮긴다.
- 2. Quant의 output을 QuantOp의 output으로 연결
- 3. Back tracking으로 찾은 Dequant의 input을 QuantOp로 연결
- 4. Op 삭제
- 5. 이렇게 하면 Dequant와 Quant가 그래프에서 분리되기 때문에 Canonicalization에서 제거가 될 것이라고 보지만(DCE), 확인 결과 지워지지 않으면 지우는 패턴을 만들어야 함.
  (여기서 지우는 것은 안된다. Dequant의 output이 다른 노드에서도 활용될 수 있기 때문에, 다른 패턴을 만들어 지워야 한다.)


![QdqToQop](https://imgs.hwan123.synology.me:19999/AICompiler/QdqToQop.png)

### 1.1.1 일반 Ops
- 일반적인 Ops들은 Attributes로 quantization 변수를 넣을 수 있게 해뒀기 때문에 새로운 QuantOp를 만들지 않고, Attributes를 추가하고, input과 output을 변경해주면 된다. 
- 이 경우에 해당하는 Ops들에 대해 하나의 Pattern으로 구현할 수 있다. 

#### 1.1.1.1 FindQSigmoidPattern
- dequant - sigmoid - quant를 찾아, sigmoid를 qsigmoid로 바꾼다. 
- 이 때, dequant와 quant는 다른 곳에서 사용할 수도 있으므로 제거하지 않고, input과 output만 옮겨준다.
- mlir의 canonicalizer를 통해 결과값이 사용되지 않는 ops는 자동으로 삭제될 것.
- 1. backward tracking으로 dequant - sigmoid - quant 패턴을 찾는다.
- 2. sigmoid를 qsigmoid로 바꾼다
- 3. qsigmoid의 input을 dequant의 input으로 바꾸고, output을 quant의 output으로 바꾼다.
- 예시
```
[Before]
%i8_in = ... : i8
%f32_1 = "inf_cap.dequantize"(%i8_in) <{quant_info = #in_q}> : (i8) -> f32
%f32_2 = "inf_cap.sigmoid"(%f32_1) : (f32) -> f32
%i8_out = "inf_cap.quantize"(%f32_2) <{quant_info = #out_q}> : (f32) -> i8

[After]
%i8_in = ... : i8
%i8_out = "inf_cap.qsigmoid"(%i8_in) <{in_quant = #in_q, out_quant = #out_q}> : (i8) -> i8
```


#### 1.1.1.2 FindQMulPattern
- dequant - mul - quant를 찾아 mul을 qmul로 바꾼다.
- sigmoid와 비슷한 규칙적용
- 예시
```
[before]
 %lhs_f = "inf_cap.dequantize"(%lhs_i8) <{quant_info = #lhs_q}> : (i8) -> f32
 %rhs_f = "inf_cap.dequantize"(%rhs_i8) <{quant_info = #rhs_q}> : (i8) -> f32
 %mul_f = "inf_cap.mul"(%lhs_f, %rhs_f) : (f32, f32) -> f32
 %q_out = "inf_cap.quantize"(%mul_f) <{quant_info = #out_q}> : (f32) -> i8

[after]
 %q_out = "inf_cap.qmul"(%lhs_i8, %rhs_i8) <{
	 lhs_quant = #lhs_q,
	 rhs_quant = #rhs_q,
	 out_quant = #out_q
 }> : (i8, i8) -> i8
```


#### 1.1.1.3 FindQSiLUPattern
- conv에 연결된 qsigmoid와 qmul을 찾으면 qsilu로 변환한다.
- 현재는 conv가 conv2d - bias - requant - quantize로 나타나도록 했으므로 
- 예시
```
[Before]
%x = ... // Any source (quantize, requant, etc.)
%qsig = "inf_cap.qsigmoid"(%x) <{in_q, sig_out_q}> : (...) -> f32
%qsilu = "inf_cap.qmul"(%x, %qsig) <{in_q, sig_out_q, final_out_q}> : (..., f32) -> i8

[After]
%x = ...
%qsilu = "inf_cap.qsilu"(%x) <{in_q = #in_q, out_q = #final_out_q}> : (...) -> i8
```
### 1.1.2 특별한 Ops
- 특별한 Ops의 경우에는 본래의 취지대로, 새로운 Ops를 만들고 input, output, attributes를 연결한 후, 기존 Ops를 삭제한다.
- 이 방식을 채택할 경우, Ops 1개당 1개의 Pattern을 추가해야 한다.
- 아래에는 해당하는 Ops에 대해 남긴다.
- List
	- TODO : Ops List