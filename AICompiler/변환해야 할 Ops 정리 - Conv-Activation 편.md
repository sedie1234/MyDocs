# 1. 모델 개요

- yolo를 타겟하기 전, yolo를 가장 많이 구축하고 있는 Conv-Activation을 타겟으로 변환을 시도해 봄.
- onnx에서 Conv-Activation은 Conv-Sigmoid-Mul로 나타남

# 2. 추가로 입력하는 옵션

- **--iree-global-opt-experimental-disable-conv-generalization** : convolution이 루프로 풀리는 것을 방지

# 3. Ops List - 1차

## 3.1 **arith.constant** 
- c++객체 : arith::ConstantOp
- ins : -
- outs : -
- 설명 : 상수 메타데이터
## 3.2 linalg.fill
- c++객체 : linalg::FillOp
- ins
	- value : 채울 값
	- buffer : 초기화 할 버퍼
- outs : 
- 설명 : 버퍼를 특정 값으로 초기화 (c언어 memset과 동일)

## 3.3 linalg.conv_2d_nchw_fchw
- c++객체 : linalg::Conv2DNchwFchwOp
- ins
	- input : NCHW Layout
	- Kernel : FCHW Layout
- outs : 
	- resutl_tensors : tensor
- attributes
	- strides - ::mlir::DenseIntElementsAttr  :  64bit unsigned int
	- dilations - ::mlir::DenseIntElementsAttr  :  64bit unsigned int
- 설명 : input은 NCHW, Kernel은 FCHW인 conv2d연산

## 3.4 linalg.generic
- c++객체 : linalg::GenericOp
- ins : load할 메모리
- outs : store할 메모리
- attributes
	- indexing_maps : load / store 할 메모리 접근순서 및 방식
	- iterator_types : 연산 방식 (parallel 등)
- 설명 : linalg.generic 하위의 데이터 접근 방법을 표시하여 접근한 데이터로 basic block에 정의된 연산을 수행

## 3.5 arith.addf
- c++객체 : arith::AddFOp
- ins
	- lhs : float 64, 32, bf16, vector, tensor
	- rhs : float 64, 32, bf16, vector, tensor
- outs
	- result : float
- 설명 : 두 float의 덧셈

## 3.6 arith.negf
- c++객체 : arith::NegFOp
- ins
	- in : f64, 32, 8, vector, tensor
- outs
	- result : float
- 설명 : 입력값의 negation을 반환 (동일 타입으로)


## 3.7 math.exp
- c++객체 : math::ExpOp
- ins
	- in : float like
- outs
	- result : float like
- attributes
	- fastmath  :  ::mlir::arith::FastMathFlagsAttr, fast math 플래그
- 설명 : base e의 exponential을 계산


## 3.8 arith.divf
- c++객체 : arith::DivFOp
- ins
	- lhs : float
	- rhs : float
- outs
	- result : float
- attributes
	- fastmath  :  ::mlir::arith::FastMathFlagsAttr, fast math 플래그
- 설명 : float인 lhs를 rhs로 나눈 값

## 3.9 arith.mulf
- c++객체 : arith::MulFOp
- ins
	- lhs : float
	- rhs : float
- outs
	- result : float
- 설명 : float인 lhs와 rhs의 곱

## 3.10 linalg.yield
- c++객체 : linalg::YieldOp
- ins : any value
- 설명 : basic block과 같은 지역의 반환값을 설정







## 3.10 arith.addf
- c++객체 : 
- ins
- outs
- 설명 : 


