- 계층 개념 : 우리 하드웨어가 계산할 수 있는 연산들을 모음
- 하드웨어의 opcodes와 1대1로 매칭되는 경우에는 바로 변환할 수 있으므로 엔트리에 넣음
- 하드웨어의 opcodes의 조합으로 바꿀 수 있으면 엔트리에 넣음
	- 예를 들면, 우리 하드웨어가 conv는 수행못하지만, matmul은 수행할 수 있다고 할 경우, conv는 matmul로 바꿀 수 있기 때문에 우선 엔트리에 넣는다.
- 연산의 조합을 하드웨어가 실행할 수 있는 경우 엔트리에 넣음
	- 예를 들면, add-neg-exp-add-div-mul은 silu이므로 silu로 변환한 뒤, 엔트리에 넣음
- 최대한 다양한 케이스를 하드웨어로 내려주기 위해 이 계층에서는 Interface를 설정하지 않음

# 1. Dialect 정의
- name : inf_cap
- cppNamespace : ::infetron_v2::cap
- hasConstantMaterializer = 1  :  필요한 경우 상수 생성
- useDefaultAttributePrinterParser = 1  :   별도의 parser 구현 x



# 2. Attributes
## 2.1 Data Layout Type (구현 완료)
- 뒤에 이어지는 계층의 Ops들이 참고할 수 있도록 String 및 Enum으로써만 정의
- Enums
```
def INF_CAP_LayoutNHWC : I32EnumAttrCase<"NHWC", 0>;
def INF_CAP_LayoutNCHW : I32EnumAttrCase<"NCHW", 1>;
```

## 2.2 Data Layout Attribute (구현 완료)
- INF_CAP_DataLayoutAttr : 2.1의 Enum으로 어떤데이터 타입인지 지정
```
def INF_CAP_DataLayoutAttr : EnumAttr<INF_CAPIR_Dialect, INF_CAP_DataLayoutEnum, "layout">;
```

## 2.3 Quantization Attribute (구현 완료)
- Op가 quantization요소를 가지고 있으면 사용하기 위해 정의
- scale과 zeropoint를 포함한다.
- INF_CAP_QuantizationAttr로 정의
- mnemonic : quant



# 3. OpBase

## 3.1 INF_CAP_Op (구현 완료)
- INF_CAP 계층의 Ops들이 상속받아 사용할 수 있는 OpBase.
- layout hint 및 metadata를 arguments로 가짐

## 3.2 INF_CAP_FusedOp (구현 완료)
- INF_CAP_Op를 상속받아 정의, 다른 Ops들이 상속받아 사용할 수 있는 OpBase.
- Fusing된 연산들을 관리하기 위한 클래스
- 필요할 지는 아직 알 수 없으나 예비차원에서 생성




# 4. Ops (INF_CAP Dialect)

## 4.1 INF_CAP_Conv2DOp (구현 완료)

- **개요**: 2D Convolution 연산. 하드웨어의 MM(Matrix Multiplication) 가속 유닛에 매핑되는 핵심 Op.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTensor
        
    - `weights`: AnyTensor
        
    - `dilation_h / dilation_w`: I64Attr
        
    - `stride_h / stride_w`: I64Attr
        
    - `pad_h / pad_w`: I64Attr
        
    - `layout_hint`: Optional<INF_CAP_DataLayoutAttr>
        
    - `quant_info`: Optional<INF_CAP_QuantizationAttr>
        
- **Results**:
    
    - `output`: AnyTensor
        

## 4.2 INF_CAP_SiLUOp (구현 완료)

- **개요**: FP32/FP16 도메인에서 수행되는 SiLU(Swish) 활성화 함수.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTensor
        
    - `segment_k`: I64Attr (PWL 근사를 위한 구간 수, 기본값 8)
        
    - `quant_info`: Optional<INF_CAP_QuantizationAttr>
        
- **Results**:
    
    - `output`: AnyTensor
        

## 4.3 INF_CAP_QuantOp (구현 완료)

- **개요**: Float 타입을 Quantized 정수 타입으로 변환하는 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTypeOf<\[AnyTensor, AnyFloat\]>
        
    - `quant_info`: INF_CAP_QuantizationAttr
        
- **Results**:
    
    - `output`: AnyTypeOf<\[AnyTensor, AnyInteger\]>
        

## 4.4 INF_CAP_DequantOp (구현 완료)

- **개요**: Quantized 정수 타입을 Float 타입으로 복원하는 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTypeOf<\[AnyTensor, AnyInteger\]>
        
    - `quant_info`: INF_CAP_QuantizationAttr
        
- **Results**:
    
    - `output`: AnyTypeOf<\[AnyTensor, AnyFloat\]>
        

## 4.5 INF_CAP_MulOp (구현 완료)

- **개요**: 요소별(Element-wise) 곱셈 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `lhs / rhs`: AnyTypeOf<\[AnyTensor, AnyFloat, AnyInteger\]>
        
    - `quant_info`: Optional<INF_CAP_QuantizationAttr>
        
- **Results**:
    
    - `output`: AnyType (입력 타입과 동일)
        

## 4.6 INF_CAP_SigmoidOp (구현 완료)

- **개요**: 기본 Sigmoid 활성화 함수. (부동소수점 도메인용)
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTypeOf<\[AnyTensor, AnyFloat\]>
        
    - `quant_info`: Optional<INF_CAP_QuantizationAttr>
        
- **Results**:
    
    - `output`: AnyType
        

## 4.7 INF_CAP_ConvBiasOp (구현 완료)

- **개요**: Convolution 결과에 Bias를 더하는 복합 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTensor (Conv 결과물)
        
    - `bias`: AnyTensor (Bias 벡터)
        
    - `quant_info`: Optional<INF_CAP_QuantizationAttr>
        
- **Results**:
    
    - `output`: AnyTensor
        

## 4.8 INF_CAP_RequantOp (구현 완료)

- **개요**: 중간 연산 결과(주로 i32)를 다른 스케일의 양자화 도메인으로 재조정하는 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyType
        
    - `quant`: INF_CAP_QuantizationAttr
        
- **Results**:
    
    - `output`: AnyType
        

## 4.9 INF_CAP_AddOp (구현 완료)

- **개요**: 요소별(Element-wise) 덧셈 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `lhs / rhs`: AnyTypeOf<\[AnyTensor, AnyFloat, AnyInteger\]>
        
    - `quant_info`: Optional<INF_CAP_QuantizationAttr>
        
- **Results**:
    
    - `output`: AnyType
        

## 4.10 INF_CAP_QSigmoidOp (구현 완료)

- **개요**: **\[Hardware Optimized\]** 하드웨어 LUT(Look-Up Table)를 사용하는 양자화된 Sigmoid 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTypeOf<\[AnyTensor, AnyInteger\]>
        
    - `in_quant`: INF_CAP_QuantizationAttr (입력 도메인 해석용)
        
    - `out_quant`: INF_CAP_QuantizationAttr (출력 도메인 결정용)
        
    - `lut_size`: I64Attr (LUT 크기 힌트, 기본값 256)
        
- **Results**:
    
    - `output`: AnyTypeOf<\[AnyTensor, AnyInteger\]>
        

## 4.11 INF_CAP_QMulOp (구현 완료)

- **개요**: **\[Hardware Optimized\]** 양자화 도메인 내에서의 요소별 곱셈. 서로 다른 스케일 정렬(Rescale) 로직 포함.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `lhs / rhs`: AnyTypeOf<\[AnyTensor, AnyInteger\]>
        
    - `lhs_quant / rhs_quant`: INF_CAP_QuantizationAttr
        
    - `out_quant`: INF_CAP_QuantizationAttr
        
- **Results**:
    
    - `output`: AnyTypeOf<\[AnyTensor, AnyInteger\]>
        

## 4.12 INF_CAP_QSiLUOp (구현 완료)

- **개요**: **\[Hardware Optimized\]** 양자화된 SiLU ($x \times \text{sigmoid}(x)$) 연산.
    
- **Side Effect**: Pure (없음)
    
- **Arguments**:
    
    - `input`: AnyTypeOf<\[AnyTensor, AnyInteger, AnyFloat\]>
        
	- `segment_k`: I64Attr (PWL 근사를 위한 구간 수, 기본값 8)
        
    - `in_quant`: INF_CAP_QuantizationAttr
        
    - `out_quant`: INF_CAP_QuantizationAttr
        
- **Results**:
    
    - `output`: AnyType

