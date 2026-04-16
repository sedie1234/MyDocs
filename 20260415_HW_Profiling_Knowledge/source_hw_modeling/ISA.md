
# ISA 정의
- ISA와 연동되어 함께 정의되어야 하는 하드웨어의 동작은 여기에서 정의한다.

## 기본 설정 값
- ISA record size : 2048 bit

## 필수 규칙
1. 각 core의 첫 instruction layer의 cmd header는 stop으로 한다. 
2. 각 core의 두번 째 instruction layer의 cmd header는 loop start로 한다.
3. cmd header가 loop jump인 경우, cmd header가 loop start인 곳으로 이동한다. (2번째 layer)
4. cmd header가 loop end인 경우, cmd header가 stop인 곳으로 이동한다. (1번째 layer)



## ISA 구성요소

### 공통 구성요소
- loop_count (32bit) : loop를 몇 회 반복할 지 횟수이다. 무한으로 반복한다면 0으로 한다.

### cmd header (3bit)
- stop
- loop start
- loop jump
- layer start
- layer run
- layer end
- loop end

### cmd header 별 구성요소
- 공통 구성요소는 포함하지 않는다. 

#### cmd header : stop
- 없음

#### cmd header : loop start
- 없음

#### cmd header : loop jump
- 없음

#### cmd header : layer start
- opcode (5bit) + operands (842bit) : 해당 layer에서 사용할 연산의 종류와 weight/bias DMA, shape params, scale/zp 등 layer 공통 정보를 설정한다.
- 상세 구성은 "opcodes definition : cmd header = layer start" 참조.

#### cmd header : layer run
- opcode (5bit) + operands (342bit) : tile 단위 연산을 수행한다. input/output DMA pack을 포함한다.
- layer start에서 설정된 공통 정보를 참조하여 동작한다.
- layer run의 opcode는 layer start의 opcode와 다를 수 있다.
- 상세 구성은 "opcodes definition : cmd header = layer run" 참조.

#### cmd header : layer end
- opcode (5bit) + operands (684bit) : layer 종료 후 후처리 DMA를 수행한다. DMA pack 최대 6개.
- 상세 구성은 "opcodes definition : cmd header = layer end" 참조.

#### cmd header : loop end
- 없음



### opcodes definition : cmd header = layer start

- opcode는 5bit를 사용한다.
- operands는 842bit로 통일한다. (Attention 기준 최대값. 남는 bit는 reserved)
- opcode bit[4] = 0 : 연산기 전용 opcode (Conv, Gemm, Attention 등). weight/bias DMA + params + scale/zp를 포함.
- opcode bit[4] = 1 : Elementwise 계열. bit[3:0]이 sub_opcode. weight 없이 shape만 포함.
- Concat, Split은 별도 opcode (bit[4]=0 범위).
- 각 opcode의 operands에서 미사용 영역은 reserved(0)로 채운다.

#### operands 사용량 요약 (layer start, operands 영역 = 842bit 통일)
| opcode | 이름 | operands 사용 (bit) | reserved (bit) |
|:---|:---|---:|---:|
| 0b00001 | Conv | 606 | 236 |
| 0b00010 | Gemm | 538 | 304 |
| 0b00011 | Attention | 842 | 0 |
| 0b00100 | Concat | 30 | 812 |
| 0b00101 | Split | 46 | 796 |
| 0b00110 | Head | 12 | 830 |
| 0b1xxxx | Elementwise | 236 | 606 |

#### Conv
- opcode 번호 : 0b00001 (1)
- 담당하는 연산의 범위 : i8 convolution + requantize + i8 Activation = i8 output. groups=1이면 standard conv, groups=C_in이면 depthwise conv. (별도 DwConv opcode 없음)
- operands 구성 :
  - weight DMA pack (114bit)
  - bias DMA pack (114bit)
  - conv params (115bit) : 아래 상세 참조
  - scale factors array (32x4=128bit) : input, weight, output, bias
  - zeropoint factors array (32x4=128bit) : input, weight, output, bias
  - activation type (4bit) : 아래 정의 참조
  - quant control (3bit) : 아래 정의 참조
  - reserved : 나머지
  - (operands 사용: 606bit)

- conv params 상세 (115bit) :
  - C_in (16bit) : 입력 채널 수. 최대 65535.
  - C_out (16bit) : 출력 채널 수. 최대 65535.
  - H_in (16bit) : 입력 높이. 최대 65535.
  - W_in (16bit) : 입력 너비. 최대 65535.
  - Kh (4bit) : kernel 높이. 최대 15.
  - Kw (4bit) : kernel 너비. 최대 15.
  - stride_h (4bit) : 수직 stride. 최대 15.
  - stride_w (4bit) : 수평 stride. 최대 15.
  - pad_top (3bit) : 상단 padding. 최대 7.
  - pad_bottom (3bit) : 하단 padding. 최대 7.
  - pad_left (3bit) : 좌측 padding. 최대 7.
  - pad_right (3bit) : 우측 padding. 최대 7.
  - dilation_h (3bit) : 수직 dilation. 최대 7. (0이면 1로 해석)
  - dilation_w (3bit) : 수평 dilation. 최대 7. (0이면 1로 해석)
  - groups (16bit) : group convolution 수. 1=standard, C_in=depthwise.
  - tensor_format (1bit) : 0=NHWC, 1=NCHW.
  - H_out, W_out 미포함. 계산: H_out = (H_in + pad_top + pad_bottom - dilation_h*(Kh-1) - 1) / stride_h + 1

##### quant control (3bit)
- per_channel_enable (1bit) : 1이면 output channel별 scale/shift 테이블 참조. 0이면 per-tensor.
- accum_mode (2bit) : 0=i32(기본), 1=i16, 2~3=reserved.

##### activation type (4bit)
- 0 : None
- 1 : Relu
- 2 : Silu
- 3 : Gelu
- 4 : Sigmoid
- 5~15 : reserved

#### Gemm
- opcode 번호 : 0b00010 (2)
- 담당하는 연산의 범위 : i8 GEMM = i8 output
- operands 구성 :
  - weight DMA pack (114bit)
  - bias DMA pack (114bit)
  - gemm params (51bit) : 아래 상세 참조
  - scale factors array (32x4=128bit) : input, weight, output, bias
  - zeropoint factors array (32x4=128bit) : input, weight, output, bias
  - quant control (3bit) : Conv와 동일
  - reserved : 나머지
  - (operands 사용: 538bit)

- gemm params 상세 (51bit) :
  - M (16bit) : 출력 행 수. A[M,K] × B[K,N] = C[M,N]
  - K (16bit) : reduction 차원.
  - N (16bit) : 출력 열 수.
  - tensor_format (1bit) : 0=NHWC, 1=NCHW.
  - accum_mode (2bit) : 0=i32(기본), 1=i16, 2~3=reserved.

#### Attention
- opcode 번호 : 0b00011 (3)
- 담당하는 연산의 범위 : Q·K^T → softmax → Attn·V (전체 attention)
- operands 구성 :
  - Wq DMA pack (114bit) : Q projection weight
  - Wk DMA pack (114bit) : K projection weight
  - Wv DMA pack (114bit) : V projection weight
  - Wo DMA pack (114bit) : output projection weight (미사용 시 dummy)
  - bias_q DMA pack (114bit) : Q bias (미사용 시 dummy)
  - bias_kv DMA pack (114bit) : K/V bias (미사용 시 dummy)
  - attention params (27bit) : 아래 상세 참조
  - scale factors array (32x2=64bit) : input, output
  - zeropoint factors array (32x2=64bit) : input, output
  - quant control (3bit) : Conv와 동일
  - reserved : 나머지
  - (operands 사용: 842bit)

- attention params 상세 (27bit) :
  - C (16bit) : 입력 채널(embedding) 수.
  - num_heads (4bit) : attention head 수. 최대 15.
  - head_dim (4bit) : head당 차원. 최대 15.
  - tensor_format (1bit) : 0=NHWC, 1=NCHW.
  - accum_mode (2bit) : 0=i32(기본).

#### Concat
- opcode 번호 : 0b00100 (4)
- 담당하는 연산의 범위 : 채널 방향 concatenate
- operands 구성 :
  - concat params (30bit) : 아래 상세 참조
  - reserved : 나머지
  - (operands 사용: 30bit)

- concat params 상세 (30bit) :
  - C_total (16bit) : concat 후 총 출력 채널 수.
  - H (12bit) : 높이.
  - tensor_format (1bit) : 0=NHWC, 1=NCHW.
  - reserved (1bit).
- 비고 : weight/bias/scale/zp 없음. 개별 입력 크기는 layer run의 DMA pack에서 결정.

#### Split
- opcode 번호 : 0b00101 (5)
- 담당하는 연산의 범위 : 하나의 입력 텐서를 지정 축에서 2개의 sub-region으로 분할. **순수 metadata 연산**이며 데이터 이동(DMA)이 없다. layer run에서 각 출력 sub-region의 (buffer, addr, size, ref_count)를 RefCountTable에 등록만 수행한다.
- operands 구성 :
  - split params (46bit) : 아래 상세 참조
  - reserved : 나머지
  - (operands 사용: 46bit)

- split params 상세 (46bit) :
  - C_total (16bit) : 입력 총 채널 수.
  - H (12bit) : 높이.
  - W (12bit) : 너비.
  - num_outputs (2bit) : 출력 개수. 현재 0=2-output 고정, 나머지는 reserved.
  - tensor_format (1bit) : 0=NHWC, 1=NCHW.
  - split_axis (2bit) : 0=C, 1=H, 2=W, 3=N.
  - reserved (1bit).
- 동작 의미 :
  - layer run 실행 시 DMA bus는 사용하지 않는다 (0 cycle data transfer).
  - 출력 sub-region 2개의 (buffer_index, addr, size, ref_count)를 RefCountTable에 등록하여 downstream 소비자의 write-lock을 설정한다.
  - 입력 region의 ref_count는 Split을 소비자로 취급하여 감소시킨다.
  - 실제 메모리의 데이터는 건드리지 않는다. 두 출력 sub-region은 입력 region의 서로 다른 주소 범위에 대한 "레이블"이 된다.
- 컴파일러 계약 :
  - split_axis는 반드시 현재 tensor_format에서 **연속 메모리 범위를 자르는 축**이어야 한다. 예를 들어 NHWC에서 split_axis=C는 인정되지 않는다 (메모리상 C는 가장 안쪽이라 sub-region이 연속이 아님). 컴파일러는 이 경우 기존 Transpose(0b10110) opcode를 Split 앞뒤에 삽입하여 `Transpose(NHWC→NCHW) → Split(NCHW, axis=C) → Transpose(NCHW→NHWC)×2` 형태로 풀어낸다. 연속성이 보장되는 조합 (NCHW-C, NHWC-H 등) 에서는 Transpose 없이 Split 단독 사용.
  - 3개 이상으로 분할해야 하는 경우 컴파일러가 2-output Split의 연쇄로 합성한다.
- 비고 : weight/bias/scale/zp 없음. 출력 sub-region의 크기는 layer run의 region descriptor에서 결정.

#### Head
- opcode 번호 : 0b00110 (6)
- 담당하는 연산의 범위 : 모델-specific post-processing 블록 (NMS / 디코드 / 기타). 선택한 `head_subop` 에 따라 하드웨어는 해당 헤드의 사양에 맞춰 입력 텐서들을 소비하고 최종 결과를 하나(또는 소수) 의 출력 텐서로 생성한다.
- operands 구성 :
  - head params (12bit) : 아래 상세 참조
  - reserved : 나머지 (각 head 가 자기 확장 필드로 사용 가능)
  - (operands 기본 사용: 12bit)

- head params 상세 (12bit) :
  - head_subop (4bit) : 어느 헤드 패밀리인지.
    - 0 : `yolov10n-detect` (YOLOv10n 의 DFL + box/cls + NMS decode)
    - 1~15 : reserved (향후 classification / segmentation / 기타 모델용)
  - num_inputs  (4bit) : 이 헤드에 공급되는 입력 텐서 개수 (최대 15).
  - num_outputs (4bit) : 이 헤드가 생성하는 출력 텐서 개수 (최대 15).
- 컴파일러 계약 :
  - 컴파일러는 모델의 post-processing 구간 (예: YOLOv10n 의 /model.23 내 NMS 관련 ops) 전체를 `Head` opcode 하나로 번들링한다.
  - Head 의 input 은 "cut point 이전에 생성되고 cut point 이후에 읽히는 모든 텐서" (pre-NMS 출력).
  - Head 의 output 은 최종 디코드 결과 (YOLOv10n 의 경우 `(1, 300, 6)` 형태의 detection 리스트 등).
  - 대안 모드로 "NMS host offload" 를 지원한다. 이 경우 Head opcode 는 생성하지 않고 pre-NMS 출력을 모델의 최종 output 으로 선언한다. 컴파일러의 `--nms-mode={onchip,offload}` 플래그로 선택.
- 비고 :
  - Head 의 compute 본체는 head_subop 별로 고정된 하드웨어 구현이거나, 런타임 라이브러리로 에뮬레이션된다. 컴파일러는 구현 세부를 몰라도 됨.
  - YOLOv10n 의 경우 inputs = 6 (3 scale × cls+box), outputs = 1.

#### 0b00111 ~ 0b01111 (7~15)
- reserved

#### Elementwise 계열
- opcode 번호 : 0b10000 ~ 0b11111 (16~31). bit[4]=1, bit[3:0]=sub_opcode.
- 담당하는 연산의 범위 : weight가 필요 없는 연산들의 공통 layer start.
- operands 구성 :
  - shape params (43bit) : 아래 상세 참조
  - tensor_format (1bit) : 0=NHWC, 1=NCHW.
  - scale factors array (32x3=96bit) : input1, input2, output에 대한 scale factor. (1-input 연산은 input2를 0으로)
  - zeropoint factors array (32x3=96bit) : input1, input2, output에 대한 zeropoint factor.
  - reserved : 나머지
  - (operands 사용: 43+1+96+96 = 236bit)
- 비고 : weight/bias DMA 없음. scale/zp는 requant가 필요한 연산(Add, Mul 등)에서 사용.

##### sub_opcode = bit[3:0] 정의
- 0b0000 (0) : Add — element-wise add + requant. **layer run opcode = 0b10000 (16)**
- 0b0001 (1) : Mul — element-wise mul + requant. **layer run opcode = 0b10001 (17)**
- 0b0010 (2) : Sigmoid — standalone sigmoid. **layer run opcode = 0b10010 (18)**
- 0b0011 (3) : MaxPool — max pooling. **layer run opcode = 0b10011 (19)**
- 0b0100 (4) : Resize — nearest neighbor upsample. **layer run opcode = 0b10100 (20)**
- 0b0101 (5) : TopK — 상위 K개 추출. **layer run opcode = 0b10101 (21)**
- 0b0110 (6) : Transpose — 축 순서 변경. **layer run opcode = 0b10110 (22)**
- 0b0111 (7) : Gather — index 기반 추출. **layer run opcode = 0b10111 (23)**
- 0b1000 (8) : Copy — 데이터 복사 (예비). **layer run opcode = 0b11000 (24)**
- 0b1100 (12) : Softmax — 지정 축에 대한 softmax. i8 입력 → i8 출력 (확률 분포). **layer run opcode = 0b11100 (28)**
  - aux (3bit) 재해석 : softmax_axis (2bit) + exp_mode (1bit, 0=LUT)
  - axis 내부는 한 번의 LAYER_RUN 안에서 원자적으로 처리된다. 다른 축이 UB 에 들어가지 않는 경우 컴파일러가 non-softmax 축을 타일 단위로 쪼개 multi-RUN 으로 emit 한다 (softmax 축은 절대 쪼개지 않음).
  - 내부 구현 : exp 는 256-entry i8 LUT, normalize 는 reduce_sum 후 divide.
- 0b1001~0b1011, 0b1101~0b1111 (9~11, 13~15) : reserved

##### shape params (43bit)
- C (16bit) : 채널 수.
- H (12bit) : 높이. 최대 4095.
- W (12bit) : 너비. 최대 4095.
- aux (3bit) : sub_opcode별 보조 파라미터.
  - MaxPool: kernel size (최대 7)
  - Resize: scale factor (2=2x, 4=4x 등)
  - Transpose: perm hint (축 순서 인코딩)
  - 기타: reserved (0)


### opcodes definition : cmd header = layer run

- opcode는 5bit를 사용한다.
- operands는 342bit로 통일한다. (Concat 기준 최대값. 남는 bit는 reserved)
- layer run의 operands에는 tile 단위로 변하는 정보만 포함한다. shape, scale 등 layer 공통 정보는 layer start에서 설정된 값을 사용한다.
- opcode 번호는 layer start와 동일한 체계를 따른다. Elementwise 계열은 layer start의 base(0b10000) + sub_opcode로 조합된 값을 직접 사용한다.
- **layer run의 opcode는 layer start의 opcode와 다를 수 있다.** 예: layer start가 Conv(0b00001)이더라도 layer run에 Add(0b10000)나 Sigmoid(0b10010) 등 Elementwise 계열 opcode를 사용할 수 있다. 하나의 layer start-end 구간 안에 서로 다른 opcode의 layer run이 혼재할 수 있다.

#### operands 사용량 요약 (layer run, operands 영역 = 342bit 통일)
| opcode | 이름 | input 수 | output 수 | operands 사용 (bit) | reserved (bit) |
|:---|:---|---:|---:|---:|---:|
| 0b00001 | Conv | 1 | 1 | 228 | 114 |
| 0b00010 | Gemm | 1 | 1 | 228 | 114 |
| 0b00011 | Attention | 1 | 1 | 228 | 114 |
| 0b00100 | Concat | 2 | 1 | 342 | 0 |
| 0b00101 | Split | 1 | 2 | 208 | 134 |
| 0b00110 | Head | variable | 1~ | variable | variable |
| 0b10000 | Add | 2 | 1 | 342 | 0 |
| 0b10001 | Mul | 2 | 1 | 342 | 0 |
| 0b10010 | Sigmoid | 1 | 1 | 228 | 114 |
| 0b10011 | MaxPool | 1 | 1 | 228 | 114 |
| 0b10100 | Resize | 1 | 1 | 228 | 114 |
| 0b10101 | TopK | 1 | 2 | 342 | 0 |
| 0b10110 | Transpose | 1 | 1 | 228 | 114 |
| 0b10111 | Gather | 2 | 1 | 342 | 0 |
| 0b11000 | Copy | 1 | 1 | 228 | 114 |

#### Conv
- opcode 번호 : 0b00001 (1)
- operands 구성 :
  - input DMA pack (114bit) : input tile의 DMA
  - output DMA pack (114bit) : output tile의 DMA
  - reserved : 나머지
  - (사용: 228bit)

#### Gemm
- opcode 번호 : 0b00010 (2)
- operands 구성 :
  - input DMA pack (114bit) : input의 DMA
  - output DMA pack (114bit) : output의 DMA
  - reserved : 나머지
  - (사용: 228bit)

#### Attention
- opcode 번호 : 0b00011 (3)
- operands 구성 :
  - input DMA pack (114bit) : input의 DMA
  - output DMA pack (114bit) : output의 DMA
  - reserved : 나머지
  - (사용: 228bit)

#### Concat
- opcode 번호 : 0b00100 (4)
- operands 구성 :
  - input1 DMA pack (114bit) : 첫 번째 입력의 DMA
  - input2 DMA pack (114bit) : 두 번째 입력의 DMA
  - output DMA pack (114bit) : 출력의 DMA
  - reserved : 나머지
  - (사용: 342bit)
- 비고 : 3개 이상 concat 시 layer run을 여러 번 사용하여 2개씩 순차 합산.

#### Split
- opcode 번호 : 0b00101 (5)
- 1-input, 2-output. **DMA pack을 사용하지 않는다.** 대신 입력/출력 region descriptor로 RefCountTable을 갱신한다.
- operands 구성 :
  - out0 region descriptor (72bit) : 아래 상세 참조
  - out1 region descriptor (72bit) : 아래 상세 참조
  - input region descriptor (64bit) : 아래 상세 참조
  - reserved : 나머지
  - (사용: 208bit)

- out_k region descriptor 상세 (72bit) :
  - buffer_index (5bit) : 출력 sub-region이 놓일 buffer (SM=24, Pcore UB=0~11 등. DMA pack의 buffer index 정의와 동일).
  - addr (32bit) : 출력 sub-region의 시작 주소 = 입력 주소 + axis offset.
  - size (27bit) : 출력 sub-region의 크기(byte).
  - ref_count (8bit) : 이 sub-region의 초기 read reference count. 이후 downstream 소비자가 읽을 때마다 감소.

- input region descriptor 상세 (64bit) :
  - buffer_index (5bit) : 입력 region이 놓인 buffer.
  - addr (32bit) : 입력 region의 시작 주소.
  - size (27bit) : 입력 region의 크기(byte).

- 동작 :
  - DMA bus 사용 없음. out0/out1 region의 (buffer, addr, size, ref_count)를 RefCountTable에 등록한다.
  - 입력 region의 ref_count를 감소시켜 write-lock을 해제한다 (Split이 입력의 유일한 소비자로 취급됨).
  - 두 출력 region의 addr 범위는 입력 region 안에 완전히 포함되어야 하며, 상호 겹치지 않아야 한다 (컴파일러 책임).

- 비고 : 연속 메모리 축 split만 허용. 비연속 축(NHWC의 C 등) split은 컴파일러가 Transpose(0b10110)로 감싸서 변환한 뒤 Split을 emit한다. 3개 이상 출력은 Split 연쇄로 합성.

#### 0b00110 ~ 0b01111 (6~15)
- reserved

#### Elementwise 계열
- opcode 번호 : 0b10000 ~ 0b11111 (16~31). bit[4]=1, bit[3:0]=sub_opcode.
- layer start에서 설정한 sub_opcode와 동일한 값을 layer run opcode에 직접 사용한다.
- opcode별로 input/output 개수와 해석이 다르므로 아래에 개별 정의한다.

##### Add — opcode 0b10000 (16)
- 2-input, 1-output. output은 별도 주소.
- operands 구성 :
  - input1 DMA pack (114bit) : 첫 번째 입력
  - input2 DMA pack (114bit) : 두 번째 입력
  - output DMA pack (114bit) : 출력 (input1/input2와 다른 주소)
  - (사용: 342bit)

##### Mul — opcode 0b10001 (17)
- 2-input, 1-output. output은 별도 주소.
- operands 구성 :
  - input1 DMA pack (114bit) : 첫 번째 입력
  - input2 DMA pack (114bit) : 두 번째 입력
  - output DMA pack (114bit) : 출력 (input1/input2와 다른 주소)
  - (사용: 342bit)

##### Sigmoid — opcode 0b10010 (18)
- 1-input, 1-output.
- operands 구성 :
  - input DMA pack (114bit) : 입력
  - output DMA pack (114bit) : 출력
  - reserved : 나머지
  - (사용: 228bit)

##### MaxPool — opcode 0b10011 (19)
- 1-input, 1-output.
- operands 구성 :
  - input DMA pack (114bit) : 입력
  - output DMA pack (114bit) : 출력
  - reserved : 나머지
  - (사용: 228bit)

##### Resize — opcode 0b10100 (20)
- 1-input, 1-output.
- operands 구성 :
  - input DMA pack (114bit) : 입력
  - output DMA pack (114bit) : 출력
  - reserved : 나머지
  - (사용: 228bit)

##### TopK — opcode 0b10101 (21)
- 1-input, 2-output (scores + indices).
- operands 구성 :
  - input DMA pack (114bit) : scores 입력
  - output_scores DMA pack (114bit) : 상위 K개 scores 출력
  - output_indices DMA pack (114bit) : 상위 K개 indices 출력
  - (사용: 342bit)

##### Transpose — opcode 0b10110 (22)
- 1-input, 1-output.
- operands 구성 :
  - input DMA pack (114bit) : 입력
  - output DMA pack (114bit) : 출력
  - reserved : 나머지
  - (사용: 228bit)

##### Gather — opcode 0b10111 (23)
- 2-input (data + indices), 1-output.
- operands 구성 :
  - data DMA pack (114bit) : 원본 데이터
  - indices DMA pack (114bit) : index 배열
  - output DMA pack (114bit) : 추출 결과
  - (사용: 342bit)

##### Copy — opcode 0b11000 (24)
- 1-input, 1-output.
- operands 구성 :
  - input DMA pack (114bit) : source
  - output DMA pack (114bit) : destination
  - reserved : 나머지
  - (사용: 228bit)

##### Softmax — opcode 0b11100 (28)
- 1-input, 1-output. softmax_axis 방향으로 원자적.
- operands 구성 :
  - input DMA pack (114bit) : 현재 RUN 이 처리할 입력 타일의 DMA. 타일 경계는 softmax 축을 **절대** 자르지 않는다.
  - output DMA pack (114bit) : 해당 타일의 출력.
  - reserved : 나머지
  - (사용: 228bit)
- 비고 : shape/axis/scale 은 LAYER_START 에서 세팅됨. RUN 은 타일별로 여러 번 호출될 수 있고, 각 호출에서 softmax 축 전체를 원자적으로 처리해야 한다.

##### Head — opcode 0b00110 (6)
- 1~15 inputs, 1~15 outputs. head_subop 별로 의미가 다르다.
- operands 구성 (컴파일러 책임) :
  - 여러 개의 DMA pack 으로 입력 텐서들의 SM 주소 / 크기 / ref_count 를 전달. 컴파일러는 필요한 수만큼 LAYER_RUN 레코드를 반복 emit 해도 되고, 3-pack 으로 묶어도 된다.
  - 출력 DMA pack 은 마지막 LAYER_RUN (또는 LAYER_END) 에서 최종 결과의 SM 주소를 지정.
  - 하드웨어는 LAYER_START 에 기록된 (head_subop, num_inputs, num_outputs) 를 보고 내부적으로 해당 head 의 고정 routine 을 실행한다.
- 비고 : head 자체의 compute 경로는 하드웨어 구현 또는 런타임 에뮬레이션이 제공. 컴파일러는 입력/출력 주소 전달만 책임진다.

##### 0b11001 ~ 0b11011 (25~27), 0b11101 ~ 0b11111 (29~31)
- reserved


### opcodes definition : cmd header = layer end
- opcode는 5bit를 사용한다.
- operands는 684bit로 통일한다. (DMA pack ×6 기준. 남는 bit는 reserved)
- opcode 번호는 layer start와 동일한 것을 사용한다.
- layer end의 operands는 opcode별로 필요에 따라 정의한다. 필요없는 경우 DMA pack을 dummy로 채운다.
- 용도 : layer 사이 sync, layer run만으로 처리하지 못하는 후처리 DMA 등.
- operands 구성은 모든 opcode에서 동일하다.

#### operands 사용량 요약 (layer end, operands 영역 = 684bit 통일)
| opcode | 이름 | operands 사용 (bit) | reserved (bit) |
|:---|:---|---:|---:|
| 모든 opcode 공통 | DMA pack ×6 | 684 | 0 |

#### 공통 operands 구성 (684bit)
- DMA pack 0 (114bit) : 미사용 시 transfer type의 buffer index를 dummy(31)로 설정
- DMA pack 1 (114bit) : 동일
- DMA pack 2 (114bit) : 동일
- DMA pack 3 (114bit) : 동일
- DMA pack 4 (114bit) : 동일
- DMA pack 5 (114bit) : 동일
- (operands 사용: 114×6 = 684bit)

## DMA
- DMA는 하나의 pack으로 정의한다. 이 pack은 cmd header와 opcode에 따라 하나의 instruction layer에 여러 번 들어갈 수 있다. 
- DMA pack (114bit) : | transfer type (10bit) | src addr (32bit) | dst addr (32bit) | data size (32bit) | read reference count (8bit) |
- read reference count : 이 데이터가 몇 번 읽혀야 사용이 끝나는 지를 기입한다. dst addr및  data size 내용을 read reference count register에 기입하고, write lock 역할을 수행한다. 데이터가 한 번 참조될 때마다 count를 1감소시킨다.

### transfer type 정의
- transfer type은 데이터를 어떤 buffer에서 어떤 buffer로 이동시킬 지를 정의한다.
- 따라오는 src addr, dst addr은 해당 buffer의 고유주소가 된다.
- transfer type (10bit) = | src buffer(5bit) | dst buffer(5bit) |
#### buffer index (5bit)
- 0 ~ 11 : p core의 번호로, 각 pcore의 Unified Buffer를 의미한다.
- 12 ~ 23 : a core의 번호로, 각각 0 ~ 11번째의 acore의 Unified Buffer를 의미한다.
- 24 : Shared Memory임을 의미한다.
- 25 : 자기 자신을 의미한다. 자신의 번호를 표기해도 되고, 이 번호를 사용하여 자기 buffer를 참조하여도 된다.
- 26 : pairing된 acore를 의미한다. acore의 번호를 표기해도 되고, 이 번호를 사용해도 된다.
- 27 : 인접해있는 이전 core를 의미한다. 마찬가지로 core의 번호를 표기해도 되고, 이 번호를 사용해도 된다.
- 28 : 인접해있는 다음 core를 의미한다. 마찬가지로 core의 번호를 표기해도 되고, 이 번호를 사용해도 된다.
- 29 ~ 30 : reserved
- 31 : dummy = DMA 하지 않음. 공간을 채우는 용도

## sync
- read reference count와 하드웨어 자체 구현으로 sync를 맞춘다. 

## stream
- reserved : 현재 버전은 사용하지 않음. 

## ISA overview

### record 구조
- record size : 2048 bit (고정)
- | cmd header (3bit) | loop count (32bit) | opcode (5bit) | operands | reserved/padding |
- operands 크기는 cmd header에 따라 다름:
  - layer start : 842bit (+ record 내 나머지 padding)
  - layer run : 342bit (+ record 내 나머지 padding)
  - layer end : 684bit (+ record 내 나머지 padding)

### opcode 번호 전체표

| opcode (5bit) | 이름 | layer start | layer run |
|:---|:---|:---|:---|
| 0b00000 (0) | reserved | - | - |
| 0b00001 (1) | Conv | weight/bias DMA + conv params + scales/zps + act + quant (606bit) | input/output DMA (228bit) |
| 0b00010 (2) | Gemm | weight/bias DMA + gemm params + scales/zps + quant (538bit) | input/output DMA (228bit) |
| 0b00011 (3) | Attention | Wq/Wk/Wv/Wo/bias DMA + attn params + scales/zps + quant (842bit) | input/output DMA (228bit) |
| 0b00100 (4) | Concat | concat params (30bit) | input1/input2/output DMA (342bit) |
| 0b00101 (5) | Split | split params (46bit) | out0/out1/input region descriptors (208bit) |
| 0b00110 (6) | Head | head params (12bit) | input DMA packs (per-head variable) |
| 0b00111~0b01111 (7~15) | reserved | - | - |
| 0b10000 (16) | Add | Elementwise shape+scales (236bit) | input/output DMA (228bit) |
| 0b10001 (17) | Mul | Elementwise shape+scales (236bit) | input/output DMA (228bit) |
| 0b10010 (18) | Sigmoid | Elementwise shape+scales (236bit) | input/output DMA (228bit) |
| 0b10011 (19) | MaxPool | Elementwise shape+scales (236bit) | input/output DMA (228bit) |
| 0b10100 (20) | Resize | Elementwise shape+scales (236bit) | input/output DMA (228bit) |
| 0b10101 (21) | TopK | Elementwise shape+scales (236bit) | input/output DMA (228bit) |
| 0b10110 (22) | Transpose | Elementwise shape+scales (236bit) | input/output DMA (228bit) |
| 0b10111 (23) | Gather | Elementwise shape+scales (236bit) | input/output DMA (228bit) |
| 0b11000 (24) | Copy | Elementwise shape+scales (236bit) | input/output DMA (228bit) |
| 0b11001~0b11011 (25~27) | reserved | - | - |
| 0b11100 (28) | Softmax | Elementwise shape+scales+axis+exp_mode (238bit) | input/output DMA (228bit) |
| 0b11101~0b11111 (29~31) | reserved | - | - |