# .ihnn
- 모든 데이터의 바이너리 결과물이다.
- meta data와 instrunction records를 포함한다.
- 구성 요소 : header + 12개의 pcore queue에 각각 기록될 instruction records + 12개의 acore queue에 각각 기록될 instruction records + 12개의 acore에 eflash로 기록될 tensor들 (weight) + Shared Memory에 기록될 Tensor들 (주로 weights)
- header는 파일 내에서 각 데이터들의 위치를 알려주는 navigator 역할을 수행한다.

## 데이터 표현 원칙
모든 tensor binary (weight / bias / input / output / 중간 activation) 는
**i8 (1 byte/element)** 으로 .ihnn 에 기록된다. scale (f32) / zero_point
(i32) 는 ISA record 의 layer_start 내 고정 필드로만 전달되며 별도
binary section 을 차지하지 않는다. 자세한 규칙은 `hw_spec.md` 의
"전역 원칙 — 내부 tensor datapath" 항목 참조.

## 구조
- header (392 bytes) = 
  | {Pcore Instruction Queue Address (4bytes) + Pcore Instruction Queue Size (4bytes)} x12
  | reserved(8x4=32bytes)
  | {Acore Instruction Queue Address (4bytes) + Acore Instruction Queue Size (4bytes)} x12
  | reserved(8x4=32bytes)
  | {Acore Tensor Address (4bytes) + Acore Tensor Size (4bytes)} x12
  | reserved(8x4=32bytes)
  | Sahred memory Pcore Tensor Address (4bytes) + Shared memory Pcore Tensor Size (4bytes)
- meta data 

# .json
- .ihnn에서의 각 바이너리의 의미를 포함한다.
- 실제 하드웨어에 바이너리를 매핑할 정보를 포함한다.
- input과 output이 저장될 위치를 포함한다.

## 구조
- memory_configuration : memory를 어떻게 나누어 사용할 지 정보가 들어있다. 
    - shared memory : shared memory를 어떻게 나누어 사용할 지 정보가 들어있다.
        - metadata : weight를 포함한 각종 meta data가 저장된 위치로 하드웨어 입장에서는 readonly이다. metadata는 하나의 serialized binary로 존재한다. 부득이하게 떨어져야 하는 경우에는 metadata항목을 하나 더 만든다. 
            - address : metadata의 시작 주소를 의미한다.
            - size : metatdata를 위해 할당한 공간의 크기를 의미한다.
        - output : output을 저장할 위치의 주소다.
            - address
            - size
        - input : input을 저장할 위치의 주소다. 
            - address
            - size
        - forbidden area : 또 다른 중요한 데이터를 둘 수도 있으므로, 예비용으로 둔다.
        - reserved
    - eflash : 초기에 acore에는 weight를 올려두어야 한다. 그 정보를 코어별로 가지고 있다.
        - acore 0
            - address
            - size
        - acore 1
            - address
            - size
        - acore 2
            - address
            - size
        - acore 3
            - address
            - size
        ...
        - acore 11
            - address
            - size

- queue_configuration : core 별 queue buffer에서 queue binary가 저장되어야 할 위치 정보. 값이 없다면 default로 0x00번지에 저장한다.
    - pcore 0 
        - address
        - size
    - pcore 1
        - address
        - size
    ...
    - pcore 11
        - address
        - size
    - acore 0
        - address
        - size
    - acore 1
        - address
        - size
    ...
    - acore 11
        - address
        - size

- input reference count : input이 몇 번 참조되어야 해당 공간의 lock을 풀 수 있는지 정보