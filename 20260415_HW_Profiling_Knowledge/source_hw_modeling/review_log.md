# HW Modeling 문서 검토 기록

> 작성일: 2026-04-10  
> 대상 문서: hw_modeling/, ISA.md, hw_architecture.md, hw_operating_flow.md, hw_spec.md, compiler_output_definition.md

---

## 1. 해결된 모순/문제

| # | 유형 | 위치 | 문제 | 해결 |
|:---|:---|:---|:---|:---|
| M1 | 모순 | hw_spec vs feasibility | SM 크기 불일치 (4MB vs 3MB) | hw_modeling 문서 내에는 불일치 없음. 별도 feasibility 문서 범위. 문제 아님. |
| M2 | 모순 | hw_architecture.md Acore | "Conv, Gemm 연산 가능" vs "depthwise만 가능" 두 문장 충돌 | Acore = depthwise conv만 가능으로 확정. 문서 수정 완료. |
| M3 | 모순 | ISA record size | record 내 padding이 많음 | 2048bit = 1 record 고정. 의도된 설계. |
| M4 | 모순 | hw_operating_flow / compiler_output | input reference count 동작 주체 불명확 | 최초: 런타임이 .json 값으로 register 기록. 이후: instruction layer DMA pack의 ref count로 HW 자율 동작. 문서 수정 완료. |
| I1 | 부족 | ISA | Acore/Pcore opcode 할당 미정의 | Acore는 stop-loop_start-loop_jump-loop_end만. 실제 연산은 Pcore만 사용 (현재 단계). |
| I2 | 부족 | ISA | weight shape 정보 없음 | conv params의 C_in, C_out, Kh, Kw, groups로 유추 가능. 별도 필드 불필요. |
| I3 | 부족 | ISA layer run | Add/Mul in-place output 주소 모호 | Elementwise 계열 layer run을 opcode별로 세부 정의. Add/Mul = 3 DMA pack (input1+input2+output). 문서 수정 완료. |
| I4 | 부족 | compiler_output | .ihnn meta data 내용 미정 | meta data는 binary로 이어져 있고 header에 시작주소/크기 표기. 현재 상태 충분. |
| I5 | 부족 | compiler_output .json | SM feature map 영역 미정의 | FM 주소는 컴파일러가 SM 여유 공간에서 계산하여 records에 직접 포함. .json 별도 정의 불필요. |
| I6 | 부족 | hw_modeling | 실험용 .ihnn/.json 제작 방법 | AI가 문서를 보고 직접 생성. 부족한 정보는 질문으로 해결. |
| I7 | 부족 | hw_architecture | DMA 엔진 객체 미정의 | 시뮬레이터에서는 객체 간 데이터 복사로 단순 구현. |
| I8 | 부족 | hw_spec | MAC 병렬도 미확정 | 블랙박스 처리. 컴파일러가 C→H 순서로 잘라서 제공, core가 계산. |
| I9 | 부족 | ISA | stop/loop record의 opcode 필드 처리 | ISA/flow 문서 정의대로만 동작. opcode 필드는 don't care. |
| I10 | 부족 | hw_architecture | UB double buffering 분할 정책 | 컴파일러가 주소 분할 관리. HW는 현재 record 연산 + 다음 record DMA 병렬 수행 (파이프라이닝). |
| A1 | 모호 | hw_architecture Conv chain | 연속 layer pipeline 정의 미반영 | 현재 단계 고려 안 함. HW가 알아서 한다고 들음. 기록만 유지. |
| A2 | 모호 | ISA Elementwise Add/Mul | in-place output 주소 불명확 | I3에서 해소. 3 DMA pack으로 분리. |
| A3 | 모호 | hw_operating_flow | loop jump 위치 | 모든 layer 끝에 1개만 존재 (records 끝부분). |
| A4 | 모호 | compiler_output .json | Queue 0x00 = stop과 저장 시작 동일 | 의도된 설계. 0번 주소는 stop 고정. |
| A5 | 모호 | hw_modeling 결과물 | "런타임 스크립트" 언어 | Python 또는 C++. 더 적합한 것으로 선택. |
| A6 | 모호 | hw_architecture | DMA 버스 arbitration 정책 없음 | 필요 시 round robin을 default로 사용. |
| A7 | 모호 | ISA cmd header | layer start 정의에 opcode+operands 누락 | cmd header별 구성요소에 opcode(5bit)+operands(842/342/684bit) 명시. 문서 수정 완료. |

---

## 2. 수정된 수치 오류

| 위치 | 기존 | 수정 | 이유 |
|:---|:---|:---|:---|
| ISA.md conv params | 113bit | **114bit** | C_in(16)+C_out(16)+H_in(16)+W_in(16)+Kh(4)+Kw(4)+stride_h(4)+stride_w(4)+pad(3×4=12)+dilation(3×2=6)+groups(16)+tensor_format(1) = 114 |
| ISA.md Conv operands 합계 | 604bit | **605bit** | conv params 113→114로 +1 |
| ISA.md Conv layer start reserved | 238bit | **237bit** | 842-605=237 |
| ISA.md overview Conv | 604bit | **605bit** | 위와 동일 |

---

## 3. 미해결 — 향후 확인/반영 필요

### 3.1 용어 통일 필요

| 용어 A | 용어 B | 위치 | 비고 |
|:---|:---|:---|:---|
| instruction layer | instruction record | hw_operating_flow vs compiler_output_definition | 동일 개념인지 확인 필요. "instruction layer"는 ISA record 1개를 의미하는 것으로 추정되나 명시적 정의 없음. |

### 3.2 loop_count=0 동작 규칙 — 해결

- **결정**: loop_count=0은 무한 반복으로 정의. loop_count=0일 때 loop counter 비교를 skip하고 무조건 loop start로 jump.
- 런타임 종료 시 NPU 전체를 강제 정지시키는 기능이 필요. (모든 core의 PC를 0으로 설정하는 halt 기능)
- 코드 구현에 반영 완료 (Simulator.cpp의 executeLoopJump).

### 3.3 Elementwise Add/Mul에 requant scale/zp 전달 방법 — 해결

- **결정**: Elementwise layer start의 operands에 scale/zp 필드를 추가.
  - scale factors array (32x3=96bit): input1, input2, output
  - zeropoint factors array (32x3=96bit): input1, input2, output
  - 1-input 연산은 input2를 0으로 설정.
- ISA.md 반영 완료 (Elementwise layer start operands: 44→236bit).
- 코드 구현 반영 필요 (ISARecord 파서에 scale/zp 파싱 추가).

### 3.4 Conv chain (A1) — 현재 고려 안 함

- 동기화마저도 하드웨어가 알아서 한다고 전해들음.
- 현재는 컴파일러의 데이터 의존성에 따른 record 구축과 DMA ref count sync에만 의존.
- 나중에 바뀔 수 있으나 현재 단계에서는 고려하지 않음.
- 관련 의문 사항은 `docs/MyDocs/20260408_Conv_Chain_아키텍처_의문사항.md`에 정리되어 있음.

### 3.5 Attention opcode의 실제 동작 상세 — 현재 블랙박스

- Attention은 하나의 opcode로 통째로 돌림. 내부 동작(MatMul+Softmax 분해)은 자세히 알려져 있지 않음.
- ISA에서는 하나의 opcode로 묶었으므로 HW 내부 구현에 위임.
- 시뮬레이터에서도 블랙박스로 처리.

### 3.6 hw_spec.md 연산 시간 예시의 전제

- MAC 병렬도 256 가정, vector processor 16-lane 가정은 추정치
- 시뮬레이터에서는 블랙박스로 처리하므로 이 수치는 참고용
- JSON config로 교체 가능하도록 설계할 것 (hw_modeling.md 지시)

---

## 4. 문서 간 참조 관계

```
hw_modeling.md (총괄 지시서)
  ├── hw_architecture.md (HW 구성요소, 연결, 동작 특이사항)
  ├── hw_operating_flow.md (phase 0~7 동작 흐름)
  ├── hw_spec.md (스펙 수치, clock, 연산 시간)
  ├── ISA.md (ISA record 구조, opcode 정의, DMA)
  └── compiler_output_definition.md (.ihnn, .json 포맷)
```

- hw_spec.md → ISA.md 참조 (ISA Record Size)
- hw_operating_flow.md → ISA.md 참조 (layer start-run-end 동작)
- hw_operating_flow.md → compiler_output_definition.md 참조 (.ihnn, .json)
- hw_architecture.md → ISA.md 참조 (opcode, cmd header)
- hw_architecture.md → hw_spec.md 참조 (size, clock)
