# HW Profiling Knowledge Package

작성일: 2026-04-15
대상: Infetron-V2 NPU 하드웨어 프로파일링 — ISA, 아키텍처, 동작 flow, 실험 방법

## 목적

이 디렉토리는 Infetron-V2 NPU 의 **하드웨어 모델링 & 프로파일링 체계** 를 팀원들과 공유하기 위한 지식 패키지다. 세 가지를 한 번에 제공:

1. **원본 지시 문서** (`source_hw_modeling/`) — ISA / 아키텍처 / 동작 flow / 컴파일러 output 의 raw 정의. 변경 시 이 쪽만 업데이트
2. **정리된 지식 문서** (`01_overview.md` ~ `05_experiment_methodology.md`) — 팀원이 맥락을 이해하고 실험·분석에 바로 활용할 수 있도록 재구성한 것
3. **FAQ + troubleshooting** — 자주 나오는 질문, 실제 실험에서 겪은 함정

## 내용 구성

```
20260415_HW_Profiling_Knowledge/
├── README.md                        # (이 파일) 개요와 탐색 안내
├── source_hw_modeling/              # 원본 지시 문서 복제
│   ├── ISA.md
│   ├── hw_architecture.md
│   ├── hw_operating_flow.md
│   ├── hw_spec.md
│   ├── hw_modeling.md
│   ├── compiler_output_definition.md
│   └── review_log.md
├── 01_overview.md                   # 30초 개요: NPU 는 무엇이고 어떻게 돌아가는가
├── 02_hw_architecture.md            # 12+12 core, SM/UB/Queue, DMA, register
├── 03_isa_structure.md              # 2048bit record, opcode, DMA pack
├── 04_operation_flow.md             # Runtime phase 0~7, weight stationary, loop
├── 05_experiment_methodology.md     # 가상 컴파일러 → 시뮬레이터 실험 루프
└── FAQ.md                           # 자주 나오는 질문
```

## 빠른 시작

- **NPU 구조부터 궁금하다** → `01_overview.md` → `02_hw_architecture.md`
- **ISA 를 확장해야 한다** → `03_isa_structure.md` + `source_hw_modeling/ISA.md`
- **하드웨어가 어떻게 실행되는지 이해해야 한다** → `04_operation_flow.md`
- **실험 (.ihnn 생성 → 시뮬레이터) 흐름을 돌려야 한다** → `05_experiment_methodology.md`
- **문제가 있다** → `FAQ.md`

## 규칙

- **원본이 우선**: 지식 문서와 `source_hw_modeling/` 이 충돌하면 원본이 정답. 지식 문서는 설명·컨텍스트 제공용
- **원본 변경 시**: `source_hw_modeling/` 의 파일을 업데이트하고 (실제 변경 위치는 `docs/agent_command/hw_modeling/`), 해당 내용이 지식 문서에 요약된 부분도 같이 갱신
- **버전**: 이 디렉토리는 2026-04-15 시점의 스냅샷. ISA/아키텍처 변경 시 새 날짜로 복제본을 만들어 이력 추적

## 용어 간단 정리

| 용어 | 의미 |
|---|---|
| Pcore | 주 연산 코어 (Conv, Gemm, Attention 등 모든 연산 가능) |
| Acore | Depthwise conv 전용 보조 코어 (+ activation 결합) |
| Cluster | Pcore + Acore pair × 2 (n=2) 묶음. 총 6 개 cluster, 합계 12 + 12 core |
| SM | Shared Memory. 4 MB. 모든 core 가 접근 |
| UB | Unified Buffer. core 내부, 256 KB. 연산기 직접 연결 |
| Queue | core 당 128 KB ISA record 저장소. PC 로 순차 실행 |
| .ihnn | 컴파일러 output 바이너리 (instruction records + tensor blobs) |
| .json | 런타임용 메타 (메모리 매핑, queue 주소 등) |
| ISA record | 2048 bit 고정 크기 명령 단위 (cmd_header + opcode + operands) |
| DMA pack | 114 bit, linear DMA 단위 (src/dst buffer + addr + size + ref_count) |

## 관련 세션 / 산출물

- 가상 컴파일러 구현: `hw-profiling/scripts/onnx_to_ihnn.py`
- 하드웨어 모델링 코드: `hw-profiling/src/hw_model/`
- 독립 컴파일러 참조: `pubH/independent/` (별도 세션)
- 실험 산출물: `docs/agent_results/virtual_compiler_experiment/artifacts/`
