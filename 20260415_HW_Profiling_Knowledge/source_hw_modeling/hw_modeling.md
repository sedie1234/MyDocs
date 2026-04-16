# HW modeling 제작 지시서
- HW model을 제작하는 지시서다.
- 상세 구현 정보는 ../hw_modeling/ 에 모두 정의되어 있다. 
- c++로 작성한다.
- hw_spec에 대한 부분은 자주 바뀔 수 있으므로, 그 때마다 새로 코딩하지 않게 json으로 설정할 수 있도록 한다.
- 구축 위치 : AICompiler 최상위의 `hw-profiling/` 디렉토리에 구축한다. (경로: `/home/keti/workspace/AICompiler/hw-profiling/`)
- git 저장소를 새로 생성하여 코드 관리를 한다.

## 목적
- 하드웨어 스펙이 완전하지 않은 상태에서 하드웨어 스펙이나 구조를 임의로 정하여 컴파일러 및 런타임과 연동하여 동작을 점검한다.
- 하드웨어의 미세한 스펙을 점검한다.
- 하드웨어 전체 아키텍쳐를 점검한다.
- ISA 구조를 점검한다.

# HW 제작 가이드
- HW의 각 컴포넌트를 객체로서 두고 제작한다.
- 각각을 어디까지 추상화할 지는 상세 문서에 자세히 기입되어 있다.
- 각 컴포넌트는 정의된대로 동작해야한다.
- 실제 시뮬레이션 시, 시간별로 데이터를 처리할 수 있어야 한다.
- 각 컴포넌트들은 클럭에 따라 상태를 처리할 수 있어야 한다. : 시뮬레이터와 연계가 되어야 하므로 시뮬레이터 제작 가이드를 적극적으로 참고한다.

# 컴파일러와 런타임 연동
- 컴파일러와 런타임은 실제로 하드웨어와 연동시킬 프로그램을 사용한다.
- 컴파일러를 통해 나오는 바이너리는 실제로 하드웨어에 올릴 바이너리이다.
- 런타임은 function symbol은 동일하지만 그 내부는 modeling된 HW와 연동하여 동작하게 설계된다.

# 시뮬레이터 제작 가이드
- 시뮬레이터는 런타임의 동작 이후에는 하드웨어만 동작시킨다.
- 하드웨어 동작 phase로 넘어가면 가상의 클럭에 따라 데이터를 처리한다. 
- 디버깅을 위해 각 시간별로 데이터가 어떻게 구성되었으며 어떻게 이동했었는지 기록이 되어야 한다.
- 클럭이 여러 번 지나도 모든 상태가 바뀌지 않는다면 한 번에 여러 클럭을 넘길 수 있다. 

# 구현해야 할 객체
- hw_architecture.md에 따른다.

# 객체별, opcode별 clock 설정
- hw_spec에 따른다.

# 결과물
- 아래 구성으로 된 shared library를 결과물로 한다.
1. c++ 하드웨어 모델 : c++ 객체들의 조합
2. 시뮬레이터 : 런타임의 한 공간에 사용할 수 있는 함수
- 실험용 컴파일된 .ihnn + .json 파일 (실제 컴파일러를 적용하지는 않는다.)
- 컴파일된 파일 + 시뮬레이터와 연동되는 런타임 스크립트

# 코드 구조 가이드
- 기능에 따라 파일과 헤더를 체계적으로 분리한다.
- directory tree를 체계적으로 구성하여 유지보수성을 확보한다.
- 아래는 권장 디렉토리 구조이다.

```
hw-profiling/
├── CMakeLists.txt                      # 최상위 빌드 설정
├── config/
│   ├── hw_spec.json                    # HW 스펙 (SM/UB/Queue 크기, clock 등)
│   └── data_flow.json                  # DMA latency, throughput 설정
├── src/
│   ├── hw_model/                       # 하드웨어 모델 객체들
│   │   ├── CMakeLists.txt
│   │   ├── SharedMemory.h / .cpp       # SM 객체
│   │   ├── UnifiedBuffer.h / .cpp      # UB 객체 (double buffering 포함)
│   │   ├── Queue.h / .cpp              # Queue 객체 (ISA record 배열 + PC)
│   │   ├── Core.h / .cpp               # Core 기본 클래스
│   │   ├── Pcore.h / .cpp              # Pcore 객체 (모든 연산 수행)
│   │   ├── Acore.h / .cpp              # Acore 객체 (depthwise conv만)
│   │   ├── Cluster.h / .cpp            # Cluster 객체 (core pair × n)
│   │   ├── RegisterFile.h / .cpp       # 레지스터 (start, PC, loop_ctr, out_buf_en, ref_count, read_only)
│   │   ├── DMAEngine.h / .cpp          # DMA 전송 (객체 간 데이터 복사)
│   │   └── NPU.h / .cpp               # 최상위 NPU 객체 (SM + Cluster[] + Ring)
│   ├── isa/                            # ISA 해석기
│   │   ├── CMakeLists.txt
│   │   ├── ISARecord.h / .cpp          # 2048bit record 파싱
│   │   ├── CmdHeader.h                 # cmd header 열거형 + 디코딩
│   │   ├── Opcode.h                    # opcode 열거형 + 디코딩
│   │   ├── DMAPack.h / .cpp            # 114bit DMA pack 파싱
│   │   └── Operands.h / .cpp           # opcode별 operands 파싱 (Conv, Gemm, Attention, Elementwise, Concat)
│   ├── simulator/                      # 시뮬레이터 엔진
│   │   ├── CMakeLists.txt
│   │   ├── Simulator.h / .cpp          # 클럭 기반 시뮬레이션 루프
│   │   ├── Clock.h / .cpp              # 가상 클럭 관리
│   │   ├── Logger.h / .cpp             # 시간별 데이터 이동/상태 기록
│   │   └── Profiler.h / .cpp           # 성능 프로파일링 출력
│   ├── runtime/                        # 런타임 인터페이스
│   │   ├── CMakeLists.txt
│   │   ├── Runtime.h / .cpp            # 런타임 메인 (phase 0~7 구현)
│   │   ├── IhnnParser.h / .cpp         # .ihnn 파일 파싱
│   │   ├── JsonParser.h / .cpp         # .json 파일 파싱
│   │   └── API.h / .cpp                # shared library 외부 인터페이스
│   └── binary_gen/                     # 실험용 바이너리 생성
│       ├── CMakeLists.txt
│       ├── BinaryGenerator.h / .cpp    # .ihnn + .json 생성기
│       └── TestCase.h / .cpp           # 테스트 케이스별 바이너리 생성
├── include/                            # 공통 헤더
│   ├── types.h                         # 공통 타입 정의 (bit 조작, 주소 등)
│   └── config.h                        # JSON config 로더
├── test/                               # 테스트
│   ├── unit/                           # 단위 테스트 (객체별)
│   ├── integration/                    # 통합 테스트 (phase 전체)
│   └── data/                           # 테스트용 .ihnn, .json 파일
├── scripts/                            # 유틸리티 스크립트
│   └── run_simulation.py               # 시뮬레이션 실행 + 결과 시각화
└── README.md
```

- 각 하위 디렉토리는 독립적인 CMakeLists.txt를 가진다.
- hw_model/ 내 객체들은 hw_architecture.md의 구성요소와 1:1 대응한다.
- isa/ 내 파서는 ISA.md의 정의를 그대로 구현한다.
- simulator/ 는 hw_model/ 객체들을 조합하여 클럭 단위 시뮬레이션을 수행한다.
- runtime/ 는 hw_operating_flow.md의 phase 0~7을 구현한다.
- binary_gen/ 는 compiler_output_definition.md의 .ihnn/.json 포맷에 맞춰 실험용 바이너리를 생성한다.
- config/ 의 JSON 파일은 hw_spec.md의 값을 반영하며, 스펙 변경 시 JSON만 수정한다.

# 결과물 활용 방안
- 런타임을 실행하면 시뮬레이터가 동작하여 검토하고자하는 내용을 검토할 수 있다.