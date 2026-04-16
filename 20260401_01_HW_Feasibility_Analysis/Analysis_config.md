# HW Feasibility Analysis — 분석 템플릿 및 설정 가이드

## 목적
Queue, Unified Buffer, Shared Memory의 동작 가능 여부를 다양한 설정 조합으로 반복 분석하기 위한 템플릿.
설정 값을 변경한 뒤 아래 프롬프트를 사용하면 동일한 형식의 보고서가 생성된다.

---

## 1. 설정 값 (변경 가능)

### 1.1 타겟 모델

| 설정 항목 | 현재 값 | 변경 시 참고 |
|:---|:---|:---|
| 모델 | YOLOv10n, YOLOv10s | 다른 모델 추가 시 conv layer 분석 선행 필요 |
| 추론 경로 | one2one only (one2many 제외) | 학습 포함 시 one2many 추가 |
| 입력 크기 | 640×640×3 | 모델에 따라 변경 |
| 입력 데이터 타입 | i8 (1 byte) | fp16 시 2배, fp32 시 4배 |
| Weight 데이터 타입 | i8 (1 byte) | fp16/fp32 시 weight 크기 변동 |
| Activation 데이터 타입 | i8 (1 byte) | 위와 동일 |

### 1.2 하드웨어 구성

| 설정 항목 | 현재 값 | 변경 시 참고 |
|:---|:---|:---|
| 사용 코어 종류 | pCore only | aCore only / aCore+pCore pair |
| 코어 수 | 12 | 실장 기준. 최대 16 (cluster 4개 시) |
| Q 크기 (per core) | 128 KiB (512 records × 256B) | record 크기는 256B 고정 |
| UB 크기 (per core) | 128 KiB | ping-pong half = 64 KiB |
| SM 크기 (공유) | 4 MiB | 확장 시나리오: 8 MiB, 16 MiB |
| eFlash (per aCore) | 사용 안 함 (0) | aCore 사용 시 1 MiB/core |

### 1.3 타일링 전략

| 설정 항목 | 현재 값 | 변경 시 참고 |
|:---|:---|:---|
| 케이스 1 | 16×16 고정 타일 | 8×8, 24×24 등 |
| 케이스 2 | 32×32 고정 타일 | 48×48, 64×64 등 |
| 케이스 3 | Adaptive (UB half 기준 최대 타일) | UB 전체 사용, 또는 weight streaming 가정 시 변경 |
| 케이스 4 | Adaptive (UB full 기준 최대 타일) |  - |

### 1.4 운용 정책

| 설정 항목 | 현재 값 | 변경 시 참고 |
|:---|:---|:---|
| Weight 로딩 | SM에 1회 전량 적재, 추론 중 변경 없음. | layer 단위 SM 적재(streaming) 허용 시 SM 제약 완화 |
| Q 로딩 | 1회 적재, 추론 중 재로딩 없음 | 재로딩 허용 시 Q 제약 사라짐 |
| Input 로딩 | SM에 1회 전량 적재 | 타일 단위 적재 허용 시 SM peak 감소 |
| 여유 계수 | 1.5× (코어 불균형 대비) | 1.0(이상적), 2.0(보수적) |
| Overhead 범위 | instruction-level만 (LAYER_START/END, EPOCH_COMMIT, STOP) | semaphore, profiling 포함 시 SM 추가 소비 |
| UB 사용 모델 | ping-pong (half = 64 KiB) | ping-pong 미사용 시 full 128 KiB |
| Weight 위치 가정 | UB에 weight 동시 적재 <br> tile 계산에 필요한 만큼만 UB로 이동시켜 사용 | Weight Streaming 시 SM에서 직접 참조 |

---

## 2. 분석 프롬프트

아래 프롬프트의 `【설정값】` 부분을 변경하여 사용한다.

```
다음의 경우의 수를 교차하여 queue, Unified Buffer, Shared Memory의 동작 가능 여부에 대해 분석한 보고서를 작성해줘.

## 분석 조건
- 타겟 모델: 【YOLOv10n, YOLOv10s】
- 추론 경로: 【one2one only (one2many 제외)】
- 입력: 【640×640×3, i8】
- 코어: 【pCore 12개, aCore 제외】
- 여유 계수: 【1.5】

## 하드웨어 자원
- Q: 【128 KiB/core, 256B/record, 512 records max】
- UB: 【128 KiB/core, ping-pong half = 64 KiB】
- SM: 【4 MiB 공유】
- eFlash: 【사용 안 함】

## 타일링 경우의 수
1. 【16×16 고정 타일】
2. 【32×32 고정 타일】
3. 【Adaptive (UB half 기준 최대 타일)】
4. 【Adaptive (UB full 기준 최대 타일)】

## 운용 정책
- Weight: 【SM에 1회 전량 적재, 추론 중 재로딩 없음】
- Q: 【1회 적재, 추론 중 재로딩 없음】
- Input: 【SM에 1회 전량 적재】
- UB weight 정책: 【weight를 tile 연산에 사용할 만큼만 UB에 동시 적재】
- Overhead: 【instruction-level만, semaphore/profiling 제외】

## 출력 요구사항
- model의 layers와 필요한 명령어 줄을 계산할 것
- input, output, weight 이외의 metadata까지 모든 공간을 고려할 것
- 여유 자리가 얼마나 남는지를 같이 표시할 것
- 표로 봐야 보기 편한 자료는 csv도 같이 작성할 것 (보고서 내부 + csv 별도)
- 보고서 경로: docs/agent_results/reports/YYYYMMDD_【분석명】.md
- 완성된 보고서 복사 경로 : docs/Mydocs/YYYYMMDD_[index prefix]_분석명/ directory
- 요약 보고서도 별도 작성

## 참조 데이터
- Conv Layer 분석: docs/agent_results/etc/20260401_yolov10_conv_layer_analysis.md
- HW Spec: docs/agent_reference/2026.01.06_02.ArchitectureSpecification.pdf
- ISA Spec: docs/agent_reference/2026.02.12_SSPC_Simulator_ISA Specification_v0.8.pdf
```

---

## 3. 설정 변경 시나리오 예시

### 시나리오 A: Weight Streaming 허용
```
변경 항목:
- Weight: SM에 전량 적재하되, UB에는 slice 단위 DMA (전량 적재 아님)
- UB weight 정책: weight를 UB에 동시 적재 → weight streaming (SM에서 직접 참조)
- 타일링 케이스 3: Adaptive 기준을 input_tile + output_tile + metadata ≤ 64KiB (weight 제외)

기대 효과: UB FAIL 레이어 대폭 감소
```

### 시나리오 B: SM 확장
```
변경 항목:
- SM: 4 MiB → 8 MiB

기대 효과: v10s weight(6.90MB) + input(1.17MB) = 8.07MB → 8MiB에 근접, 여전히 빠듯
         v10n은 완전 해소
```

### 시나리오 C: aCore + pCore 동시 사용
```
변경 항목:
- 코어: aCore + pCore pair (12 NPU_Core = 24 실행 엔진)
- eFlash: 1 MiB/aCore 사용
- Q: aCore Q(128KiB) + pCore Q(128KiB) = 코어당 2개
- UB: aUB(128KiB) + pUB(128KiB) = 코어당 256KiB

기대 효과: Q 용량 2배, UB 용량 2배, eFlash에 weight 분산 가능
```

### 시나리오 D: 큰 고정 타일
```
변경 항목:
- 케이스 1: 48×48 고정
- 케이스 2: 64×64 고정
- 케이스 3: Adaptive 유지

기대 효과: tile-ops 감소 → Q 부담 감소, 단 UB 부담 증가
```

### 시나리오 E: Q 재로딩 허용
```
변경 항목:
- Q: 1회 적재 → layer 그룹 단위 재로딩 허용

기대 효과: Q 제약 사실상 해소 (Adaptive에서도 PASS 가능)
```

### 시나리오 F: 보수적 분석 (여유 계수 2.0)
```
변경 항목:
- 여유 계수: 1.5 → 2.0

기대 효과: PASS 기준 더 엄격해짐, 현실적 worst-case 반영
```

---

## 4. 분석 이력

| 날짜 | 시나리오 | 설정 요약 | 결과 | 보고서 위치 |
|:---|:---|:---|:---|:---|
| 2026-04-01 | 01 기본 (Baseline) | pCore×12, SM 4MiB, UB half(64K), 3 tiling, Q 1회적재 | 6/6 FAIL | `20260401_01_HW_Feasibility_Analysis/` |
| 2026-04-01 | 02 Adaptive Full 추가 | Baseline + Adaptive Full(UB 128K) 추가, 4 tiling | 7/8 FAIL, **v10n Adaptive Full = COND** | `20260401_02_HW_Feasibility_Analysis/` |
| | | | | |
| | | | | |

---

## 5. 핵심 수치 레퍼런스 (빠른 참조)

### 모델 요약

| 모델 | Conv layers | Total ops (추정) | Weight (s8) | Max single weight |
|:---|---:|---:|---:|---:|
| v10n (inference) | 83 | ~113 | 2.19 MB | 144 KB |
| v10s (inference) | 87 | ~117 | 6.90 MB | 512 KB |

### 해상도별 타일 수

| 해상도 | 8×8 | 16×16 | 32×32 | 48×48 | 64×64 |
|:---|---:|---:|---:|---:|---:|
| 320×320 | 1,600 | 400 | 100 | 49 | 25 |
| 160×160 | 400 | 100 | 25 | 16 | 9 |
| 80×80 | 100 | 25 | 9 | 4 | 4 |
| 40×40 | 25 | 9 | 4 | 1 | 1 |
| 20×20 | 9 | 4 | 1 | 1 | 1 |

### 자원 한계

| 자원 | 용량 | 단위 | 비고 |
|:---|---:|:---|:---|
| Q | 512 | records/core | 256B/record, 128KiB total |
| UB | 128 KiB | per core | ping-pong half = 64 KiB |
| UB (half) | 64 KiB | per core | 실효 사용량 |
| SM | 4 MiB | 공유 | 12코어 전체 |
| eFlash | 1 MiB | per aCore | aCore 미사용 시 0 |
