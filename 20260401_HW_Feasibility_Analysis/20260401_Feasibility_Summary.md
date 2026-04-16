# Queue / UB / SM 동작 가능 여부 분석 — 결과 요약

생성일: 2026-04-01
상세 보고서: `20260401_Queue_UB_SM_Feasibility_Analysis.md`

---

## 1. 분석 조건

| 항목 | 값 |
|:---|:---|
| 대상 모델 | YOLOv10n (83 layers, weight 2.19 MB) / YOLOv10s (87 layers, weight 6.90 MB) |
| 추론 경로 | backbone + neck + one2one (one2many 제외) |
| 코어 | 12 pCore (aCore 제외) |
| Q | 128 KiB/core = 512 records (256B/record) |
| UB (pUB) | 128 KiB/core, ping-pong half = 64 KiB |
| SM | 4 MiB 공유 (12코어 전체) |
| eFlash | 없음 (aCore 전용, 분석 제외) |
| 타일링 | (1) 16×16 고정 (2) 32×32 고정 (3) Adaptive (UB half 기준 최대 타일) |
| Weight/Q | 추론 전 1회 로드, 추론 중 재로딩 없음 |
| 입력 | 640×640×3 i8 = 1.17 MB, 1회 SM 로드 |
| 여유 계수 | 1.5× (코어 분배 불균형 대비) |

---

## 2. 종합 판정표

| 모델 | 타일링 | Q (512 max) | UB (64 KiB half) | SM (4 MiB) | **종합** |
|:---|:---|:---|:---|:---|:---|
| v10n | 16×16 | **PASS** (343/512, 67%) | **FAIL** (52/83 layers 초과) | **COND** (3 layers 초과) | **FAIL** |
| v10n | 32×32 | **PASS** (124/512, 24%) | **FAIL** (80/83 layers 초과) | **COND** (3 layers 초과) | **FAIL** |
| v10n | Adaptive | **FAIL** (908/512, 177%) | **COND** (11/83 layers 초과) | **COND** (3 layers 초과) | **FAIL** |
| v10s | 16×16 | **PASS** (346/512, 68%) | **FAIL** (76/87 layers 초과) | **FAIL** (wt 6.90MB > 4MB) | **FAIL** |
| v10s | 32×32 | **PASS** (125/512, 24%) | **FAIL** (86/87 layers 초과) | **FAIL** (wt 6.90MB > 4MB) | **FAIL** |
| v10s | Adaptive | **FAIL** (6404/512) | **FAIL** (38/87 layers 초과) | **FAIL** (wt 6.90MB > 4MB) | **FAIL** |

> **6개 케이스 모두 종합 FAIL.** Q, UB, SM이 동시에 PASS하는 조합 없음.

---

## 3. 자원별 병목 분석

### 3.1 Queue — 대체로 양호

| 타일링 | v10n core당 records | v10s core당 records | 판정 |
|:---|---:|---:|:---|
| 16×16 | 343 / 512 | 346 / 512 | PASS (여유 33%) |
| 32×32 | 124 / 512 | 125 / 512 | PASS (여유 76%) |
| Adaptive | 908 / 512 | 6,404 / 512 | FAIL |

- 16×16, 32×32에서는 양 모델 PASS
- Adaptive FAIL 원인: weight > UB_half인 레이어가 tile=1로 퇴화 → tile 수 폭발

### 3.2 Unified Buffer — 가장 심각한 병목

**FAIL 원인: weight + input_tile + output_tile + metadata(512B)가 UB half(64 KiB)를 초과**

| 타일링 | v10n PASS layers | v10n FAIL layers | v10s PASS layers | v10s FAIL layers |
|:---|---:|---:|---:|---:|
| 16×16 | 31 / 83 | **52** | 11 / 87 | **76** |
| 32×32 | 3 / 83 | **80** | 1 / 87 | **86** |
| Adaptive | 72 / 83 | **11** | 49 / 87 | **38** |

- 32×32 타일: tile 자체가 커서 weight 없이도 input+output만으로 UB 초과하는 경우 다수
- 16×16 타일: tile은 작지만 weight를 UB에 동시 적재해야 하므로 weight>수십KB 레이어에서 FAIL
- Adaptive: weight > 64KiB 레이어는 어떤 타일 크기로도 해결 불가

**v10n UB FAIL 레이어 (Adaptive 기준, 11개):**

| Layer | 이름 | Weight (KB) | 초과량 |
|:---|:---|---:|:---|
| model.6.m.0.cv1 | Conv 3×3, C=64→64 | 36 | weight+tile > 64KiB |
| model.6.m.0.cv2 | Conv 3×3, C=64→64 | 36 | 동일 |
| model.6.m.1.cv1 | Conv 3×3, C=64→64 | 36 | 동일 |
| model.6.m.1.cv2 | Conv 3×3, C=64→64 | 36 | 동일 |
| model.8.m.0.cv1 | Conv 3×3, C=128→128 | 144 | weight 단독 > UB half |
| model.8.m.0.cv2 | Conv 3×3, C=128→128 | 144 | 동일 |
| model.9.cv2 | 1×1, C=512→256 | 128 | weight 단독 > UB half |
| model.10.cv2 | 1×1, C=256→256 | 64 | 경계 초과 |
| one2one_cv2.2.0 | Conv 3×3, C=256→64 | 144 | weight 단독 > UB half |
| model.19.m.0.cv1 | Conv 3×3, C=64→64 | 36 | weight+tile > 64KiB |
| model.19.m.0.cv2 | Conv 3×3, C=64→64 | 36 | 동일 |

### 3.3 Shared Memory

| 항목 | v10n | v10s |
|:---|:---|:---|
| Weight | 2.19 MB | 6.90 MB |
| Input | 1.17 MB | 1.17 MB |
| 정적 합산 | 3.36 MB | **8.07 MB** |
| SM 용량 | 4.00 MB | 4.00 MB |
| 정적 판정 | PASS (여유 0.64MB) | **FAIL (4.07MB 초과)** |

**v10n SM — Layer 전이 Peak 분석:**

| Layer | Input FM | Output FM | Peak (wt+in+out) | SM 4MiB | 판정 |
|:---|---:|---:|---:|---:|:---|
| 0 (stem) | 1.17 MB | 1.56 MB | 4.92 MB | 4.00 MB | **FAIL (-0.92MB)** |
| 1 (stride-2) | 1.56 MB | 0.78 MB | 4.53 MB | 4.00 MB | **FAIL (-0.53MB)** |
| 5 (cv2) | 1.17 MB | 0.78 MB | 4.14 MB | 4.00 MB | **FAIL (-0.14MB)** |
| 2 (cv1) | 0.78 MB | 0.78 MB | 3.75 MB | 4.00 MB | PASS (+0.25MB) |
| 6~ (이후 전부) | ≤0.78 MB | ≤0.78 MB | ≤3.75 MB | 4.00 MB | PASS |

- v10s: **물리적 불가** (weight 단독으로 SM 초과)
- v10n: Layer 0, 1, 5만 초과 → 우회 가능

---

## 4. 실현 가능성 순위

| 순위 | 자원 | 실현성 | 비고 |
|---:|:---|:---|:---|
| 1 | Q | **높음** | 16×16/32×32에서 양 모델 PASS |
| 2 | SM (v10n) | **중간** | 정적 PASS, 초기 3개 레이어만 우회 필요 |
| 3 | UB (v10n) | **중간-낮음** | Weight Streaming 적용 시 해결 가능성 |
| 4 | SM (v10s) | **불가** | 외부 메모리 또는 SM 확장 없이 해결 불가 |

---

## 5. 권고사항

### 5.1 v10n 실현 가능 시나리오

```
전략: 16×16 고정 타일 + Weight Streaming + Layer 0~1 특별처리

Q:   PASS — 343/512 records (여유 33%)
UB:  Weight Streaming 적용
     → weight를 UB에 전량 적재하지 않고 SM에서 slice 단위 DMA
     → UB에는 input_tile + output_tile + weight_slice만 적재
     → 16×16 tile + C=128 기준: input+output ≈ 32KB, weight_slice 여유 충분
SM:  Layer 0~1: 입력을 타일 단위로 SM 적재 (전체 1.17MB 대신)
     Layer 2~: 정상 동작 (여유 0.25MB+)
```

**전제 조건:** Weight Streaming이 ISA/HW에서 지원되어야 함

### 5.2 v10s 실현을 위한 아키텍처 변경

| 방안 | 변경 사항 | 효과 |
|:---|:---|:---|
| SM 확장 | 4 MiB → **8 MiB** 이상 | weight(6.9) + input(1.2) = 8.1MB 수용 |
| 외부 메모리 활용 | DRAM/eFlash에 weight 상주, layer 단위 SM 적재 | SM 부담 제거 |
| Weight 전용 버퍼 | UB와 별도의 weight cache 추가 | UB 병목 해소 |
| 모델 경량화 | v10s → v10n 사용 | 즉시 적용 가능 |

### 5.3 우선순위

1. **즉시**: v10n + 16×16 + Weight Streaming 설계 검토
2. **단기**: Weight Streaming ISA 지원 여부 하드웨어팀과 협의
3. **중기**: SM 확장 또는 외부 메모리 경로 검토 (v10s 지원 목표)

---

## 6. 참조 파일

| 파일 | 경로 |
|:---|:---|
| 상세 보고서 | `docs/agent_results/reports/20260401_Queue_UB_SM_Feasibility_Analysis.md` |
| Conv Layer 분석 | `docs/agent_results/etc/20260401_yolov10_conv_layer_analysis.md` |
| Queue CSV | `docs/agent_results/reports/20260401_queue_analysis.csv` |
| UB CSV | `docs/agent_results/reports/20260401_ub_analysis.csv` |
| SM CSV | `docs/agent_results/reports/20260401_sm_analysis.csv` |
| 본 요약 CSV | `docs/agent_results/reports/20260401_feasibility_summary.csv` |
