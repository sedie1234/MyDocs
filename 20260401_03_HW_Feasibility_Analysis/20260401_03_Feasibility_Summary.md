# HW Feasibility Analysis Summary (v3)

생성일: 2026-04-01

## 분석 조건 (v3 주요 변경사항)

- UB: **256 KiB** (v2: 128 KiB)
- 실행 모델: **Sequential** (v2: 병렬 균등)
- SM: **Input/Output overlap 가능** (peak = weight + max(in, out))
- UB 모델: **Weight-first** (weight 전량 적재, 잔여에 input slice)
- 판정: **PASS/FAIL only** (COND 폐지)

## 종합 결과

| Model | Tiling | Q | UB | SM | **종합** |
|:---|:---|:---|:---|:---|:---|
| v10n | 16x16 | FAIL (1,150) | FAIL (5) | PASS | **FAIL** |
| v10n | 32x32 | PASS (320) | FAIL (24) | PASS | **FAIL** |
| v10n | Adaptive Half | FAIL (6,577) | FAIL (4) | PASS | **FAIL** |
| **v10n** | **Adaptive Full** | **PASS (236)** | **PASS (83/83)** | **PASS** | **PASS** |
| v10s | 16x16 | FAIL (1,177) | FAIL (31) | FAIL | **FAIL** |
| v10s | 32x32 | PASS (331) | FAIL (59) | FAIL | **FAIL** |
| v10s | Adaptive Half | FAIL (27,954) | FAIL (29) | FAIL | **FAIL** |
| v10s | Adaptive Full | FAIL (6,621) | FAIL (8) | FAIL | **FAIL** |

> Q 열의 숫자는 worst core의 instruction records (한계: 512)
> UB 열의 숫자는 FAIL 레이어 수

## 유일한 PASS: v10n Adaptive Full

| 자원 | Worst Case | 한계 | 여유율 |
|:---|:---|:---|:---|
| Q | Core5: 236 records | 512 | 53.9% |
| UB | Layer 3: 262,048 B | 262,144 B | 0.04% |
| SM | Layer 0: 3,930,976 B | 4,194,304 B | 6.3% |

## 버전별 비교

| | v1 (Baseline) | v2 | **v3** |
|:---|:---|:---|:---|
| v10n 최선 | FAIL | COND | **PASS** |
| v10n UB FAIL | 11 | 4 | **0** |
| v10n SM FAIL | 3 | 3 | **0** |
| v10s 최선 | FAIL | FAIL | **FAIL** |
| v10s UB FAIL 최소 | 38 | 29 | **8** |

## v10s 근본 원인

- Weight 6.90 MB > SM 4 MB: 어떤 타일링으로도 해결 불가
- SM 8 MiB 이상 확장 필수

## 상세 보고서

`20260401_03_HW_Feasibility_Analysis.md`
