# HW Feasibility Analysis Summary (v2)

생성일: 2026-04-01 | Baseline: 20260401_Queue_UB_SM_Feasibility_Analysis.md

## Baseline 대비 변경

- Tiling Case 4 추가: **Adaptive Full** (UB 128 KiB, ping-pong 미사용)
- 총 8 cases = 2 models x 4 tiling strategies

## 8-Case 종합 판정 매트릭스

| Model | Tiling | Q (512 rec) | UB | SM (4 MiB) | **종합** |
|:---|:---|:---|:---|:---|:---|
| v10n | 16x16 | **PASS** (343) | **FAIL** (52 fail) | **COND** (3 fail) | **FAIL** |
| v10n | 32x32 | **PASS** (124) | **FAIL** (79 fail) | **COND** (3 fail) | **FAIL** |
| v10n | Adaptive Half | **FAIL** (908) | **COND** (11 fail) | **COND** (3 fail) | **FAIL** |
| **v10n** | **Adaptive Full** | **PASS** (326) | **COND** (4 fail) | **COND** (3 fail) | **COND** |
| v10s | 16x16 | **PASS** (346) | **FAIL** (76 fail) | **FAIL** | **FAIL** |
| v10s | 32x32 | **PASS** (125) | **FAIL** (85 fail) | **FAIL** | **FAIL** |
| v10s | Adaptive Half | **FAIL** (6,404) | **FAIL** (38 fail) | **FAIL** | **FAIL** |
| v10s | Adaptive Full | **FAIL** (3,577) | **FAIL** (29 fail) | **FAIL** | **FAIL** |

## 핵심 결과

### v10n Adaptive Full = 유일한 COND

Baseline 6-case 분석에서는 모든 조합 FAIL. Adaptive Full 도입으로 최초의 COND 출현.

| 자원 | Adaptive Half | Adaptive Full | 변화 |
|:---|:---|:---|:---|
| Q | FAIL (908/512) | **PASS** (326/512) | 64% 감소 |
| UB fail | 11/83 | **4/83** | 7개 해소 |
| SM | COND (3 fail) | COND (3 fail) | 변동 없음 |

### v10n 잔여 이슈 (7개 레이어)

**UB 4개 fail** (weight > 128 KiB):
- Layer 24, 25: Conv 3x3 C=128->128, weight=144KB
- Layer 28: 1x1 C=512->256, weight=128KB (단 1.25KB 초과)
- Layer 65: Conv 3x3 C=256->64, weight=144KB

**SM 3개 fail** (초기 고해상도):
- Layer 0, 1, 5: 640x640/320x320/160x160 FM

### v10s: 여전히 FAIL

Weight(6.90MB) > SM(4MB)으로 근본적 해결 불가. SM 8MiB 이상 확장 필요.

## 권고

1. **v10n: Adaptive Full 채택** + 4개 레이어 weight tiling + SM Layer 0~1 특별처리
2. **Trade-off 인지**: Ping-pong 미사용 = DMA overlap 불가 = 타일당 latency 증가. 그러나 tile 수 감소로 상쇄 가능.
3. **Hybrid 전략**: weight <= 64KB 레이어는 ping-pong, 64~128KB는 single buffer
4. **v10s**: SM 확장 또는 외부 메모리 도입 없이 실행 불가

---

상세 분석: `20260401_02_HW_Feasibility_Analysis.md`
