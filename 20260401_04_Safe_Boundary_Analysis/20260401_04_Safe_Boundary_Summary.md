# Safe Boundary 분석 요약

**날짜**: 2026-04-01  
**Tiling 전략**: Adaptive Full (256 KiB budget)

---

## 핵심 결론

| 모델 | UB | Q | SM | 종합 |
|---|---|---|---|---|
| **YOLOv10n** | PASS (83/83) | PASS (236/512) | PASS (2.19/4.00 MB) | **PASS** |
| **YOLOv10s** | FAIL (79/87) | FAIL (6621/512) | FAIL (6.90/4.00 MB) | **FAIL** |

## YOLOv10n

- 모든 83개 layer가 UB/Q/SM 제약을 만족
- 최소 UB margin: 90,304 bytes (Layer 65, model.23.one2one_cv2.2.0.conv)
- Worst core Q 사용률: 46.1% (Core 5, 236/512)
- 하드웨어 여유 충분, 추가 최적화 불필요

## YOLOv10s

- **8개 layer의 weight가 UB(256 KiB) 초과** -- tiling으로 해결 불가
- 최대 weight: 524,288 bytes (model.9.cv2.conv, UB의 200%)
- Weight 초과 → T=1 강제 → tile 수 폭증 → Q 6개 core 초과
- 총 weight 6.90 MB > SM 4 MiB

## 필요 하드웨어 스펙 (v10s 지원)

| 리소스 | 현재 | 최소 요구 | 권장 |
|---|---|---|---|
| UB | 256 KiB | 514 KiB | 576 KiB |
| SM | 4 MiB | 7.3 MiB | 8 MiB |
| Q | 512 records | - | - (UB 확장 시 자동 해결) |

## 상세 보고서

- 전문: `20260401_04_Safe_Boundary_Analysis.md`
- CSV: `safe_boundary.csv`
