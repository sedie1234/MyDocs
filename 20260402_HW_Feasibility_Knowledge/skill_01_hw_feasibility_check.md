# Skill: HW Feasibility 분석

> **사용자**: HW 설계자, ISA 설계자  
> **상황**: 하드웨어 스펙을 변경했을 때 특정 모델이 동작 가능한지 확인하고 싶을 때  
> **사용법**: 아래 프롬프트의 【 】 부분을 자신의 값으로 채워서 대화창에 입력

---

## 프롬프트

```
다음 하드웨어 스펙으로 feasibility 분석을 수행해줘.
"HW Feasibility 분석 지식 문서"의 출력 포맷을 따를 것.

## 하드웨어 스펙
- 코어: 【12 pCore】
- Q: 【128 KiB/core, 256B/record, 512 records max】
- UB: 【256 KiB/core, ping-pong half = 128 KiB】
- SM: 【4 MiB 공유】

## 타겟 모델
- 【YOLOv10n】

## 타일링 전략
1. 【Adaptive Half (UB half 기준)】
2. 【Adaptive Full (UB 전체 기준)】

## 운용 정책
- Weight: 【SM에 1회 전량 적재, UB에 weight 동시 적재】
- SM overlap: 【true】
- 여유 계수: 【1.5】

## 특히 확인하고 싶은 점
- 【예: UB를 512KiB로 늘렸을 때 FAIL layer가 몇 개 줄어드는지】
```

---

## 간단 버전 (빠른 질문용)

```
현재 baseline 스펙에서 【UB를 512 KiB로 변경】하면 YOLOv10n feasibility가 어떻게 바뀌나?
FAIL → PASS 되는 layer 목록과 계산 과정을 보여줘.
```

---

## 예상 응답 형식

문서에 정의된 7개 섹션으로 응답:
1. 분석 조건 요약
2. 종합 판정표
3. Queue 상세
4. UB 상세 (FAIL layer는 계산식 전개)
5. SM 상세
6. Safe Boundary
7. 권고사항 + 런타임 적재 검증
