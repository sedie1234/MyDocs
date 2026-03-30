# Step2 Canonicalize + CSE

**Pass**: `--canonicalize --cse`
**파일**: MLIR 내장 pass

---

## 역할

- **Canonicalize**: 각 dialect의 canonicalization pattern 적용 (상수 전파, 항등 변환, dead code 정리)
- **CSE** (Common Subexpression Elimination): 동일 연산 결과를 하나로 합침

## 결과 (yolov10n)

| | Step1 | Step2 |
|---|---|---|
| lines | 5705 | **4215** (-26%) |
| constant | 1475 | **684** (중복 제거) |
| 연산 op 수 | 변화 없음 | 변화 없음 |

연산 그래프 구조는 변하지 않고, SSA value 중복만 정리.

## 관련

- [[Step1 InlineConstants]] — 이전 단계
- [[Step3 IdentityElim]] — 다음 단계
