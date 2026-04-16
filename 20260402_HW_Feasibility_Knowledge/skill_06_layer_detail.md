# Skill: Layer 단위 상세 분석

> **사용자**: HW 설계자, 컴파일러 개발자  
> **상황**: 특정 layer가 왜 FAIL인지, 어떤 값을 바꾸면 PASS가 되는지 상세히 확인하고 싶을 때  
> **사용법**: FAIL layer 인덱스 또는 이름을 입력

---

## 프롬프트 1: FAIL layer 원인 분석

```
【YOLOv10n】의 Layer 【47 (model.8.m.0.cv1)】이 UB에서 FAIL하는 원인을 
단계별로 계산 전개해서 보여줘.

다음을 포함:
- 이 layer의 kernel, stride, channel 정보
- weight 크기 계산 과정
- input tile 크기 계산 과정 (receptive field 역산)
- output tile 크기 계산 과정
- 총 UB 필요량 vs 예산
- PASS가 되려면 어떤 값이 얼마로 바뀌어야 하는지
```

---

## 프롬프트 2: "이 layer를 돌리려면?"

```
Conv layer 스펙이 다음과 같다.
- Kernel: 【3×3】
- Stride: 【1】
- Input: 【80×80×128】
- Output: 【80×80×128】
- Weight: 【128×128×3×3×1 = 147,456 bytes】

이 layer를 UB 【128 KiB】에서 실행하려면:
1. 가능한 최대 타일 크기는?
2. 그 타일 크기에서의 Q record 수는?
3. Weight streaming 없이 가능한가?
4. Weight streaming이 있으면?

각 답에 계산 과정을 보여줘.
```

---

## 프롬프트 3: Safe Boundary 역산

```
UB가 【256 KiB】일 때, 【Conv 3×3, stride=1】 layer에서:
- 최대 허용 weight는 몇 bytes?
- 최대 허용 채널 폭은?
- 입력 채널과 출력 채널이 같다고 가정했을 때 최대 C는?

계산식과 함께 보여줘.
```
