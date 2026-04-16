# Skill: 빠른 계산기

> **사용자**: 모든 팀원  
> **상황**: 회의 중이나 빠른 판단이 필요할 때 특정 수치를 즉석으로 계산하고 싶을 때  
> **사용법**: 아래 예시처럼 짧게 질문

---

## UB 계산

```
Conv 3×3, stride=1, C_in=【128】, C_out=【128】, tile=【32】일 때 UB 필요량은?
```

```
Weight 【144 KB】인 layer가 UB 【128 KiB】에 들어가나?
tile을 줄이면 되나? 최소 tile은?
```

---

## SM 계산

```
총 weight 【2.19 MB】, 입력 【640×640×3 i8 = 1.17 MB】일 때 SM 【4 MiB】에 들어가나?
여유는 얼마?
```

```
SM 【X MiB】에 총 weight 【Y MB】를 넣으려면 SM이 최소 얼마여야 하나?
```

---

## Q 계산

```
【83】 conv layers, 타일 크기 【32×32】일 때 총 tile-ops는?
12코어 sequential 모델에서 코어당 records는?
Q 【512】에 들어가나?
```

---

## 타일 수 계산

```
출력 【160×160】을 【32×32】로 타일링하면 몇 tiles?
```

```
출력 【80×80】을 tile 【T】로 나눌 때, UB 【128 KiB】에 맞는 최대 T는?
(Conv 3×3, C_in=64, C_out=64)
```

---

## 크기 변환

```
【147,456】 bytes는 몇 KiB? 몇 KB?
```

```
Conv weight: C_in=【128】, C_out=【256】, K=【3×3】, i8 → 몇 bytes?
```

---

## 모델 총량

```
Conv layers 【83개】, 총 weight 【2.19 MB】일 때:
- 평균 weight per layer?
- SM 4MiB에서 weight 외 여유?
- 가장 큰 weight가 【144 KB】이면 UB 【128 KiB】 대비 몇 % 초과?
```
