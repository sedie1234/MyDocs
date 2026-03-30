# Step5 QuantizeRegion 변환 규칙

**최종 갱신**: 2026-03-30
**브랜치**: `step5-quantize-region`
**상태**: 실험

---

## 핵심 아이디어

DQ를 직접 제거하지 않고, **allowed op의 input을 DQ output → DQ input으로 교체** (rewire).
DQ는 dead code가 되면 canonicalize에서 자동 제거.

```
변환 전: DQ(quant→f32) → allowed_op(f32) → Q(f32→quant)
변환 후: allowed_op(quant) → Q(quant→quant)    [DQ는 dead → 자동 제거]
```

## 알고리즘

반복 수렴 방식 (max 30 iterations):

```
while (changed):
  1. 모든 DQ의 consumer 중 allowed op 탐색
  2. allowed op의 해당 operand를 DQ input(quant type)으로 교체
     - conv: 양쪽 모두 DQ일 때만 (한쪽만 바꾸면 verifier 에러)
     - 기타: 개별 교체 가능
  3. conv: 양쪽 quant이면 result를 i32 quant으로 설정
     scale = input_scale × weight_scale
  4. 일반 op: operand에 quant 있으면 result도 quant으로 전파
     + 나머지 f32 operand의 DQ도 건너뛰기
  5. 경계 보호: not-allowed op에 quant type이 흘러가면 DQ 삽입
```

## Op 분류

[[i8 허용 Op 목록]] 참조.

## 변환 예시

### Conv + Bias Add + SiLU

```mlir
// step4 (변환 전)
%0 = uniform_quantize %arg0 : f32 → quant<i8, 0.007>
%1 = uniform_dequantize %0 → f32
%2 = bitcast_convert %c : i8 → quant<i8, 0.085>
%3 = uniform_dequantize %2 → f32
%4 = convolution(%1, %3) : f32
%8 = add %4, %bias_broadcast : f32
%9 = uniform_quantize %8 → quant<i8, 0.206>
%10 = uniform_dequantize %9 → f32
%11 = logistic %10 : f32
%14 = multiply %10, %sigmoid : f32
%15 = uniform_quantize %14 → quant<i8, 0.178>

// step5 (변환 후)
%0 = uniform_quantize %arg0 : f32 → quant<i8, 0.007>
%2 = bitcast_convert %c : i8 → quant<i8, 0.085>
%4 = convolution(%0, %2) : quant<i8> × quant<i8> → quant<i32>    ★
%8 = add %4, %bias : quant<i32>                                   ★
%9 = uniform_quantize %8 : quant<i32> → quant<i8, 0.206>
%11 = logistic %9 : quant<i8>                                     ★
%14 = multiply %9, %sigmoid : quant<i8>                           ★
%15 = uniform_quantize %14 : quant<i8> → quant<i8, 0.178>
```

제거된 DQ: %1(input), %3(weight), bias DQ, %10(conv output), sigmoid DQ = **5개**

## 결과 (YOLOv10n)

| | step4 | step5b |
|---|---|---|
| DQ | 479 | **17** (-96%) |
| conv quant | 0 | **83/83** |
| logistic quant | 0 | **70/70** |
| add quant | 0 | **93/99** |

## QuantCleanupPass

Step5a 후에 추가 적용:

1. **NoopQuantize 제거**: same-scale `uniform_quantize(i8→i8)` → identity
2. **ConcatFusion**: f32 concat의 모든 input이 DQ + 모든 user가 Q → quant concat

## 관련

- [[QDQ Direct 파이프라인]] — 전체 파이프라인
- [[i8 허용 Op 목록]] — op 분류 config
- [[Step5 fp32 잔여 분석]] — 변환 불가 영역
