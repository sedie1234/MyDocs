# input_locations 기반 InlineConstantsPass

**최종 갱신**: 2026-03-27

---

## 문제

torch_xla가 StableHLO 생성 시 function arg 순서를 재배열.
이전 FIFO(shape bucket) 매칭이 bias를 74/82 잘못 배정.

- Weight (4D): shape가 대부분 유일 → FIFO 정확
- **Bias (1D)**: `(64,)` 24개, `(128,)` 20개 등 중복 → FIFO 순서 불일치

## 해결: input_locations

`StableHLOGraphModule._bundle.stablehlo_funcs[0].meta.input_locations[i]`가 MLIR `%arg_i`에 대응하는 parameter name을 정확히 제공.

```python
meta.input_locations[i].type_    # PARAMETER | INPUT_ARG | CONSTANT
meta.input_locations[i].name     # "conv1.weight", "conv1.bias" 등
```

## 매핑 흐름

```
[Python]
input_locations[i] → state_dict[name] → .npy 저장
manifest.json: { "0": {mlir_arg_idx: 0, file: "conv1.weight.npy"} }

[C++]
manifest 로드 → argIdxToNpy[0] = conv1.weight.npy
%arg0 → DenseElementsAttr → stablehlo.constant 교체
```

## 검증

| 항목 | 결과 |
|---|---|
| Weight ONNX 일치 | 83/83 ✅ |
| Bias ONNX 일치 | 82/82 ✅ |
| Weight↔Bias 일관성 | 82/82 ✅ |
| 불일치 | **0** |

## 관련

- [[QDQ Direct 파이프라인]] — 이 매핑을 사용하는 파이프라인
- [[Quantized StableHLO 파이프라인]] — 이전 FIFO 방식 (bias 매칭 버그)
