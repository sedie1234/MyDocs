# Step1 InlineConstants

**Pass**: `--inline-constants --inline-constants-manifest=manifest.json`
**파일**: `InlineConstantsPass.cpp`

---

## 역할

StableHLO의 function arguments를 manifest의 npy 파일에서 로드한 `stablehlo.constant`로 교체.

## 동작

```
입력: func @main(%arg0: tensor<f32>, %arg1: tensor<i8>, ..., %arg1361: tensor<1x3x640x640xf32>)

manifest.json:
  "0": {mlir_arg_idx: 0, file: "scale.npy"}
  "1": {mlir_arg_idx: 1, file: "weight.npy"}
  ...

출력: func @main(%arg0: tensor<1x3x640x640xf32>)  ← user input만 남음
  %cst_0 = stablehlo.constant dense<0.007> : tensor<f32>
  %c_1 = stablehlo.constant dense<"0x..."> : tensor<16x3x3x3xi8>
  ...
```

1. manifest.json 로드 → `{mlir_arg_idx → npy raw bytes}` map 구축
2. 각 function arg에 대해 map에 있으면 → `DenseElementsAttr` 생성 → `stablehlo.constant` 교체
3. 교체된 args를 function signature에서 역순 제거

## 결과 (yolov10n)

- 1361 args → 1 arg (user input)
- 5705 lines

## 관련

- [[Step0 모델 준비]] — manifest 생성
- [[input_locations 기반 InlineConstantsPass]] — 매핑 정확성
- [[Step2 Canonicalize + CSE]] — 다음 단계
