# Independent Compiler 파이프라인

**최종 갱신**: 2026-03-26

---

## 확정 파이프라인

```
[Optional Pipeline] — ONNX 모델인 경우
  1. fp32 ONNX → int8 symmetric QDQ quantize (onnxruntime)
  2. ONNX QDQ metadata 추출 → .npy (ONNX node name 기반 tagging)
  3. fp32 ONNX → onnx2torch → PyTorch model

[Model to Compiler Importer]
  4. PyTorch model + conv_metadata → per-conv fake QDQ wrapping
  5. torch.export → torch_xla → StableHLO
  6. weights manifest 생성 (.npy + manifest.json)

[Compiler Main Pipeline] — 모두 ind-opt pass로 실행
  C1. --inline-constants     : 함수 인자 → stablehlo.constant (manifest.json 참조)
  C2. --fake-qdq-to-quant    : fake QDQ 패턴 → uniform_quantize/dequantize
  C3. --composite-and-qfusion : composite(silu, bias_add, bias_silu) + const fold
      --stablehlo-legalize-qdq-to-quantized-op : DQ-conv-Q → quantized conv
      --stablehlo-legalize-quant-to-math       : !quant.uniform → plain i8/i32/f32
```

## 실행 명령

```bash
# 환경: xla-test pyenv
SCRIPTS=pubH/independent/scripts
IND_OPT=pubH/build_ind/independent/tools/ind-opt
OUT=pubH/independent/test/rework

# [Optional Pipeline]
python $SCRIPTS/onnx_prepare.py \
  --onnx model.onnx \
  --fp32-onnx model.onnx \
  --output-dir $OUT

# [Importer]
python $SCRIPTS/model_to_stablehlo.py \
  --model-dir $OUT \
  --onnx model.onnx \
  --output-dir $OUT

# [Compiler Pipeline]
$IND_OPT --inline-constants --inline-constants-manifest=$OUT/weights/manifest.json \
  $OUT/stablehlo.mlir -o $OUT/step1_inlined.mlir

$IND_OPT --fake-qdq-to-quant \
  $OUT/step1_inlined.mlir -o $OUT/step2_fakeqdq.mlir

$IND_OPT --composite-and-qfusion \
  --stablehlo-legalize-qdq-to-quantized-op \
  --stablehlo-legalize-quant-to-math \
  $OUT/step2_fakeqdq.mlir -o $OUT/step3_final.mlir
```

## Pass 목록

| Pass | 종류 | 역할 |
|---|---|---|
| `--inline-constants` | custom (C++) | manifest.json + .npy → 함수 인자를 constant로 교체 |
| `--fake-qdq-to-quant` | custom (C++) | fake QDQ 패턴 → uniform_quantize/dequantize |
| `--composite-and-qfusion` | custom (C++) | SiLU/BiasAdd/BiasSiLU composite + QuantizeConstFold |
| `--stablehlo-legalize-qdq-to-quantized-op` | upstream | DQ-conv-Q → quantized conv |
| `--stablehlo-legalize-quant-to-math` | upstream | !quant.uniform → plain i8/i32/f32 |

## Custom onnx2torch Converters

| Op | 원칙 |
|---|---|
| Split, Reshape, Resize, Slice, TopK, Tile, Unsqueeze | initializer 값을 converter 등록 시점에 Python 상수로 추출 (runtime `.tolist()`/`.item()` 금지) |

## Weight/Metadata 처리 원칙

- **경로 A** (fp32 + int8): fp32 ONNX로 onnx2torch 변환, int8 ONNX에서 scale/weight/bias 추출
- **경로 B** (int8만): `fp32_val = (int8_val - zp) * scale`로 역산하여 변환

## 검증 결과 (YOLOv10n)

| 항목 | 값 |
|---|---|
| conv i8 | 83/83 |
| quant.uniform | 0 |
| bias_silu composite | 69 |
| bias_add composite | 13 |
| func args (inline 후) | 3 (입력 + 2 미매칭) |
| ultralytics 의존성 | 없음 |

## 관련 파일

| 파일 | 위치 |
|---|---|
| onnx_prepare.py | `pubH/independent/scripts/` |
| model_to_stablehlo.py | `pubH/independent/scripts/` |
| InlineConstantsPass.cpp | `pubH/independent/lib/Transforms/` |
| FakeQDQToQuantPass.cpp | `pubH/independent/lib/Transforms/` |
| CompositeAndQFusionPass.cpp | `pubH/independent/lib/Transforms/` |
| 지시서 | `docs/agent_command/independent_pipeline1.md` |
