# Step0 모델 준비

**최종 갱신**: 2026-03-30

---

## 개요

PyTorch/ONNX 모델을 StableHLO MLIR로 변환하는 과정. Step1 이전에 Python에서 수행.

```
QDQ ONNX → onnx2torch (functional conv) → torch.export → torch_xla → StableHLO
                                                            ↓
                                                     input_locations → manifest.json
```

---

## Step0a: QDQ 양자화 (선택)

이미 QDQ ONNX가 있으면 생략.

```bash
python onnx_prepare.py --onnx model_fp32.onnx --output-dir ./work
```

- 입력: fp32 ONNX
- 출력: `model_qdq_int8sym.onnx`
- 방법: onnxruntime `quantize_static` (int8 symmetric, per-tensor, MinMax)

### 모델 요구사항

| 항목 | 요구 |
|---|---|
| ONNX opset | 17 (최소 13) |
| Shape | **정적** (dynamic 미지원) |
| Batch size | 1 |
| 양자화 형식 | **QDQ** (QOperator 미지원) |
| 양자화 타입 | **int8 symmetric** (per-tensor) |
| Conv+BN | **fusion 후 export** |

---

## Step0b: onnx2torch 변환

QDQ ONNX를 PyTorch 모델로 변환.

```python
from onnx_prepare import register_qdq_converters
register_qdq_converters()
from onnx2torch import convert
model = convert('model_qdq_int8sym.onnx')
```

### Custom Converters

onnx2torch가 기본 지원하지 않는 op에 대한 converter:

| Op | Converter | 설명 |
|---|---|---|
| QuantizeLinear | `OnnxQuantizeLinear` | `round(x/scale) + zp`, clamp |
| DequantizeLinear | `OnnxDequantizeLinear` | `(x - zp) * scale` |
| **Conv (QDQ)** | `OnnxFunctionalConv2d` | weight를 forward 입력으로 받음 |
| Split | `OnnxStaticSplit` | static split sizes |
| Reshape | `OnnxStaticReshape` | static target shape |
| Resize | static | static scale/size |
| Slice | `OnnxStaticSliceOp` | static start/end/step |
| TopK | `OnnxStaticTopK` | static k |
| Tile | `OnnxStaticTile` | static repeats |
| Unsqueeze | `OnnxStaticUnsqueeze` | static axes |

**핵심: Functional Conv**

QDQ 모델에서 Conv의 weight가 initializer가 아닌 DequantizeLinear의 출력.
기존 `nn.Conv2d`는 weight를 내부 파라미터로 저장하므로 불가.
→ `F.conv2d(x, weight, bias)` 방식으로 weight를 forward 입력으로 받음.

---

## Step0c: torch.export → torch_xla → StableHLO

```python
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo

exported = export(model, (torch.randn(1, 3, 640, 640),), strict=False)
shlo = exported_program_to_stablehlo(exported)
text = shlo.get_stablehlo_text('forward')
```

모든 weight/bias/scale이 function argument가 됨.

---

## Step0d: input_locations 기반 manifest 생성

```python
meta = shlo._bundle.stablehlo_funcs[0].meta
sd = shlo._bundle.state_dict

for i, loc in enumerate(meta.input_locations):
    # loc.type_ = PARAMETER → %arg_i에 대응하는 parameter name
    # sd[loc.name] → tensor 값 → .npy 저장
    manifest[str(i)] = {mlir_arg_idx: i, name: loc.name, file: fname}
```

**`input_locations[i]`가 MLIR `%arg_i`에 정확히 대응.** torch_xla의 arg 재배열 후 순서로 정렬됨.

→ 상세: [[input_locations 기반 InlineConstantsPass]]

---

## 출력

```
work/
├── stablehlo.mlir              ← weights = function args (1362개)
└── weights/
    ├── manifest.json           ← {mlir_arg_idx → npy file} 매핑
    └── *.npy                   ← parameter 값 (1361개)
```

---

## 실행 (통합)

```bash
python model_to_stablehlo.py \
  --model-dir ./work --onnx model_fp32.onnx --output-dir ./work
```

---

## 관련

- [[QDQ Direct 파이프라인]] — 전체 파이프라인
- [[input_locations 기반 InlineConstantsPass]] — manifest → inline 매핑
- [[Step1 InlineConstants]] — 다음 단계
