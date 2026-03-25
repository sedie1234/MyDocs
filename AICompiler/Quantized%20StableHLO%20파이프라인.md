# Quantized StableHLO 파이프라인

**최종 갱신**: 2026-03-25
**커밋**: pubH `9e226e7` (independent-test)

---

## 확정 파이프라인

```
[Step 1] fp32 ONNX → int8 symmetric QDQ quantize (onnxruntime)
[Step 2] ONNX QDQ per-conv scale 추출 → conv_metadata.json
[Step 3] ultralytics .pt → BN folding → bias 분리 → per-conv fake QDQ
[Step 4] torch.export → torch_xla → StableHLO (weight inline + elide)
[Step 5] ind-opt --fake-qdq-to-quant (FakeQDQToQuantPass)
[Step 6] ind-opt --stablehlo-legalize-qdq-to-quantized-op (QFusion)
```

## 실행 명령

```bash
~/.pyenv/versions/xla-test/bin/python \
  pubH/independent/scripts/onnx_to_quantized_stablehlo.py \
  --onnx testStage/work1/yolov10n.onnx \
  --pt testStage/work1/yolov10n.pt \
  -o output_dir \
  --elide-hex 4 --decode-hex
```

### 옵션

| 옵션 | 설명 |
|---|---|
| `--qdq-onnx` | 이미 양자화된 ONNX 지정 (Step 1 생략) |
| `--elide-hex 0` | `dense<...>` (최소 표시) |
| `--elide-hex 4` | 앞 4개 element 표시 |
| `--elide-hex -1` | 전체 데이터 |
| `--decode-hex` | hex를 float/int 값으로 디코딩 표시 |

## 검증 결과 (YOLOv10n)

| 항목 | 결과 |
|---|---|
| quantized conv | **83/83 (100%)** |
| f32 conv | 0 |
| batch_norm | **0** (BN folding) |
| 고유 scale | **150** (per-conv 보존) |

## 출력 파일

```
output_dir/
├── step4_stablehlo.mlir       ← 열람용 (elide, ~180K)
├── step4_stablehlo_full.mlir  ← pass용 (전체 hex, ~18M)
├── step5_fakeqdq.mlir         ← 열람용
├── step5_fakeqdq_full.mlir    ← pass용
├── step6_qfusion.mlir         ← 열람용 (최종)
├── step6_qfusion_full.mlir    ← pass용 (최종)
├── weights/                   ← weight .npy 파일 (168개, ~9.5M)
│   └── manifest.json          ← argN ↔ parameter 매핑
└── conv_metadata.json         ← ONNX per-conv scale/zp
```

## 핵심 설계 결정

### BN folding + bias 분리
- `pt_model.fuse()`: BN을 Conv에 흡수 → batch_norm 0개
- Conv의 bias를 분리: `conv(no bias) → Q → DQ → add(bias)`
- QFusion이 `DQ-conv-Q` 패턴 매칭 가능

### Weight 관리
- pass용 MLIR: 전체 hex inline (`_full.mlir`, 18MB)
- 열람용 MLIR: elide (`step4/5/6.mlir`, ~180K)
- 바이너리 데이터: `weights/*.npy` (별도 관리)

### FakeQDQToQuantPass 확장
- Order A/B: `divide→round→clamp→multiply` vs `divide→clamp→round→multiply`
- ZP 지원: `add(zp)/subtract(zp)` 포함 패턴
- uint8 → int8 ZP 변환

## 경로 탐색 이력

| 경로 | 결과 | 양자화 보존 |
|---|---|---|
| **확정: fake QDQ (per-conv)** | **성공, 83/83** | **150개 고유 scale** |
| torch_xla + PT2E | HLO 성공, StableHLO 실패 | xla#9291 Open |
| onnx2tf → TF PTQ | 성공 (math 분해) | scale 유실 |
| torch-mlir → StableHLO | 실패 (sort) | - |
| torch-mlir → TOSA | 실패 (sort, full) | - |
| onnx2torch (QDQ/QOp) | 실패 (QDQ 미지원) | - |
| onnx-mlir | 빌드 실패 / QDQ 미구현 | - |

## 관련 이슈

- openxla/xla#9291: XLA HLO quant type 미지원 → PT2E 경로 차단 원인
