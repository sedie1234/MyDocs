- IREE의 정규식에서 Conv의 Input은 Pad를 포함하고 있다. 이것을 원래 Input으로 바꾸고, Pad는 Attributes로써 Conv에 포함시킨다.

# 1. Patterns

## 1.1 VailPadPattern
- 아래 식에서 %8의 값을 %7에 넣음으로써 padding을 가진 input을 만들고 있다. 
```
    %7 = linalg.fill ins(%c-128_i8 : i8) outs(%6 : tensor<3x642x642xi8>) -> tensor<3x642x642xi8>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed : tensor<3x640x640xf32>) outs(%5 : tensor<3x640x640xi8>) {
    
    ...
    
    } -> tensor<3x640x640xi8>
    %inserted_slice = tensor.insert_slice %8 into %7[0, 1, 1] [3, 640, 640] [1, 1, 1] : tensor<3x640x640xi8> into tensor<3x642x642xi8>
```

- trigger : `inf_cap.conv2d`
- 1. input이 `tensor.insert_slice`인지 확인
- 2. insert_slice의 offset[1], offset[2]에서 pad_h, pad_w 추출
- 3. conv2d의 pad 속성을 업데이트하고, input을 unpadded source로 교체

```
[before]
 %padded_empty = tensor.empty() : tensor<3x642x642xi8>
 %fill_const = linalg.fill ins(%cst) outs(%padded_empty)
 %quant_out = linalg.generic { ... }  // 3x640x640 i8
 %padded_input = tensor.insert_slice %quant_out into %fill_const [0, 1, 1] ...
                 // tensor<3x640x640xi8> into tensor<3x642x642xi8>

 %0 = "inf_cap.conv2d"(%padded_input, %weight) <{pad_h = 0, pad_w = 0, ...}>

[after]
 %0 = "inf_cap.conv2d"(%quant_out, %weight) <{pad_h = 1, pad_w = 1, ...}>
```

---

## 1.2 VailPadDepthwiseConvPattern

- trigger : `inf_cap.depthwise_conv2d`
- depthwise conv는 IREE에서 CHW layout으로 padding된 후 NHWC transpose generic을 거쳐 입력됨
- 체인 구조: `insert_slice → expand_shape → linalg.generic(CHW→NHWC transpose) → depthwise_conv2d`

**매칭 체인:**
1. `depthwise_conv2d.input` ← `linalg.generic` (passthrough: `yield %in`)
2. `generic.input[0]` ← `tensor.expand_shape`
3. `expand_shape.src` ← `tensor.insert_slice`
4. `insert_slice.offsets[1,2]` → pad_h, pad_w 추출

**변환 동작:**
- unpadded source에 새 `expand_shape` 생성 (output shape에서 pad 크기 제거)
- 새 `linalg.generic` 생성 (축소된 output type, 동일 indexing_maps/iterator_types)
- `depthwise_conv2d`의 pad_h/pad_w 속성 업데이트, input 교체

```
[before]
 %fill   = linalg.fill ins(%cst : i8) outs(%empty : tensor<128x82x82xi8>)
 %padded = tensor.insert_slice %src into %fill [0, 1, 1] [128, 80, 80] [1,1,1]
           // tensor<128x80x80xi8> into tensor<128x82x82xi8>
 %exp    = tensor.expand_shape %padded [[0],[1,2],[3]]
           -> tensor<128x1x82x82xi8>
 %nhwc   = linalg.generic {yield %in} ins(%exp) outs(...)
           -> tensor<1x82x82x128xi8>
 %out    = "inf_cap.depthwise_conv2d"(%nhwc, %w)
           <{pad_h = 0, pad_w = 0, stride_h = 2, stride_w = 2, ...}>

[after]
 %exp2   = tensor.expand_shape %src [[0],[1,2],[3]]
           -> tensor<128x1x80x80xi8>
 %nhwc2  = linalg.generic {yield %in} ins(%exp2) outs(...)
           -> tensor<1x80x80x128xi8>
 %out    = "inf_cap.depthwise_conv2d"(%nhwc2, %w)
           <{pad_h = 1, pad_w = 1, stride_h = 2, stride_w = 2, ...}>
```

YOLOv10 검증 결과: depthwise pad=1 정상 반영, 입력 텐서 크기 `82x82 → 80x80` 축소 확인 ✅

---

## 1.3 VailPadPoolingMaxPattern

- trigger : `inf_cap.pooling_max`
- pooling은 CHW layout으로 padding된 후 expand_shape으로 NCHW로 변환되어 입력됨
- 체인 구조: `insert_slice → expand_shape → pooling_max`

**매칭 체인:**
1. `pooling_max.input` ← `tensor.expand_shape`
2. `expand_shape.src` ← `tensor.insert_slice`
3. `insert_slice.offsets[1,2]` → pad_h, pad_w 추출

**변환 동작:**
- unpadded source에 새 `expand_shape` 생성 (output shape에서 pad 크기 제거)
- `pooling_max`의 pad_h/pad_w 속성 업데이트, input 교체

```
[before]
 %fill   = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x24x24xf32>)
 %padded = tensor.insert_slice %src into %fill [0, 2, 2] [128, 20, 20] [1,1,1]
           // tensor<128x20x20xf32> into tensor<128x24x24xf32>
 %exp    = tensor.expand_shape %padded [[0,1],[2],[3]]
           -> tensor<1x128x24x24xf32>
 %out    = "inf_cap.pooling_max"(%exp)
           <{kernel_h = 5, kernel_w = 5, pad_h = 0, pad_w = 0, ...}>

[after]
 %exp2   = tensor.expand_shape %src [[0,1],[2],[3]]
           -> tensor<1x128x20x20xf32>
 %out    = "inf_cap.pooling_max"(%exp2)
           <{kernel_h = 5, kernel_w = 5, pad_h = 2, pad_w = 2, ...}>
```

YOLOv10 검증 결과: pooling pad=2 정상 반영, 입력 텐서 크기 `24x24 → 20x20` 축소 확인 ✅
