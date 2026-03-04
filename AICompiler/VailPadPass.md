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

- trigger : tensor.insert_slice
- 1. inf_cap.conv2d를 찾음
- 2. input이 insert_slice인지 확인.
- 3. pad가 되었는지 확인하고, pad의 수치를 conv2d의 attributes로 넣음. %6, %7, %inserted_slice는 삭제하고, conv의 input은 %8로 한다. (quantize는 유지)

- 예시
```
[before]
 %padded_empty = tensor.empty() : tensor<3x642x642xi8>
 %fill_const = linalg.fill ins(%cst) outs(%padded_empty)
 %quant_out = linalg.generic { ... } // input quantize result (3x640x640)
 %padded_input = tensor.insert_slice %quant_out into %fill_const [0, 1, 1] ...

 %0 = "inf_cap.conv2d"(%padded_input, %weight) <{pad_h = 0, pad_w = 0, ...}>

[after]
 %0 = "inf_cap.conv2d"(%quant_out, %weight) <{pad_h = 1, pad_w = 1, ...}>
```
