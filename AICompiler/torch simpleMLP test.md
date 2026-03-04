- Linear(Matmul)와 ReLU로 구성된 간단한 1 layer를 만들어 컴파일 해본다.
- torch mlir과 stableHLO로 변환시키고, 이를 iree 정규식으로 가져온 뒤 customIR까지 변환시켜본다. (torch mlir + stable HLO -> iree : 완료  |  iree -> customIR : X)
- 최신 버전의 torch mlir을 사용하는 경우, torch_mlir 내부의 함수를 제대로 가져오지 못하는 현상 -> 20231229 버전을 받아서 사용
```
pip install torch-mlir==20231229.1067 -f https://llvm.github.io/torch-mlir/package-index
```

- torch mlir의 경우에는 iree로 import 되었을 때 이미 hal dialect를 호출하여 메모리의 계획이 잡혀있지만, StabelHLO는 iree로 import 되었을 때 아직 실행계획이 잡혀있는 상태. -> StableHLO를 통하는 방식을 선호

## 1. 모델 정의

- Linear와 ReLU로 표현한 SimpleMLP 1layer
```
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.fc(x))
  
model = SimpleMLP().eval()
example_input = torch.randn(1, 128)
```

## 2. torch mlir로 컴파일
### 2.1 컴파일 방법
```
# 컴파일
module = torch_mlir.compile(
	model,
	[example_input],
	output_type=torch_mlir.OutputType.TORCH
)
# 파일 저장
with open("mlp_torch.mlir", "w") as f:
    f.write(str(module))
```

### 2.2 컴파일 결과
```
module attributes {torch.debug_module_name = "SimpleMLP"} {
  func.func @forward(%arg0: !torch.vtensor<[1,128],f32>) -> !torch.vtensor<[1,64],f32> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %0 = torch.vtensor.literal(dense<[0.0504985414, ... , -0.0699762106]> : tensor<64xf32>) : !torch.vtensor<[64],f32>
    %1 = torch.vtensor.literal(dense<"0xF85...13D"> : tensor<64x128xf32>) : !torch.vtensor<[64,128],f32>
    %2 = torch.aten.transpose.int %1, %int0, %int1 : !torch.vtensor<[64,128],f32>, !torch.int, !torch.int -> !torch.vtensor<[128,64],f32>
    %3 = torch.aten.mm %arg0, %2 : !torch.vtensor<[1,128],f32>, !torch.vtensor<[128,64],f32> -> !torch.vtensor<[1,64],f32>
    %4 = torch.aten.add.Tensor %3, %0, %float1.000000e00 : !torch.vtensor<[1,64],f32>, !torch.vtensor<[64],f32>, !torch.float -> !torch.vtensor<[1,64],f32>
    %5 = torch.aten.relu %4 : !torch.vtensor<[1,64],f32> -> !torch.vtensor<[1,64],f32>
    return %5 : !torch.vtensor<[1,64],f32>
      }
}
```


## 3. StableHLO로 컴파일
### 3.1 컴파일 방법
```
# 컴파일
hlo_module = torch_mlir.compile(model, [example_input], output_type="stablehlo")
# 파일 저장
with open("mlp_hlo.mlir", "w") as f:
    f.write(str(hlo_module))
```

### 3.2 컴파일 결과
```
module attributes {torch.debug_module_name = "SimpleMLP"} {
  func.func @forward(%arg0: tensor<1x128xf32>) -> tensor<1x64xf32> {
    %0 = stablehlo.constant dense<[0.0504985414, ... , -0.0699762106]> : tensor<64xf32>
	%1 = stablehlo.constant dense<"0xF85...13D"> : tensor<64x128xf32>
	%2 = chlo.constant dense<0.000000e+00> : tensor<1x64xf32>
    %3 = stablehlo.transpose %1, dims = [1, 0] : (tensor<64x128xf32>) -> tensor<128x64xf32>
    %4 = stablehlo.dot %arg0, %3 : (tensor<1x128xf32>, tensor<128x64xf32>) -> tensor<1x64xf32>
    %5 = chlo.broadcast_add %4, %0 : (tensor<1x64xf32>, tensor<64xf32>) -> tensor<1x64xf32>
    %6 = stablehlo.maximum %5, %2 : tensor<1x64xf32>
    return %6 : tensor<1x64xf32>
  }
}
    
```

## 4. torch to iree
```
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "SimpleMLP"} {
  util.func public @forward$async(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %cst = arith.constant dense<[0.0504985414, ... , -0.0699762106]> : tensor<64xf32>
    %cst_0 = arith.constant dense<"0xF85...13D">: tensor<64x128xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<1x128xf32>
    %1 = tensor.empty() : tensor<128x64xf32>
    %transposed = linalg.transpose ins(%cst_0 : tensor<64x128xf32>) outs(%1 : tensor<128x64xf32>) permutation = [1, 0]
    %2 = tensor.empty() : tensor<1x64xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %4 = linalg.matmul ins(%0, %transposed : tensor<1x128xf32>, tensor<128x64xf32>) outs(%3 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %cst : tensor<1x64xf32>, tensor<64xf32>) outs(%2 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.addf %in, %in_2 : f32
      linalg.yield %9 : f32
    } -> tensor<1x64xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x64xf32>) outs(%2 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.cmpf ugt, %in, %cst_1 : f32
      %10 = arith.select %9, %in, %cst_1 : f32
      linalg.yield %10 : f32
    } -> tensor<1x64xf32>
    %7 = hal.tensor.barrier join(%6 : tensor<1x64xf32>) => %arg2 : !hal.fence
    %8 = hal.tensor.export %7 : tensor<1x64xf32> -> !hal.buffer_view
    util.return %8 : !hal.buffer_view
  }
  util.func public @forward(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1 = util.call @forward$async(%arg0, %0, %fence) : (!hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.return %1 : !hal.buffer_view
  }
}    
```

## 5. StableHLO to iree
```
#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "SimpleMLP"} {
  func.func @forward(%arg0: tensor<1x128xf32>) -> tensor<1x64xf32> {
    %0 = shape.const_witness true
    %cst = arith.constant 0.000000e+00 : f32
	%cst_0 = arith.constant dense<[0.0504985414, ... , -0.0699762106]> : tensor<64xf32>
    %cst_1 = arith.constant dense<"0xF85...13D">: tensor<64x128xf32>
	%1 = tensor.empty() : tensor<128x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<64x128xf32>) outs(%1 : tensor<128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x64xf32>
    %3 = tensor.empty() : tensor<1x64xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %5 = linalg.matmul ins(%arg0, %2 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%4 : tensor<1x64xf32>) -> tensor<1x64xf32>
    %6 = shape.assuming %0 -> (tensor<1x64xf32>) {
	%9 = tensor.empty() : tensor<1x64xf32>
	%10 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<64xf32>) outs(%9 : tensor<1x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<1x64xf32>
	%11 = tensor.empty() : tensor<1x64xf32>
	%12 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%5, %10 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%11 : tensor<1x64xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
	%13 = arith.addf %in, %in_2 : f32
        linalg.yield %13 : f32
      } -> tensor<1x64xf32>
      shape.assuming_yield %12 : tensor<1x64xf32>
    }
    %7 = tensor.empty() : tensor<1x64xf32>
    %8 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<1x64xf32>) outs(%7 : tensor<1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.maximumf %in, %cst : f32
      linalg.yield %9 : f32
    } -> tensor<1x64xf32>
    return %8 : tensor<1x64xf32>
  }
}
```