## 1. Importer
- Importer의 목적은 프레임워크 별로 서로 다르게 표현되어 있는 모델 표현식을 LLVM-MLIR 프로젝트에서 제공하는 기본 IR들 (특히, linalg 및 arith)을 이용하여 IREE의 정규식으로 표현하는 것.
- HLO도 동일한 역할을 하지만, IREE의 Middle-end로 이어지는 데 동일한 규칙을 가진 식들로 표현된다는 데 의의가 있음
- 그 구조를 보기 위해 Torch Input conversion을 확인
![Pasted image 20260107172128](images/Pasted%20image%2020260107172128.png)
![Pasted image 20260107172204](images/Pasted%20image%2020260107172204.png)
- 기본적으로 torch-mlir을 third-party로 두고 거기에 구현된 Pass들을 등록하여 IREE의 정규식이 되도록 구현
- onnx의 경우에는 onnx가 직접 IREE로 import되는 경우는 없으며, torch-mlir을 이용하여 torch의 dialect로 만든 뒤, TorchOnnxToTorch Pass를 이용하여 IREE의 정규식으로 들어오게 되어 있음.
![Pasted image 20260107172447](images/Pasted%20image%2020260107172447.png)
- IREE의 정규식으로 편입이 필요하지만 정의가 되지 않은 경우에 위 그림처럼 RewritePattern부터 Conversion - Pass까지 구현이 되어 있음.
![Pasted image 20260107172707](images/Pasted%20image%2020260107172707.png)
- onnx 변환을 예시로 보면 torch mlir에는 onnx를 torch mlir로 가져와 관리하기 위한 dialect와 operation들이 구현되어 있음. (여기에서 찾아봤을 때, QLinear가 포함된 Operation은 Conv와 Matmul뿐인 것을 확인)


## 2. Tools

- IREE는 라이브러리로서 지원하고자 함. 아직은 라이브러리로 활용하기는 부족하다고 생각됨
- IREE는 라이브러리화 시켜둔 source file들을 가져와 다양한 도구들을 제공하고 있고, 그 코드는 /path/to/iree/compiler/tools/에 모아 둠
- 가장 먼저 살펴봐야 할 것은 iree-compiler-main 코드, iree-run-module-main, iree-benchmark-module임
- 각각 import된 mlir을 compile하는 코드 / compile된 파일을 실행하는 런타임 코드 / benchmarking하는 코드다.

### 2.1 iree-compiler-main (컴파일러 코드 분석)
- import된 mlir을 compile하는 코드로, 실제로는 /path/to/iree/compiler/src/iree/compiler/Tools/iree_compiler_lib.cc : runIreecMain 함수에서 수행됨. 
- 자세한 분석은 [[컴파일러 코드 분석]]에서 진행

### 2.2 iree-run-module-main


### 2.3 iree-benchmark-module


## 3. vm bytecode와 hal executable을 생산하는 원리
- 동일 연산을 통해 hal executable과 vm bytecode를 모두 생산해 냄
	<img src="images/iree_lowering_to_binary.png" width="400">
   
- example mlir 을 변환시키고 dump 된 파일을 확인하여 코드가 변해가는 과정을 통해 검증
```
[example.mlir]
func.func @simple_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.add ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
                  outs(%arg0 : tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
```

- 변환 및 dump
```
$ ./iree-compile example.mlir \
--iree-hal-target-backends=llvm-cpu \
--mlir-print-ir-after-all \
--mlir-elide-elementsattrs-if-larger=32 \
-o test.vmfb > full_pipeline.mlir 2>&1
```

- dump 결과(full_pipeline.mlir) 및 [[iree pipeline 흐름 분석]]
[full_pipeline.mlir](full_pipeline.mlir)

