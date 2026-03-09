- torch model이 llvm-cpu를 target으로 IREE가 적용되는 과정을 보여주면서, 새롭게 제작하는 컴파일러가 적용될 위치를 보여줌
# 1. 실험용 모델
- 단순한 MLP 1 Layer로, 아래와 같이 구성
```
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.fc(x))
```
- IREE에 넣기 위해서 IREE가 지원하는 포맷 중 하나인 StableHLO 포맷으로 컴파일
```
model = SimpleMLP().eval()
example_input = torch.randn(1, 128)

hlo_module = torch_mlir.compile(model, [example_input], output_type="stablehlo")
with open("mlp_hlo.mlir", "w") as f:
    f.write(str(hlo_module))
```
- IREE의 input으로 들어갈 mlir
![Pasted image 20260116094333](Pasted%20image%2020260116094333.png)
# 2. IREE 정규식 표현으로 변경
- IREE의 Importer부분이 적용되면 mlir은 아래와 같이 변화 
- 공식 mlir이 제공하는 dialect들로 변경된 것을 확인할 수 있음.
- iree의 정규식으로 변경 됨
![Pasted image 20260116094434](Pasted%20image%2020260116094434.png)

# 3. Stream / Flow / HAL 적용
- 연산의 순서를 정비하고, 최적화하는 패스들(Stream, Flow 계층)을 적용하고, HAL 패스를 적용하면 아래와 같이 변화
![Pasted image 20260116094859](Pasted%20image%2020260116094859.png)
- hal.executalbe.variant가 생성되고, 그 아래 Region으로 builtin.module이 생성.
- builtin.module 안에는 아직 IREE의 정규식으로 표현된 IR
- **==해당 위치가 Plugin이 적용될 위치==**
- ==**builtin.module의 내부의 IR을 hal 표현식으로 바꿔가는 것이 컴파일러 개발의 최종 목표**==

# 4. Plugin 변환 적용
- Plugin이 모두 적용되면 아래와 같이 변화
- llvm-cpu의 경우 llvmIR로 변화시키고, 시스템에 저장된 바이너리를 호출하는 형식이지만, custom HW에 대한 Plugin의 기본 전략은 hal interface로 변경시키는 것
- **==Plugin의 결과물로 원하는 것은 본래의 연산을 hal interface로 표현하고, 메타데이터를 저장하는 것을 목표로 함==**
![Pasted image 20260116100201](Pasted%20image%2020260116100201.png)

# 5. VM 생성
- builtin에 저장된 결과를 바탕으로 2가지를 생성.
- 그 중 하나는 vm의 생성
- builtin에 저장된 결과를 통해 runtime이 hal을 호출할 스캐줄을 구성해 줌.
![Pasted image 20260116101013](Pasted%20image%2020260116101013.png)

# 6. IREE의 최종 형태
- IREE의 최종 형태는 모두 vm dialect로 표시되며, 3가지 부분으로 나뉨
## 6.1 메타데이터
- 어떤 메타데이터가 어디에 있는지, 또는 어떤 값인지를 기억
![Pasted image 20260116101517](Pasted%20image%2020260116101517.png)

## 6.2 HAL Interface 정의
- 사용할 HAL Interface를 선언 및 정의함. 이는 runtime 구현의 **vtable**로 이어짐
![Pasted image 20260116102611](Pasted%20image%2020260116102611.png)

## 6.3 실제 연산을 정의
- 위에 선언한 HAL Interface와 rodata, metadata를 조합하여 연산의 실행을 구현
![Pasted image 20260116102729](Pasted%20image%2020260116102729.png)
- vm.export \<function name\>을 통해 function을 수행

## 6.4 기타 내용들
- 최종형태 코드를 보면 init이 뒤에 실행되는 것처럼 보이는데, runtime이 코드에서 init을 먼저 찾아서 수행해주는 것으로 보임
- init은 export된 이후에 정의가 나오는데, 이것이 표준이라고 함. 
![Pasted image 20260116102925](Pasted%20image%2020260116102925.png)
# 7. Runtime으로 연결
- 최종 코드는 모두 VM으로 되어 있어, Runtime은 VM code를 읽고 거기에 맞는 행동을 수행한다. 
- 일반적인 기능들(vm.rodat, vm.const, ... 등)은 Runtime이 자원과 메모리를 관리하며 적절하게 호출하도록 IREE가 구축해 뒀다.
- vm.call의 경우에는 사용자가 정의하는 함수들이 호출이 되는데, 기본적으로 HAL command들을 호출하게 된다. 

## 7.1 vm.call HAL
- Runtime이 vm.call 명령어를 실행할 순간이 되면, HAL Interface에 따라runtime의 vtable로 이동하고, vtable이 정의하는 함수가 실행되는 방식
- 기본적으로 사용할 수 있는 Interface를 IREE가 제공해주기 때문에 이 안에서 조합하는 것이 좋음
- 예시 (iree-amd-aie의 xrt 연결방식)

1) VM이 hal.device.queue.alloca 라는 인터페이스를 사용
![Pasted image 20260116134733](Pasted%20image%2020260116134733.png)

2) hal.device.queue.alloca 선언
![Pasted image 20260116134851](Pasted%20image%2020260116134851.png)

3) xrt의 vtable에서 인터페이스와 백엔드 함수 연결
![Pasted image 20260116134948](Pasted%20image%2020260116134948.png)

4) 연결된 함수 실행 : 해당함수에 queue alloca에 대한 행동이 구현되어 있음. **==device drvier나 시뮬레이션의 alloca에 해당하는 함수의 호출도 이 함수로 연결됨==**
![Pasted image 20260116135018](Pasted%20image%2020260116135018.png)