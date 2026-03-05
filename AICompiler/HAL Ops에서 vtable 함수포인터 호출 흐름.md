# 1. 개요
- 컴파일러에서 HAL Ops에 대해 vm symbol을 포함하여 vm bytecode를 생성
- 런타임은 이를 실행하기 위해 vtable의 함수포인터를 vm symbol에 연결되도록 구성
- 함수 포인터에 개인 함수를 연결함으로써 vm bytecode에 기록된 HAL Ops를 실행하는 방식
- 그런데, 아래 예시처럼 vtable 함수포인터의 이름과 vm symbol의 이름이 1:1 매칭도 안될 뿐더러 일정한 규칙도 없음
- 예시) HAL Ops에서 vm symbol까지는 hal.allocator.allocate 형태로 생성되도록 되어 있음. rewritepattern까지 확인하면 더 명확함
![Pasted image 20260123132615](https://imgs.hwan123.synology.me:19999/AICompiler/Pasted image 20260123132615.png)
	하지만 vtable을 보면 매칭되는 함수가 없음. allocate_buffer가 그나마 매칭이 되지만 일정한 규칙이 없으므로 어디선가 그 규칙을 만들고 있을 것.
![Pasted image 20260123132740](https://imgs.hwan123.synology.me:19999/AICompiler/Pasted image 20260123132740.png)
- 때문에 HAL Ops와 vtable 함수포인터의 매칭이 명확하지 않고, 심지어는 하나의 HAL Ops에 여러 vtable에 정의된 함수가 여러 개 호출되는 경우도 있을 수 있음
- 이 의존성을 명확하게 정의할 수 있어야 사용할 수 있는 HAL Ops 및 구현해야 할 vtable의 list를 명확하게 작성할 수 있음.
- 그러므로 HAL Ops에서 vtable까지의 흐름을 분석 (자세한 분석은 생략하고, 그 의존성을 파악할 수 있는 코드들만 분석


# 2. vm symbol에서 vtable 함수포인터 호출 흐름
## 2.1 vtable 객체 호출
- vtable도 그룹을 나누어 그룹마다 다른 구조체를 활용한다. 
- vm symbol로부터 적절한 구조체 호출의 정의
- iree/runtime/iree.natvis에 정의되어 있다. 이것을 runtime 실행 도중 읽어서 매칭함
![Pasted image 20260123133441](https://imgs.hwan123.synology.me:19999/AICompiler/Pasted image 20260123133441.png)

## 2.2 각 그룹의 function 호출
- 각 그룹에는 여러 function들이 존재하는데, vm symbol에 나타나는 각 그룹의 함수 이름은 HAL Ops와 매칭되어 있다. 
- 이 function들로부터 vtable의 함수포인터를 매칭하는 방법
- iree/runtime/src/iree/modules/hal/exports.ini에 아래와 같이 vm symbol에 따라 이어지는 함수가 있다. (뒤의 두 arguments는 함수의 arguments와 results의 포맷을 약자로 나타낸 것)
![Pasted image 20260123133937](https://imgs.hwan123.synology.me:19999/AICompiler/Pasted image 20260123133937.png)
- 여기에서 allocator.allocate를 따라가보면
![Pasted image 20260123134157](https://imgs.hwan123.synology.me:19999/AICompiler/Pasted image 20260123134157.png)
![Pasted image 20260123134209](https://imgs.hwan123.synology.me:19999/AICompiler/Pasted image 20260123134209.png)
![Pasted image 20260123134237](https://imgs.hwan123.synology.me:19999/AICompiler/Pasted image 20260123134237.png)
- 위와 같이 매크로를 통해 함수가 생성되도록 되어 있음
- 위의 allocator.allocate에서는 allocator.allocate 구현을 위해 필요한 함수들이 진행되고, _VTABLE_DISPATCH를 통해 iree_hal_allocator_vtable_t의 allocate_buffer 함수포인터가 할당된다.
