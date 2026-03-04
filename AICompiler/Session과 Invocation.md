- compile 수행을 session과 Invocation으로 나누어, session은 실행 환경을 관리 invocation은 해당 환경에서 실제 작업을 수행
- session과 Invocation 사이에서 wrap과 unwrap 동작을 통해 값이나 포인터를 동기화
# 1. Session

- 컴파일러에서
	- 컴파일 작업을 수행하기 위한 설정 및 환경을 관리
	- 컴파일 옵션을 포함하여 input / output 등을 관리하고 있음
- 런타임에서
	- 컴파일된 바이너리를 실행하는 시점에서, 메모리와 리소스 환경을 관리

# 2. Invocation
- 컴파일러에서
	- 컴파일 파이프라인을 실제로 수행, Invocation 한 번은 파이프라인 한 번 수행하는 것을 의미
	- 컴파일 옵션에 따라 실제로 IR 변환을 적용함
- 런타임에서
	- predict나 run_module / run_function 등 하나의 함수를 수행
