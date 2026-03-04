- OpFusionPass의 뒷처리를 한다. : 앞선 규칙을 모두 적용하고 남은 1:1 op를 변환한다. 예시로, 일반 mul이나 add가 있다. 

# 1. Patterns


## 1.1 IREEAddConversionPattern
- 남아있는 Add를 CAP Layer의 Add로 변환한다.
- 가장 마지막에 남아있는 줄을 변환하므로 benefit은 1을 준다.
- benefit=1
- 예시
```
[before]
	%0 = arith.addi %arg0, %arg1 : i32
	%1 = arith.addf %arg2, %arg3 : f32
	
[after]
	%0 = "inf_cap.add"(%arg0, %arg1) : (i32, i32) -> i32 
	%1 = "inf_cap.add"(%arg2, %arg3) : (f32, f32) -> f32
```

## 1.2 IREEMulConversionPattern
- 남아있는 Mul를 CAP Layer의 Mul로 변환한다.
- 가장 마지막에 남아있는 줄을 변환하므로 benefit은 1을 준다.
- benefit=1
- 예시
```
[before]
	%0 = arith.muli %arg0, %arg1 : i32
	%1 = arith.mulf %arg2, %arg3 : f32
	
[after]
	%0 = "inf_cap.mul"(%arg0, %arg1) : (i32, i32) -> i32 
	%1 = "inf_cap.mul"(%arg2, %arg3) : (f32, f32) -> f32
```
