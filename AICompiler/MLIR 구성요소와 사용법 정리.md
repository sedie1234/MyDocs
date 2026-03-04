# 1. Dialect 하위 구성요소

## 1.1 operation
- Dialect를 구성하는 연산들, 일반 기계어의 opcode와 동일한 역할을 수행
- Dialect를 정의하는 것은 Dialect를 구성하는 operation들을 정의하는 것과 같다고 할 수 있음
### 1.1.1 구성요소
- Mnemonic : Dialect 내에서 Op를 식별하는 고유 이름 (ex : myIR.dma_copy)
- Operands : 일반 기계어의 operand와 같은 개념. 연산의 입력값. (ex : %input, %weight)
- Results : 연산의 결과값. (ex : %output)
- Regions : Op 내부에 포함된 하위 코드 블록 (ex : 반복문의 경우, scf.for 내부의 연산들)
- [Attribute](#Attribute) : 속성, 컴파일타임에 결정되는 상수 값으로 각 연산의 설정값과 같은 역할
- [Interface](#Interface) : Operation이 할 수 있는 행동을 나타냄
- Builders : rewriter.create\<MyOp\>(...)를 호출할 때, arguments와 attributes 등을 어떻게 초기화할 지 정의. 
  - 구현 예시
```
let builders = [
  OpBuilder<(ins "Value":$src, "Value":$dst), [{
    build($_builder, $_state, src, dst, $_builder.getI64IntegerAttr(0));
  }]>
];
```
- Verifiers : 연산이 올바른지 검사하는 규칙 유무, \[let hasVerifiler = 1;\]로 설정하고, 검증하는 함수 LogicalResult verify()를 구현해야 함.

### 1.1.2 Operation 정의 예시
- tablegen file에서 구현하는 것이 편리
```
def HW1IR_DmaCopyOp : HW1IR_Op<"dma_copy", [
  Pure,                        // Trait: 부수효과 없음
  hw1ir_MemcpyInterface        // Interface: 복사 인터페이스 구현
]> {
  let summary = "메모리 간 데이터 복사 연산";

  let arguments = (ins
    AnyMemRef:$src,            // 입력 메모리
    AnyMemRef:$dst,            // 출력 메모리
    I64Attr:$size              // 복사할 크기 (상수)
  );

  let results = (outs
    HW1IR_AsyncToken:$token    // 비동기 실행을 위한 토큰
  );

  // 인터페이스 메서드 구현
  let extraClassDeclaration = [{
    Value getSrcMemref() { return getSrc(); }
    Value getDstMemref() { return getDst(); }
  }];
}

** Pure : 해당 연산이 메모리에 대해 Read, Write, Alloc, Free 중 어떤 효과도 발생시키지 않음을 나타내는 속성
```
## 1.2 Attribute
- 연산에 부착되는 메타데이터, 런타임 중에 값이 변하지 않음
- 메모리 정렬정보나 디바이스정보, 타일링크기 등 다양한 상수를 저장 가능
### 1.2.1 속성 정의
- 기본적으로 정의된 속성 : I32Attr, I64Attr 등
- Enum 속성 정의
```
// 1. 케이스 정의
def HW1IR_DMA_READ  : I32EnumAttrCase<"Read", 0>;
def HW1IR_DMA_WRITE : I32EnumAttrCase<"Write", 1>;

// 2. Enum 속성 본체 정의
def HW1IR_DmaModeAttr : I32EnumAttr<"DmaMode", "DMA 전송 모드",
  [HW1IR_DMA_READ, HW1IR_DMA_WRITE]> {
  let cppNamespace = "::keti::hw1ir";
}
```
- 커스텀 속성 정의
```
// 1. 다이얼렉트 내에 AttrDef 정의
def HW1IR_MemoryConfigAttr : AttrDef<HW1IR_Dialect, "MemoryConfig"> {
  // 2. IR에서 불릴 이름 (예: #hw1ir.mem_config<...>)
  let mnemonic = "mem_config";

  // 3. 속성이 가질 파라미터(구성 요소) 정의
  let parameters = (ins
    "int64_t":$tile_size,
    "bool":$is_cached
  );

  // 4. IR에서 어떻게 보일지 형식 지정
  let assemblyFormat = "`<` $tile_size `,` $is_cached `>`";
}
```

### 1.2.2 활용
- 주로 operation 정의에서 데이터의 속성으로 사용함
```
def HW1IR_LoadOp : HW1IR_Op<"load"> {
  let arguments = (ins
    AnyMemRef:$input,
    HW1IR_MemoryConfigAttr:$config // 위에서 정의한 커스텀 속성
  );
}
```

## 1.3 Interface
- operation의 정체를 몰라도 특정 행동을 수행할 수 있도록 약속한 규격
- 특정 행동을 수행하는 operation들을 분별하여 변환을 정의하거나 공통된 메소드를 호출하는 형식으로 활용 가능

### 1.3.1 Interface 정의
```
def MyIR_MemcpyInterface : OpInterface<"MemcpyInterface"> {
  let description = "메모리 복사 기능을 가진 연산들을 위한 인터페이스";
  let cppNamespace = "::publisher::myir";

  let methods = [
    // 메서드 정의
    InterfaceMethod<
      "설명", 
      "반환타입", "메서드이름", 
      (ins "인자타입":$인자이름),
      [{}], // Interface 클래스 내 본문 (보통 비움)
      [{}]  // 기본 구현 (Default Implementation, 선택사항)
    >
  ];
}
```
- 해당 Interface를 Operation에 인용하면, Interface가 제공하는 함수를 반드시 제공해야 함. 기본적으로 Interface에 정의가 되어 있으나, 없는 경우에는 Operation 정의 또는 Builder에 함수들을 정의해 둬야 함
```
def MyIR_PowerAnalysisInterface : OpInterface<"PowerAnalysisInterface"> {
  let description = [{
    가속기 내의 각 연산이 소비하는 예상 전력량을 
    계산하기 위한 인터페이스입니다.
  }];

  let cppNamespace = "::publisher::myir";

  let methods = [
    // 1. 기본 구현이 없는 메서드 (Op에서 반드시 로직을 짜야 함)
    InterfaceMethod<"연산의 기본 소비 전력을 반환 (Watt 단위)",
      "float", "getBasePower">,

    // 2. 기본 구현이 있는 메서드 (따로 안 짜면 이 로직을 사용)
    InterfaceMethod<"데이터 크기에 따른 가중 전력 계산",
      "float", "getTotalPower", (ins "float":$dataScale), [{}], [{
        // $_op를 통해 현재 연산의 메서드에 접근 가능
        return $_op.getBasePower() * dataScale;
      }]
    >
  ];
}
```

### 1.3.2 활용
- operation에 인용
```
def MyIR_DmaOp : MyIR_Op<"dma", [MyIR_MemcpyInterface]> {
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst, I64Attr:$size);

  // 인터페이스 메서드 실제 로직 연결
  let extraClassDeclaration = [{
    ::mlir::Value getSrc() { return getOperand(0); }
  }];
}
```

- 변환에 활용
```
void myOptimizationPass(Operation *op) {
    // 1. 해당 연산이 인터페이스를 가지고 있는지 확인 (Cast)
    if (auto memcpyOp = llvm::dyn_cast<publisher::myir::MemcpyInterface>(op)) {
        
        // 2. Op의 실제 종류(DmaOp인지 다른 것인지) 몰라도 메서드 호출 가능
        mlir::Value source = memcpyOp.getSrc();
        
        // 3. 공통 로직 수행
        optimizeTransfer(source);
    }
}
```

# 2. Pass 하위 구성요소

## 2.1 구성요소
- RewritePattern : 특정 Op를 찾아 새로운 Op로 교체하는 최소 실행단위 (일반적으로 MatchAndRewrite로 진행)
- RewritePatternSet : Pass에서 사용할 여러 Pattern들을 담는 객체
- PatternRewriter : IR을 안전하게 수정해주는 도구 (Op 생성, 삭제, 교체 등을 담당)
- GreedyPatternRwriteDriver : 등록된 Pattern들을 반복적으로 적용하여 해당 Op가 없을때까지 변환을 수행하는 엔진

## 2.2 Pass 구현

### 2.2.1 RewritePattern 정의
- 특정 Op를 중심으로 mlir코드 내에 location을 기점으로 pattern을 분석 및 match한 뒤, operation을 생성하거나 삭제 또는 교체하는 식으로 IR을 변화시킴
- class로써 정의 (structure도 괜찮은 것으로 아는데, 실험해보지는 않음)
- 예시
```
class LinalgMatmulToHW1Pattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                  PatternRewriter &rewriter) const override {

        Value inputA = matmulOp.getInputs()[0];
        Value inputB = matmulOp.getInputs()[1];
        Value outputC = matmulOp.getOutputs()[0];
        
        rewriter.replaceOpWithNewOp<keti::hw1ir::MatmulOp>(
            matmulOp,
            matmulOp.getResult(0).getType(),
            inputA,
            inputB
        );
  
        return success();
    }
}
```

### 2.2.2 Pass 생성 및 Pattern 등록
- Pass 생성 - tablegen  : 어떤 Pass들을 생성할 지 tablegen으로 기록하면 컴파일러 빌드 시, pass creator와 pass register를 생성해줌
```
[Passes.td]

#ifndef HW1IR_CONVERSION_PASSES
#define HW1IR_CONVERSION_PASSES
  
include "mlir/Pass/PassBase.td"
  
def LinalgToHW1 : Pass<"linalg-to-hw1", "ModuleOp">{
    let summary = "Convert Linalg operations to HW1IR dialect";
    let constructor = "keti::hw1ir::createLinalgToHW1Pass()";
    let description = [{
        This pass converts Linalg operations to HW1IR dialect operations.
    }];
}
```
- Pass 생성 - Pass 헤더 : create를 포함시켜 줌. 이름 포맷은 Passes.td에 def로 정의한 이름과 맞춰야 함.
```
[ConvertToHW1IRPass.h]

#ifndef CONVERT_TO_HW1IR_H
#define CONVERT_TO_HW1IR_H

#include "HW1/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>
  
namespace keti {
namespace hw1ir {

std::unique_ptr<mlir::Pass> createLinalgToHW1Pass();
    
} // namespace hw1ir
} // namespace keti
  
#endif // CONVERT_TO_HW1IR_H
```
- Pass 생성 - RewritePattern : class로써 Pattern을 정의, class를 만들 때 사용한 argument Op를 통해 argument Op에 해당하는 모든 location을 가져옴. matchAndRewrite에서 패턴과 변환규칙을 정의
```
[ConvertToHW1IRPass.cpp]

class LinalgMatmulToHW1Pattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  
    LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                  PatternRewriter &rewriter) const override {
  
        Value inputA = matmulOp.getInputs()[0];
        Value inputB = matmulOp.getInputs()[1];
        Value outputC = matmulOp.getOutputs()[0];
        rewriter.replaceOpWithNewOp<keti::hw1ir::MatmulOp>(
            matmulOp,
            matmulOp.getResult(0).getType(),
            inputA,
            inputB
        );

        return success();
    }
}
```
- Pass 생성 - Pass 구조체 : 이름은 Passes.td에 정의한 이름과 맞춰서 작성, runOnOperation에서 패턴들을 등록, 이외에도 dialect나 operations에 대해 legal, illegal을 선언하거나 메소드 선언 등을 수행할 수도 있음.
```
[ConvertToHW1IRPass.cpp]

struct LinalgToHW1Pass
    : public keti::hw1ir::impl::LinalgToHW1Base<LinalgToHW1Pass> {
    LinalgToHW1Pass() = default;
    LinalgToHW1Pass(const LinalgToHW1Pass &pass) {}

    void runOnOperation() override;
}

...

void LinalgToHW1Pass::runOnOperation() {
    auto module = getOperation();
    auto context = module.getContext();
  
    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    RewritePatternSet patterns(context);
    // Add patterns to convert Linalg ops to HW1IR ops here
    patterns.add<LinalgMatmulToHW1Pattern>(context);
  
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
    }
  
    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));
}

```
- 이렇게 생성된 Pass들은 Passes.h와 Passes.cpp에서 등록
```
[Passes.h]

#ifndef HW1IR_CONVERSION_PASSES_H
#define HW1IR_CONVERSION_PASSES_H

#include "HW1/Conversion/ConvertToHW1IRPass.h"

namespace keti {
namespace hw1ir {
  
void registerConversionPasses();
  
} // namespace hw1ir
} // namespace keti
  
#endif // HW1IR_CONVERSION_PASSES_H

----------------------------------------------------------
[Passes.cpp]

#include "HW1/Conversion/Passes.h"
  
namespace {
#define GEN_PASS_REGISTRATION
#include "HW1/Conversion/Passes.h.inc"
}
void keti::hw1ir::registerConversionPasses() { ::registerPasses(); }

```