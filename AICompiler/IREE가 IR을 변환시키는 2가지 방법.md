- IREE는 IR을 변환시키는 방법으로 2가지 방법을 모두 사용하고 있다.

## 1. PASS를 이용한 방법
- MLIR의 정석적인 방법으로, rewritePattern을 정의하고, Pattern을 Conversion 및 Pass에 등록 후 runOnOperator에서 변환을 적용하는 방법
- IREE에 있는 Canonicalize.cpp를 예시로 Pass를 이용하는 예시를 보여줌
- 1) rewritePattern 정의
  -> 아래 처럼 정의, structure 선언에 사용된 argument를 이용하여 해당 operation에 해당하는 location을 가져올 수 있음
  -> matchAndRewrite 함수에서 location을 중심으로 pattern을 찾고, 새로운 operation으로 rewrite하는 규칙을 정의
```
struct FoldFullInsertSlice : public OpRewritePattern<tensor::InsertSliceOp> {
  using Base::Base;
  
  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!insertSliceOp.hasUnitStride() || !insertSliceOp.hasZeroOffset()) {
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "non-unit stride or non-zero offset.");
    }
  
    RankedTensorType sourceType = insertSliceOp.getSourceType();
    RankedTensorType resultType = insertSliceOp.getResultType();
    if (sourceType != resultType) {
      return rewriter.notifyMatchFailure(
          insertSliceOp,
          "unimplemented: Cast-like or reshape-like insert ops.");
    }
  
    std::optional<SmallVector<OpFoldResult>> mixedSizes =
        getDefiningMixedSizes(insertSliceOp.getDest());
        
    if (!mixedSizes) {
      return rewriter.notifyMatchFailure(
          insertSliceOp, "Could not find producer with list of tensor sizes.");
    }
  
    for (auto [insertSize, destSize] :
         llvm::zip_equal(insertSliceOp.getMixedSizes(), mixedSizes.value())) {
      if (isa<Value>(insertSize) || isa<Value>(destSize)) {
        if (insertSize != destSize) {
          return rewriter.notifyMatchFailure(insertSliceOp,
                                             "dynamic size mismatch");
        }
        continue;
      }
  
      // `getMixedSizes` for different ops returns different attribute types
      // (`index` or `i64`) so we compare the values of the ints directly here.
      int64_t staticInsertSize = getConstantIntValue(insertSize).value();
      int64_t staticDestSize = getConstantIntValue(insertSize).value();
      if (staticInsertSize != staticDestSize) {
        return rewriter.notifyMatchFailure(insertSliceOp,
                                           "static size mismatch");
      }
    }
  
    rewriter.replaceOp(insertSliceOp, insertSliceOp.getSource());
    return success();
  }
};  
```

- 2) rewrite pattern을 Pass에 추가
  -> Pass Structure에서 Initialize에 Pattern을 추가
```
struct CanonicalizePass : public impl::CanonicalizePassBase<CanonicalizePass> {
  using IREE::Flow::impl::CanonicalizePassBase<
      CanonicalizePass>::CanonicalizePassBase;
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Inherit the same config defaults from the upstream canonicalizer pass.
    config.setUseTopDownTraversal().setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Normal);
  
    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);
  
    // Pull in some borderline/downstream canonicalizations for the Flow
    // compilation phase.
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(owningPatterns);
    owningPatterns.add<FoldFullInsertSlice>(context);
    owningPatterns.add<AffineApplyLowering>(context);
  
    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }
...
};

void mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(
    RewritePatternSet &patterns) {
  patterns.add<MergeConsecutiveExtractSlice,
               MergeConsecutiveInsertSlice<InsertSliceOp>,
               MergeConsecutiveInsertSlice<ParallelInsertSliceOp>>(
      patterns.getContext());
}

```

- 3) runOnOperation으로 pattern 적용
  -> Pass Structure에 runOnOperation에서 applyPatterns를 통해 Pass가 적용되는 시점에 2)에서 추가한 패턴이 적용됨
```
struct CanonicalizePass : public impl::CanonicalizePassBase<CanonicalizePass> {
	
	...
	
  void runOnOperation() override {
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    config.enableConstantCSE(cseConstants);
    LogicalResult didConverge =
        applyPatternsGreedily(getOperation(), *patterns, config);
    if (this->testConvergence && failed(didConverge)) {
      getOperation()->emitError("Canonicalizer failed to converge");
      return signalPassFailure();
    }
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};
```

- 4) 예제 : 적용 전
```
util.func public @fold_full_insert_into_extract(
    %source: tensor<8x?xf32>,
    %dest: tensor<10x?xf32>,
    %size: index) -> tensor<8x?xf32> {
  %extract = tensor.extract_slice %dest [1, 1] [8, %size] [1, 1] : tensor<10x?xf32> to tensor<8x?xf32>
  %insert = tensor.insert_slice %source into %extract [0, 0] [8, %size] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
  util.return %insert : tensor<8x?xf32>
}
```

- 5) 예제 : 적용 후
```
module {
  util.func public @fold_full_insert_into_extract(%arg0: tensor<8x?xf32>, %arg1: tensor<10x?xf32>, %arg2: index) -> tensor<8x?xf32> {
    util.return %arg0 : tensor<8x?xf32>
  }
	...
}
```

## 2. transform Dialect를 이용한 방법
- transform Dialect를 이용하여 mlir 내부에 변환규칙을 추가하는 방법 ([링크](https://mlir.llvm.org/docs/Dialects/Transform/#overview))
- fuse_consumer 예제를 통한 기본 원리 파악
- 아래와 같은 mlir을 보면, 변환하고자하는 source IR과 변환 규칙을 기술한 transform IR이 하나의 mlir 파일에 존재
```
// RUN: iree-opt %s --iree-transform-dialect-interpreter --transform-dialect-drop-schedule | FileCheck %s
  
#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0)>
func.func @pack_consumer_fusion(%arg0: tensor<32xf32>) -> tensor<2x16xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %1 = scf.forall (%arg1) in (2) shared_outs(%arg2 = %0) -> (tensor<32xf32>) {
    %3 = affine.apply #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[%3] [16] [1] : tensor<32xf32> to tensor<16xf32>
    %extracted_slice_0 = tensor.extract_slice %arg2[%3] [16] [1] : tensor<32xf32> to tensor<16xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice : tensor<16xf32>) outs(%extracted_slice_0 : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %in : f32
      linalg.yield %5 : f32
    } -> tensor<16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %arg2[%3] [16] [1] : tensor<16xf32> into tensor<32xf32>
    }
  }
  %2 = tensor.empty() : tensor<2x16xf32>
  %pack = linalg.pack %1 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %2 : tensor<32xf32> -> tensor<2x16xf32>
  return %pack : tensor<2x16xf32>
}




// CHECK-LABEL: @pack_consumer_fusion
// CHECK:       scf.forall
// CHECK:         %[[GENERIC:.+]] = linalg.generic
// CHECK:         %[[PACK:.+]] = linalg.pack %[[GENERIC]]
// CHECK:         scf.forall.in_parallel {
// CHECK:           tensor.parallel_insert_slice %[[PACK]]
  
  
  
  
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %slice_op = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg0
    : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %arg0
    : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.iree.fuse_consumer %slice_op in (%loop)
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
     transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }
}
```
- 아래의 transform dialect에 정의된 규칙에 따라 IR을 변환, 아래는 변환결과
```
#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @pack_consumer_fusion(%arg0: tensor<32xf32>) -> tensor<2x16xf32> {
    %0 = tensor.empty() : tensor<32xf32>
    %1 = tensor.empty() : tensor<2x16xf32>
    %2 = scf.forall (%arg1) in (2) shared_outs(%arg2 = %1) -> (tensor<2x16xf32>) {
      %3 = affine.apply #map(%arg1)
      %extracted_slice = tensor.extract_slice %arg0[%3] [16] [1] : tensor<32xf32> to tensor<16xf32>
      %extracted_slice_0 = tensor.extract_slice %0[%3] [16] [1] : tensor<32xf32> to tensor<16xf32>
      %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice : tensor<16xf32>) outs(%extracted_slice_0 : tensor<16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = arith.addf %in, %in : f32
        linalg.yield %5 : f32
      } -> tensor<16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg1, 0] [1, 16] [1, 1] : tensor<2x16xf32> to tensor<1x16xf32>
      %pack = linalg.pack %4 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [16] into %extracted_slice_1 : tensor<16xf32> -> tensor<1x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %pack into %arg2[%arg1, 0] [1, 16] [1, 1] : tensor<1x16xf32> into tensor<2x16xf32>
      }
    }
    return %2 : tensor<2x16xf32>
  }
}
```
- IREE에서는 /path/to/iree/compiler/src/iree/compiler/Codegen/Common/의 TransformDialectInterpreterPass.cpp에서 transform dialect entry point를 만들어, transform dialect로 변환 규칙을 등록하면 해당지점에 변환규칙들이 추가되도록 해 둠.