- IREE는 모듈화된 구조를 최대한 활용하게 하기 위해 plugin 기능을 제공
- 해당 기능을 활용함으로써 IREE는 그대로 유지하면서 새로운 backend를 추가하거나 새로운 최적화 pass를 추가해볼 수 있다.

# 1. 사용방법

- AMD AIR-AIE 사용 예를 참고
- IREE의 PluginSession class를 상속받고, 그 안에 Pass, Dialect, Device 정보(HAL 등) 등을 입력하고, compiler_plugin 함수를 mangling하여 내보냄
- Plugin에 등록하는 Pass, Dialect, Device 관련 라이브러리들은 IREE와는 독립적으로 개발
```
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
#include "aie/AIEDialect.h"
#include "aie/AIEXDialect.h"
#include "aie/Passes.h"
#include "aievec/AIEVecDialect.h"
#include "aievec/Passes.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Passes.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Target/AIETarget.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
  
namespace mlir::iree_compiler {
namespace {
  
namespace {
#define GEN_PASS_REGISTRATION
#include "aie/Passes.h.inc"
}  // namespace
  
struct AMDAIESession
    : public PluginSession<AMDAIESession, AMDAIE::AMDAIEOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    AMDAIE::registerAMDAIEPasses();
    registerAMDAIEAssignBufferAddresses();
    AMDAIE::registerAMDAIEAssignBufferDescriptorIDs();
    registerAMDAIECoreToStandard();
    AMDAIE::registerAMDAIELocalizeLocks();
    AMDAIE::registerAMDAIENormalizeAddressSpaces();
    registerAMDAIERouteFlowsWithPathfinder();
    AMDAIE::registerAMDAIEDmaToNpu();
    AMDAIE::registerAMDAIEIncrementRepeatCount();
    AMDAIE::registerAIRConversionPasses();
    AMDAIE::registerAIRTransformPasses();
    aievec::registerConvertAIEVecToLLVMPass();
    aievec::registerAlignTransferReadsPass();
    aievec::registerCanonicalizeVectorForAIEVecPass();
    aievec::registerLowerVectorToAIEVecPass();
  }
  
  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<AMDAIE::AMDAIEDialect, xilinx::AIE::AIEDialect,
                    aievec::AIEVecDialect, xilinx::AIEX::AIEXDialect,
                    xilinx::air::airDialect>();
  }
  
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    // #hal.device.target<"xrt", ...
    targets.add("xrt", [=] {
      options.deviceHal = AMDAIE::AMDAIEOptions::DeviceHAL::XRT;
      return AMDAIE::createTarget(options);
    });
    // #hal.device.target<"xrt-lite", ...
    targets.add("xrt-lite", [=] {
      options.deviceHal = AMDAIE::AMDAIEOptions::DeviceHAL::XRT_LITE;
      return AMDAIE::createTarget(options);
    });
  }
  
  void populateHALTargetBackends(
      IREE::HAL::TargetBackendList &targets) override {
    targets.add("amd-aie", [=]() { return AMDAIE::createBackend(options); });
  }
};
  
}  // namespace
}  // namespace mlir::iree_compiler
  
IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::AMDAIE::AMDAIEOptions);
  
extern "C" bool iree_register_compiler_plugin_amd_aie(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::AMDAIESession>("amd_aie");
  return true;
}
```

- iree-amd-aie 프로젝트 구조 참고
       <img src="https://imgs.hwan123.synology.me:19999/AICompiler/iree-air-aie-architecture 2.png" width="600">


- plugin 적용 위치
![Apply_Plugin](https://imgs.hwan123.synology.me:19999/AICompiler/Apply_Plugin.png)

# 2. Plugin 동작의 이해

- [[컴파일러 적용과 mlir의 변화]]에 Plugin의 동작을 포함하여 컴파일러가 적용되는 흐름을 정리