add_llvm_library(MLIRIntelQuantOps
  IR/IntelQuantOps.cpp
  IR/DialectRegistration.cpp
  Transforms/LowerUniformRealMath.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/IntelQuantOps
  )
add_dependencies(MLIRIntelQuantOps
                 MLIRIntelQuantOpsIncGen
                 MLIRQuantOps
                 MLIRIR
                 MLIRPass
                 MLIRSupport
                 MLIRStandardOps)
