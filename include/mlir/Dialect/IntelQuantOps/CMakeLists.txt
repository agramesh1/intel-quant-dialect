set(LLVM_TARGET_DEFINITIONS IntelQuantOps.td)
mlir_tablegen(IntelQuantOps.h.inc -gen-op-decls)
mlir_tablegen(IntelQuantOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRIntelQuantOpsIncGen)
