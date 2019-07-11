# Support.
add_llvm_library(MLIRQuantizerSupport
  Support/Configuration.cpp
  Support/ConstraintAnalysisGraph.cpp
  Support/Metadata.cpp
  Support/Statistics.cpp
  Support/TypeUtils.cpp
  Support/UniformConstraints.cpp
  Support/UniformSolvers.cpp

  ADDITIONAL_HEADER_DIRS
  )
add_dependencies(MLIRQuantizerSupport
                 MLIRIR
                 MLIRQuantOps
                 MLIRSupport
                 MLIRStandardOps)

# Configurations.
add_llvm_library(MLIRQuantizerFxpMathConfig
  Configurations/FxpMathConfig.cpp

  ADDITIONAL_HEADER_DIRS
  )
add_dependencies(MLIRQuantizerFxpMathConfig
                 MLIRQuantizerSupport)

add_llvm_library(MLIRQuantizerIntelQuantConfig
  Configurations/IntelQuantConfig.cpp

  ADDITIONAL_HEADER_DIRS
  )
add_dependencies(MLIRQuantizerIntelQuantConfig
                 MLIRQuantizerSupport)
# Transforms.
add_llvm_library(MLIRQuantizerTransforms
  Transforms/AddDefaultStatsTestPass.cpp
  Transforms/InferQuantizedTypesPass.cpp
  Transforms/InferQuantizedTypesIntelPass.cpp
  Transforms/RemoveInstrumentationPass.cpp
  Transforms/Conv2DFusionPass.cpp

  ADDITIONAL_HEADER_DIRS
  )
add_dependencies(MLIRQuantizerTransforms
  MLIRQuantizerFxpMathConfig
  MLIRQuantizerIntelQuantConfig
  MLIRQuantizerSupport
  MLIRPass)
target_link_libraries(MLIRQuantizerTransforms
  MLIRQuantizerFxpMathConfig
  MLIRQuantizerIntelQuantConfig
  MLIRQuantizerSupport
  MLIRPass)
