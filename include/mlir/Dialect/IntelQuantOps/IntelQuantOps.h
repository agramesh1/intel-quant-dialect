//===- IntelQuantOps.h - Fixed point ops ---------------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef MLIR_DIALECT_INTELQUANTOPS_INTELQUANTOPS_H_
#define MLIR_DIALECT_INTELQUANTOPS_INTELQUANTOPS_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace intelquant {

/// Defines the 'IntelQuantOps' dialect.
class IntelQuantOpsDialect : public Dialect {
public:
  IntelQuantOpsDialect(MLIRContext *context);
};

#define GET_OP_CLASSES
#include "mlir/Dialect/IntelQuantOps/IntelQuantOps.h.inc"

} // namespace intelquant
} // namespace mlir

#endif // MLIR_DIALECT_INTELQUANTOPS_INTELQUANTOPS_H_
