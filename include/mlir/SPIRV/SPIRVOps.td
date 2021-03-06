//===-- SPIRVOps.td - MLIR SPIR-V Op Definitions Spec ------*- tablegen -*-===//
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
//
// This is the main operation definition specification file for SPIR-V
// operations.
//
//===----------------------------------------------------------------------===//

#ifdef SPIRV_OPS
#else
#define SPIRV_OPS

#ifdef SPIRV_BASE
#else
include "mlir/SPIRV/SPIRVBase.td"
#endif // SPIRV_BASE

def SPV_FMulOp : SPV_Op<"FMul", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Floating-point multiplication of Operand 1 and Operand 2";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
  }];

  let arguments = (ins
    SPV_ScalarOrVectorOf<AnyFloat>:$operand1,
    SPV_ScalarOrVectorOf<AnyFloat>:$operand2
  );

  let results = (outs
    SPV_ScalarOrVectorOf<AnyFloat>:$result
  );

  let parser = [{ return impl::parseBinaryOp(parser, result); }];
  let printer = [{ return impl::printBinaryOp(getOperation(), p); }];
}

#endif // SPIRV_OPS
