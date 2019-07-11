//===- IntelQuantOps.td - Fixed point ops  --------------------*- tablegen -*-===//
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
// This is the operation definition file for fixed point ops (and real
// equivalents).
//
//===----------------------------------------------------------------------===//

#ifdef DIALECT_INTELQUANTOPS_INTELQUANT_OPS_
#else
#define DIALECT_INTELQUANTOPS_INTELQUANT_OPS_
#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/Dialect/QuantOps/QuantPredicates.td"

def intelquant_Dialect : Dialect {
  let name = "intelquant";
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// Real value for an (inclusive) min/max clamp limit.
def intelquant_ClampValueAttr : OptionalAttr<F64Attr>;

// Element-wise activation function to apply.
// Note that RELU activations are not here: they are expressed as clamps.
def intelquant_EwUnaryFnAttr :
    StringBasedAttr<CPred<"true">, "element-wise unary function"> {
  let returnType = [{ StringRef }];
  let defaultValue = "IDENTITY";
}

class intelquant_ConstEwUnaryFn<string val> : ConstantAttr<intelquant_EwUnaryFnAttr, val>;
def intelquant_EwUnaryFn_Abs     : intelquant_ConstEwUnaryFn<"ABS">;
def intelquant_EwUnaryFn_Exp     : intelquant_ConstEwUnaryFn<"EXP">;
def intelquant_EwUnaryFn_Identity: intelquant_ConstEwUnaryFn<"IDENTITY">;
def intelquant_EwUnaryFn_Log     : intelquant_ConstEwUnaryFn<"LOG">;
def intelquant_EwUnaryFn_Neg     : intelquant_ConstEwUnaryFn<"NEG">;
def intelquant_EwUnaryFn_Rsqrt   : intelquant_ConstEwUnaryFn<"RSQRT">;
def intelquant_EwUnaryFn_Sigmoid : intelquant_ConstEwUnaryFn<"SIGMOID">;
def intelquant_EwUnaryFn_Sign    : intelquant_ConstEwUnaryFn<"SIGN">;
def intelquant_EwUnaryFn_Sin     : intelquant_ConstEwUnaryFn<"SIN">;
def intelquant_EwUnaryFn_Sqrt    : intelquant_ConstEwUnaryFn<"SQRT">;
def intelquant_EwUnaryFn_Square  : intelquant_ConstEwUnaryFn<"SQUARE">;
def intelquant_EwUnaryFn_Tanh    : intelquant_ConstEwUnaryFn<"TANH">;

//===----------------------------------------------------------------------===//
// Comparison functions (compares relative to zero on a subtraction result).
//===----------------------------------------------------------------------===//

def intelquant_CompareZ    : EnumAttrCase<"CMPZ">;
def intelquant_CompareNZ   : EnumAttrCase<"CMPNZ">;
def intelquant_CompareLZ   : EnumAttrCase<"CMPLZ">;
def intelquant_CompareLZE  : EnumAttrCase<"CMPLZE">;
def intelquant_CompareGZ   : EnumAttrCase<"CMPGZ">;
def intelquant_CompareGZE  : EnumAttrCase<"CMPGZE">;

def intelquant_CompareFnAttr : EnumAttr<"ComparisonFn",
    "Type of subtraction-result comparison to perform.",
    [
      intelquant_CompareZ,
      intelquant_CompareNZ,
      intelquant_CompareLZ,
      intelquant_CompareLZE,
      intelquant_CompareGZ,
      intelquant_CompareGZE
    ]>;

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class intelquant_Op<string mnemonic, list<OpTrait> traits> :
    Op<intelquant_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Fixed-point (fxp) arithmetic ops used by kernels.
// Some of these are temporary pending inclusion into a more core dialect.
//===----------------------------------------------------------------------===//

def intelquant_ClampISOp : intelquant_Op<"clampis", [NoSideEffect, SameOperandsAndResultType]> {
  let summary =
      "Clamps a signed-integer like argument to a min/max range.";
  let description = [{
    Element-wise equivalent to:
      r = std::min(clamp_max, std::max(e, clamp_min))
  }];
  let arguments = (ins IntegerLike:$operand,
                       APIntAttr:$clamp_min,
                       APIntAttr:$clamp_max);
  let results = (outs IntegerLike);
}

def intelquant_ConvertISOp :
    intelquant_Op<"convertis",
               [NoSideEffect, SameOperandsAndResultShape]> {
  let summary =
      "Does an element-wise conversion from a signed integer to signed integer";
  let description = [{
    Similar to an element-wise static_cast in C++, from a one signed integer
    element type to another.
  }];
  let arguments = (ins IntegerLike:$operand);
  let results = (outs IntegerLike);
}

def intelquant_ConvertISToFOp :
    intelquant_Op<"convertistof",
               [NoSideEffect, SameOperandsAndResultShape]> {
  let summary =
      "Does an element-wise conversion from a signed integer to a float";
  let description = [{
    Similar to an element-wise static_cast in C++, from a signed integer
    element type to a floating point element type, rounding to the nearest
    floating point value.
  }];
  let arguments = (ins IntegerLike:$operand);
  let results = (outs FloatLike);
}


def intelquant_VecScalarSaturatingRoundingDoublingHighMulISOp :
    intelquant_Op<"vs_saturating_rounding_doubling_high_mulis",
               [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Implements equivalent functionality to ARMv7 NEON VQRDMULH";
  let description = [{
    Equivalent to the ARMv7 NEON VQRDMULH instruction.
    See gemmlowp::SaturatingRoundingDoublingHighMul for a reference
    implementation.
  }];
  let arguments = (ins IntegerLike:$a, APIntAttr:$b);
  let results = (outs IntegerLike);
}

def intelquant_RoundingDivideByPotISOp :
    intelquant_Op<"rounding_divide_by_potis", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = [{
    Computes a rounding arithmetic right shift.
  }];
  let description = [{
    Computes integer division by a power-of-two, correctly rounded-to-nearest.
    Also known as a rounding arithmetic right shift. See
    gemmlowp::RoundingDivideByPOT for a reference implementation.
  }];
  let arguments = (ins IntegerLike:$operand, APIntAttr:$exponent);
  let results = (outs IntegerLike:$res);
  let verifier = [{
    auto verifyExponent = exponent().getSExtValue();
    if (verifyExponent < 0 || verifyExponent > 31) {
      return emitOpError("exponent must be in range [0..31]");
    }
    return success();
  }];
}

//===----------------------------------------------------------------------===//
// Real math ops.
//
// Math ops on real numbers which may have a representation in quantized
// arithmetic. It is expected that eligible ops are lowered from a source
// dialect to this set of ops prior to the process of converting a compuation
// to a quantized form. It is a non-goal of these ops to preserve enough
// information to convert back to the higher level, source dialect.
//
// These ops support either real/floating point or QuantizedTypes as operands
// and results. Since not all transformations are supported (globally or
// sometimes for specific targets), a computation may end up with
// untransformable RealMathOps, in which case they need to be lowered as is
// (using floating point math).
//
// This op set takes advantage of the fact that it is typically trivial to
// combine a math function with a compatible bias addition and real-valued
// clamp (which can be done at a higher accumulation bit depth).
//
// In addition, all element-wise unary functions are collapsed into a single
// intelquant_RealUnaryEwOp and selected via an enum-like attribute. Especially at
// low bit depths, this makes matching simpler and allows the construction of
// generic LUT-based implementations. It also allows specific lowering rules
// to consolidate runs of chained unary ops and fuse them to preceding math
// ops, potentially allowing them to operate directly on higher precision
// intermediates without resorting to lots of custom kernels for common
// formulas that can suffer from insufficient precision at low bit depths.
//
// Comparison operators are modeled as element-wise unary functions (i.e.
// CMPZ, CMPNZ, CMPLZ, CMPGZ) intended to follow a sub and output a 1bit
// quantized value. It is expected that lowering rules can fuse them with
// the preceding sub.
//===----------------------------------------------------------------------===//

class intelquant_RealMathOp<string mnemonic, list<OpTrait> traits = [], dag args> :
    intelquant_Op<mnemonic, traits>,
    Arguments<!con(args, (ins
        intelquant_ClampValueAttr:$clamp_min, intelquant_ClampValueAttr:$clamp_max))>;

//===----------------------------------------------------------------------===//
// Element wise binary real math ops.
//===----------------------------------------------------------------------===//

class intelquant_RealBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    intelquant_RealMathOp<mnemonic, traits,
                     (ins quant_RealValueType:$lhs,
                      quant_RealValueType:$rhs)>,
    Results<(outs quant_RealValueType:$res)>;

class intelquant_RealBinaryBiasOp<string mnemonic, list<OpTrait> traits = []> :
    intelquant_RealMathOp<mnemonic, traits,
                     (ins quant_RealValueType:$lhs, quant_RealValueType:$rhs,
                          quant_RealValueType:$bias)>,
    Results<(outs quant_RealValueType:$res)>;

class intelquant_RealTernaryBiasOp<string mnemonic, list<OpTrait> traits = []> :
    intelquant_RealMathOp<mnemonic, traits,
                     (ins quant_RealValueType:$lhs, quant_RealValueType:$rhs,
                          quant_RealValueType:$bias, quant_RealValueType:$sum)>,
    Results<(outs quant_RealValueType:$res)>;

def intelquant_RealAddEwOp :
    intelquant_RealBinaryOp<"real_add_ew", [NoSideEffect]>;

def intelquant_RealSubEwOp :
    intelquant_RealBinaryOp<"real_sub_ew", [NoSideEffect]>;

def intelquant_RealMulEwOp :
    intelquant_RealBinaryOp<"real_mul_ew", [NoSideEffect]>;

def intelquant_RealDivEwOp :
    intelquant_RealBinaryOp<"real_div_ew", [NoSideEffect]>;

//===----------------------------------------------------------------------===//
// Element wise unary real math op.
//===----------------------------------------------------------------------===//

def intelquant_RealUnaryEwOp :
    intelquant_RealMathOp<"real_unary_ew", [NoSideEffect],
        (ins quant_RealValueType:$operand, intelquant_EwUnaryFnAttr:$fn)>,
    Results<(outs quant_RealValueType:$res)>;

def intelquant_RealCompareZeroEwOp : intelquant_Op<"compare", [NoSideEffect]>,
    Arguments<(ins quant_RealValueType:$operand, intelquant_CompareFnAttr:$fn)>,
    Results<(outs I1Tensor:$res)> {
  let description = [{
    Compares a real value to zero, returning an I1 (boolean) tensor with the
    result of applying the comparison function.
  }];
}

//===----------------------------------------------------------------------===//
// Dot op with fused bias addition.
//===----------------------------------------------------------------------===//

def intelquant_RealMatMulOp :
    intelquant_RealBinaryOp<"real_matmul", [NoSideEffect]> {
  let summary = "Matmul";
  let description = [{
    A matrix multiply of [m, k] and [k, n] -> [m, n] where the bias vector is
    of shape [n]. Also accepts rank 3 or more input tensors, in which case
    the leading dimensions are batch dims.

    Many real systems have specific library calls optimized for this precise
    operation, which is why it is handled explicitly versus purely as a
    generalized tensor contraction.
  }];
}

def intelquant_RealMatMulBiasOp :
    intelquant_RealBinaryBiasOp<"real_matmul_bias", [NoSideEffect]> {
  let summary = "Matmul with bias";
  let description = [{
    A specialization of a RealMatMulOp that also accepts an [n] dimension
    bias vector.

    In addition, there is often special support for a fused bias and clamp,
    which is why they are included.
  }];
}

//===----------------------------------------------------------------------===//
// Fused MatMul ops
//===----------------------------------------------------------------------===//

/*def intelquant_RealMatMulBiasReluOp :
    intelquant_RealBinaryBiasOp<"real_matmul_bias_relu", [NoSideEffect]> {
  let summary = "Matmul with bias and relu";
  let description = [{
    A fused version of a RealMatMulOp that also takes an [n] dimension bias vector and performs relu.
  }];
}

def intelquant_RealMatMulBiasAddReluOp :
    intelquant_RealBinaryBiasOp<"real_matmul_bias_sum_relu", [NoSideEffect]> {
  let summary = "Matmul with bias add and relu";
  let description = [{
    A fused version of a RealMatMulOp that also takes an [n] dimension bias vector and performs sum and relu.
  }];
}*/

//===----------------------------------------------------------------------===//
// Fused Conv2D ops
//===----------------------------------------------------------------------===//

def intelquant_RealBiasOp :
    intelquant_RealBinaryOp<"real_bias", [NoSideEffect]> {
  let summary = "Bias";
  let description = [{
    Broadcast Add
  }];
}

def intelquant_RealAddOp :
    intelquant_RealBinaryOp<"real_add", [NoSideEffect]> {
  let summary = "Add";
  let description = [{
    Add
  }];
}

def intelquant_RealReluOp : intelquant_Op<"real_relu", [NoSideEffect]> {
  let summary =
      "Clamps a signed argument to (0,max) range.";
  let description = [{
    Element-wise equivalent to:
      r = std::min(clamp_max, std::max(e, 0))
  }];
  let arguments = (ins FloatLike:$operand);
  let results = (outs FloatLike);
}

def intelquant_RealConv2DOp :
    intelquant_RealBinaryOp<"real_conv2d", [NoSideEffect]> {
  let summary = "Conv2D";
  let description = [{
    Conv2D
  }];
}

def intelquant_RealConv2DReluOp :
    intelquant_RealBinaryOp<"real_conv2d_relu", [NoSideEffect]> {
  let summary = "Conv2D with Relu";
  let description = [{
    Conv2D with Relu
  }];
}

def intelquant_RealConv2DBiasOp :
    intelquant_RealBinaryBiasOp<"real_conv2d_bias", [NoSideEffect]> {
  let summary = "Conv2D with bias";
  let description = [{
    A fused version of RealConv2DOp that also takes an [n] dimension bias vector.
  }];
}

def intelquant_RealConv2DBiasReluOp :
    intelquant_RealBinaryBiasOp<"real_conv2d_bias_relu", [NoSideEffect]> {
  let summary = "Conv2D with bias and relu";
  let description = [{
    A fused version of RealConv2DOp that also takes an [n] dimension bias vector and performs relu.
  }];
}

def intelquant_RealConv2DBiasSumReluOp :
    intelquant_RealTernaryBiasOp<"real_conv2d_bias_sum_relu", [NoSideEffect]> {
  let summary = "Conv2D with bias add and relu";
  let description = [{
    A fused version of RealConv2DOp that also takes an [n] dimension bias vector and performs sum and relu.
  }];
}

def intelquant_RealConv2DRequantizeOp :
    intelquant_RealBinaryOp<"real_conv2d_requantize", [NoSideEffect]> {
  let summary = "Conv2D with Requantize";
  let description = [{
    Conv2D with Requantize
  }];
}

def intelquant_RealConv2DReluRequantizeOp :
    intelquant_RealBinaryOp<"real_conv2d_relu_requantize", [NoSideEffect]> {
  let summary = "Conv2D with Relu and Requantize";
  let description = [{
    Conv2D with Relu and Requantize
  }];
}

def intelquant_RealConv2DBiasRequantizeOp :
    intelquant_RealBinaryBiasOp<"real_conv2d_bias_requantize", [NoSideEffect]> {
  let summary = "Conv2D with bias and requantize";
  let description = [{
    A fused version of RealConv2DRequantizeOp that also takes an [n] dimension bias vector.
  }];
}

def intelquant_RealConv2DBiasReluRequantizeOp :
    intelquant_RealBinaryBiasOp<"real_conv2d_bias_relu_requantize", [NoSideEffect]> {
  let summary = "Conv2D with bias and relu and requantize";
  let description = [{
    A fused version of RealConv2DRequantizeOp that also takes an [n] dimension bias vector and performs relu.
  }];
}

def intelquant_RealConv2DBiasSumReluRequantizeOp :
    intelquant_RealTernaryBiasOp<"real_conv2d_bias_sum_relu_requantize", [NoSideEffect]> {
  let summary = "Conv2D with bias add and relu and requantize";
  let description = [{
    A fused version of RealConv2DRequantizeOp that also takes an [n] dimension bias vector and performs sum and relu.
  }];
}
#endif  // DIALECT_INTELQUANTOPS_INTELQUANT_OPS_
