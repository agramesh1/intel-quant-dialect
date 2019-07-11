//===- IntelQuantConfig.cpp - Reference fixed point config -------------------===//
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
// This file defines a TargetConfiguration for reference fixed-point math
// quantization scheme based on the IntelQuantOps (plus a small category of
// extension ops that can be added from other dialects).
//
//===----------------------------------------------------------------------===//

#include "mlir/Quantizer/Configurations/IntelQuantConfig.h"

#include "mlir/Dialect/IntelQuantOps/IntelQuantOps.h"
#include "mlir/Dialect/QuantOps/QuantOps.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Quantizer/Support/ConstraintAnalysisGraph.h"
#include "mlir/Quantizer/Support/Metadata.h"
#include "mlir/Quantizer/Support/Statistics.h"
#include "mlir/Quantizer/Support/UniformConstraints.h"
#include "mlir/StandardOps/Ops.h"

using namespace mlir;
using namespace mlir::quantizer;
using namespace mlir::intelquant;
using namespace mlir::quant;
using namespace std::placeholders;

namespace {

struct IntelQuantTargetConfigImpl : public IntelQuantTargetConfig {
  IntelQuantTargetConfigImpl(SolverContext &context)
      : IntelQuantTargetConfig(context) {
    Builder b(&context.getMlirContext());
    IntegerType i8Type = b.getIntegerType(8);
    IntegerType i16Type = b.getIntegerType(16);
    IntegerType i32Type = b.getIntegerType(32);

    q8 = addCandidateType(
        AnyQuantizedType::get(QuantizationFlags::Signed, i8Type, nullptr,
                              std::numeric_limits<int8_t>::min(),
                              std::numeric_limits<int8_t>::max()),
        CandidateQuantizedType::Scheme::UniformPerLayer);
    q16 = addCandidateType(
        AnyQuantizedType::get(QuantizationFlags::Signed, i16Type, nullptr,
                              std::numeric_limits<int16_t>::min(),
                              std::numeric_limits<int16_t>::max()),
        CandidateQuantizedType::Scheme::UniformPerLayer);
    q32ExplicitFixedPoint = addCandidateType(
        AnyQuantizedType::get(QuantizationFlags::Signed, i32Type, nullptr,
                              std::numeric_limits<int32_t>::min(),
                              std::numeric_limits<int32_t>::max()),
        CandidateQuantizedType::Scheme::UniformExplicitFixedPointScale);
    quint8 = addCandidateType(
        AnyQuantizedType::get(0, i8Type, nullptr,
                              std::numeric_limits<uint8_t>::min(),
                              std::numeric_limits<uint8_t>::max()),
        CandidateQuantizedType::Scheme::UniformPerLayer);

    // Op handlers.
    addOpHandler<ConstantOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleConstant, this, _1, _2));
    addOpHandler<ReturnOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleTerminal, this, _1, _2));
    addOpHandler<quant::StatisticsOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleStats, this, _1, _2));

    // IntelQuantOps.
    addOpHandler<RealAddEwOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleAdd, this, _1, _2));
    addOpHandler<RealMulEwOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleMul, this, _1, _2));
    addOpHandler<RealMatMulOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleMatMul, this, _1, _2));
    addOpHandler<RealMatMulBiasOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleMatMulBias, this, _1, _2));
    //addOpHandler<RealMatMulBiasReluOp>(
    //    std::bind(&IntelQuantTargetConfigImpl::handleMatMulBiasRelu, this, _1, _2));
    //addOpHandler<RealMatMulBiasSumReluOp>(
    //    std::bind(&IntelQuantTargetConfigImpl::handleMatMulBiasSumRelu, this, _1, _2));
    addOpHandler<RealReluOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleRelu, this, _1, _2));
    addOpHandler<RealBiasOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleBias, this, _1, _2));
    addOpHandler<RealAddOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleRealAdd, this, _1, _2));
    addOpHandler<RealConv2DOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleConv2D, this, _1, _2));
    addOpHandler<RealConv2DRequantizeOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleConv2DRequantize, this, _1, _2));
    addOpHandler<RealConv2DReluOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleConv2D, this, _1, _2));
    addOpHandler<RealConv2DReluRequantizeOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleConv2DRequantize, this, _1, _2));
    addOpHandler<RealConv2DBiasOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleConv2DBias, this, _1, _2));
    addOpHandler<RealConv2DBiasRequantizeOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleConv2DBiasRequantize, this, _1, _2));
    addOpHandler<RealConv2DBiasReluOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleConv2DBiasRelu, this, _1, _2));
    addOpHandler<RealConv2DBiasReluRequantizeOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleConv2DBiasReluRequantize, this, _1, _2));
    addOpHandler<RealConv2DBiasSumReluOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleConv2DBiasSumRelu, this, _1, _2));
    addOpHandler<RealConv2DBiasSumReluRequantizeOp>(
        std::bind(&IntelQuantTargetConfigImpl::handleConv2DBiasSumReluRequantize, this, _1, _2));

    // Require stats ops.
    addRequireStatsOp<RealAddEwOp>();
    addRequireStatsOp<RealSubEwOp>();
    addRequireStatsOp<RealDivEwOp>();
    addRequireStatsOp<RealMulEwOp>();
    addRequireStatsOp<RealMatMulOp>();
    addRequireStatsOp<RealMatMulBiasOp>();
  }

  bool isHandledType(Type t) const final {
    if (t.isa<FloatType>())
      return true;
    return (t.isa<VectorType>() || t.isa<TensorType>()) &&
           t.cast<ShapedType>().getElementType().isa<FloatType>();
  }

  void finalizeAnchors(CAGSlice &cag) const override {
    cag.enumerateImpliedConnections(
        [&](CAGAnchorNode *from, CAGAnchorNode *to) {
          UniformConstraintsBuilder(cag).coupleAnchors(from, to);
        });
  }

  void addValueIdentityOpByName(StringRef opName) override {
    addOpHandlerByName(
        opName,
        std::bind(&IntelQuantTargetConfigImpl::handleValueIdentity, this, _1, _2));
  }

  void handleValueIdentity(Operation *op, CAGSlice &cag) const {
    assert(op->getNumResults() == 1);
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto resultNode = cag.getResultAnchor(op, 0);
    resultNode->setTypeTransformRule(
        CAGAnchorNode::TypeTransformRule::DirectStorage);

    for (unsigned opIdx = 0, e = op->getNumOperands(); opIdx < e; ++opIdx) {
      if (!isHandledType(op->getOperand(opIdx)->getType()))
        continue;
      auto operandNode = cag.getOperandAnchor(op, opIdx);
      operandNode->setTypeTransformRule(
          CAGAnchorNode::TypeTransformRule::DirectStorage);
      UniformConstraintsBuilder(cag).coupleAnchors(operandNode, resultNode);
    }
  }

  void handleConstant(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto resultNode = cag.getResultAnchor(op, 0);
    resultNode->setTypeTransformRule(
        CAGAnchorNode::TypeTransformRule::ExpressedOnly);
    Attribute valueAttr;
    if (!matchPattern(op, m_Constant(&valueAttr))) {
      return;
    }

    AttributeTensorStatistics stats(valueAttr);
    TensorAxisStatistics layerStats;
    if (!stats.get(layerStats)) {
      op->emitOpError("could not compute statistics");
      return;
    }

    UniformConstraintsBuilder(cag).applyStats(resultNode, layerStats);
  }

  void handleTerminal(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getOperand(0)->getType()))
      return;
    auto operandNode = cag.getOperandAnchor(op, 0);
    operandNode->setTypeTransformRule(
        CAGAnchorNode::TypeTransformRule::ExpressedOnly);
  }

  void handleStats(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto argNode = cag.getOperandAnchor(op, 0);
    auto resultNode = cag.getResultAnchor(op, 0);
    UniformConstraintsBuilder(cag).coupleAnchors(argNode, resultNode);

    TensorAxisStatistics layerStats;
    auto statsOp = cast<quant::StatisticsOp>(op);
    auto layerStatsAttr = statsOp.layerStats();
    layerStats.minValue =
        layerStatsAttr.getValue({0}).cast<FloatAttr>().getValueAsDouble();
    layerStats.maxValue =
        layerStatsAttr.getValue({1}).cast<FloatAttr>().getValueAsDouble();
    UniformConstraintsBuilder(cag).applyStats(resultNode,
                                              std::move(layerStats));
  }

  void handleAdd(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Add supports 8/16 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({q8, q16});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes = disableMask;
    // NOTE: We couple the add such that the scale/zeroPoint match between
    // both args and the result. This is overly constrained in that it is
    // possible to write efficient add kernels with a bit more freedom (i.e.
    // zeroPoints can vary, scales can differ by a power of two, etc).
    // However, fully coupled yields the simples solutions on the fast path.
    // Further efficiency can be had by constraining the zeroPoint to 0, but
    // there isn't a constraint for this yet (and there are tradeoffs).
    UniformConstraintsBuilder(cag).coupleAnchors(lhs, resultNode);
    UniformConstraintsBuilder(cag).coupleAnchors(rhs, resultNode);
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleMul(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Mul supports 8/16 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({q8, q16});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes = disableMask;
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleMatMul(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Mul supports 8/16 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({q8, q16});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes = disableMask;
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleMatMulBias(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto bias = cag.getOperandAnchor(op, 2);
    bias->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});

    auto resultNode = cag.getResultAnchor(op, 0);
    UniformConstraintsBuilder(cag).propagateExplicitScale(resultNode, bias);

    // Mul supports 8/16 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({q8, q16});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes = disableMask;
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleRelu(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Relu supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleBias(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Bias supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleRealAdd(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Add supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleConv2D(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Conv2D supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleConv2DRequantize(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Conv2D supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q8});
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleConv2DRelu(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Conv2D supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleConv2DReluRequantize(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto resultNode = cag.getResultAnchor(op, 0);
    // Conv2D supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes = disableMask;
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleConv2DBias(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto bias = cag.getOperandAnchor(op, 2);
    auto resultNode = cag.getResultAnchor(op, 0);
    UniformConstraintsBuilder(cag).propagateExplicitScale(resultNode, bias);

    // Conv2D supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    bias->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    resultNode->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleConv2DBiasRequantize(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto bias = cag.getOperandAnchor(op, 2);
    auto resultNode = cag.getResultAnchor(op, 0);
    UniformConstraintsBuilder(cag).propagateExplicitScale(resultNode, bias);

    // Conv2D supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    bias->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    resultNode->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q8});
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleConv2DBiasRelu(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto bias = cag.getOperandAnchor(op, 2);
    auto resultNode = cag.getResultAnchor(op, 0);
    UniformConstraintsBuilder(cag).propagateExplicitScale(resultNode, bias);

    // Conv2D supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    bias->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    resultNode->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleConv2DBiasReluRequantize(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto bias = cag.getOperandAnchor(op, 2);
    auto resultNode = cag.getResultAnchor(op, 0);
    UniformConstraintsBuilder(cag).propagateExplicitScale(resultNode, bias);

    // Conv2D supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    bias->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    resultNode->getUniformMetadata().disabledCandidateTypes = disableMask;
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleConv2DBiasSumRelu(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto bias = cag.getOperandAnchor(op, 2);
    auto summand = cag.getOperandAnchor(op, 3);

    auto resultNode = cag.getResultAnchor(op, 0);
    UniformConstraintsBuilder(cag).propagateExplicitScale(resultNode, bias);

    // Conv2D supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    summand->getUniformMetadata().disabledCandidateTypes = disableMask;
    bias->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    resultNode->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void handleConv2DBiasSumReluRequantize(Operation *op, CAGSlice &cag) const {
    if (!isHandledType(op->getResult(0)->getType()))
      return;

    auto lhs = cag.getOperandAnchor(op, 0);
    auto rhs = cag.getOperandAnchor(op, 1);
    auto bias = cag.getOperandAnchor(op, 2);
    auto summand = cag.getOperandAnchor(op, 3);

    auto resultNode = cag.getResultAnchor(op, 0);
    UniformConstraintsBuilder(cag).propagateExplicitScale(resultNode, bias);

    // Conv2D supports 8 bit math.
    llvm::SmallBitVector disableMask =
        getCandidateTypeDisabledExceptMask({quint8});
    lhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    rhs->getUniformMetadata().disabledCandidateTypes = disableMask;
    bias->getUniformMetadata().disabledCandidateTypes =
        getCandidateTypeDisabledExceptMask({q32ExplicitFixedPoint});
    summand->getUniformMetadata().disabledCandidateTypes = disableMask;
    resultNode->getUniformMetadata().disabledCandidateTypes = disableMask;
    addRealMathOptionalConstraints(op, resultNode, cag);
  }

  void addRealMathOptionalConstraints(Operation *op, CAGAnchorNode *anchor,
                                      CAGSlice &cag) const {
    // TODO: It would be nice if these all extended some base trait instead
    // of requiring name lookup.
    auto clampMinAttr = op->getAttrOfType<FloatAttr>("clamp_min");
    auto clampMaxAttr = op->getAttrOfType<FloatAttr>("clamp_max");

    if (clampMinAttr || clampMaxAttr) {
      auto nan = APFloat::getQNaN(APFloat::IEEEdouble());
      auto clampMin = clampMinAttr ? clampMinAttr.getValue() : nan;
      auto clampMax = clampMaxAttr ? clampMaxAttr.getValue() : nan;
      UniformConstraintsBuilder(cag).clamp(anchor, clampMin, clampMax);
    }
  }

  unsigned q8;
  unsigned quint8;
  unsigned q16;
  unsigned q32ExplicitFixedPoint;
};

} // anonymous namespace

std::unique_ptr<IntelQuantTargetConfig>
IntelQuantTargetConfig::create(SolverContext &context) {
  return llvm::make_unique<IntelQuantTargetConfigImpl>(context);
}
