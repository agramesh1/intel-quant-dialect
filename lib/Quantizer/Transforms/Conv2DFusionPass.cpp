//===- Conv2dFusionPass.cpp - Fuses Conv2D ------------===//
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
// This file defines a pass to fuse conv2d with bias/add/relu/qcast ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QuantOps/QuantOps.h"
#include "mlir/Dialect/FxpMathOps/FxpMathOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Quantizer/Transforms/Passes.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/UseDefLists.h"

using namespace mlir;
using namespace mlir::quantizer;
using namespace mlir::quant;
using namespace mlir::fxpmath;

namespace {

class Conv2dFusionPass
    : public FunctionPass<Conv2dFusionPass> {
  void runOnFunction() override;
};

template <typename OpTy>
class QuantizedConv2DRequantize : public RewritePattern {
public:
  QuantizedConv2DRequantize (MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto result = op->getResult(0);

    if (!result->hasOneUse())
      return matchFailure();
  
    for (auto *qcast : result->getUsers()) {
      if (matchPattern(qcast, m_Op<QuantizeCastOp>())) {
        // convert conv2d + qcast -> fused Conv2DRequantize
        std::vector<Value *> operands;
	operands.push_back(op->getOperand(0));
	operands.push_back(op->getOperand(1));
        rewriter.replaceOpWithNewOp<fxpmath::RealConv2DRequantizeOp>(qcast, qcast->getResult(0)->getType(), operands, op->getAttrs());
        return matchSuccess();
      }
    }

    return matchFailure();
  }
};

template <typename OpTy>
class QuantizedConv2DRelu : public RewritePattern {
public:
  QuantizedConv2DRelu (MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto result = op->getResult(0);

    if (!result->hasOneUse())
      return matchFailure();
  
    for (auto *relu : result->getUsers()) {
      if (matchPattern(relu, m_Op<RealReluOp>())) {
        // convert conv2d + qcast -> fused Conv2DRequantize
        std::vector<Value *> operands;
	operands.push_back(op->getOperand(0));
	operands.push_back(op->getOperand(1));
        rewriter.replaceOpWithNewOp<fxpmath::RealConv2DReluOp>(relu, relu->getResult(0)->getType(), operands, op->getAttrs());
        return matchSuccess();
      }
    }

    return matchFailure();
  }
};

template <typename OpTy>
class QuantizedConv2DReluRequantize : public RewritePattern {
public:
  QuantizedConv2DReluRequantize (MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 2, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto result = op->getResult(0);

    if (!result->hasOneUse())
      return matchFailure();
  
    for (auto *relu : result->getUsers()) {
      if (matchPattern(relu, m_Op<RealReluOp>())) {
        // convert conv2d + qcast -> fused Conv2DRequantize
        if (!result->hasOneUse())
          return matchFailure();
        for (auto *qcast : relu->getResult(0)->getUsers()) {
          if (matchPattern(qcast, m_Op<QuantizeCastOp>())) {
            std::vector<Value *> operands;
	    operands.push_back(op->getOperand(0));
	    operands.push_back(op->getOperand(1));
            rewriter.replaceOpWithNewOp<fxpmath::RealConv2DReluRequantizeOp>(qcast, qcast->getResult(0)->getType(), operands, op->getAttrs());
            return matchSuccess();
	  }
	}
      }
    }
    return matchFailure();
  }
};


template <typename OpTy>
class QuantizedConv2DBiasRelu : public RewritePattern {
public:
  QuantizedConv2DBiasRelu (MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto result = op->getResult(0);

    if (!result->hasOneUse())
      return matchFailure();
  
    for (auto *bias : result->getUsers()) {
      if (matchPattern(bias, m_Op<RealBiasOp>())) {
        // convert conv2d + qcast -> fused Conv2DRequantize
        for (auto *relu : bias->getResult(0)->getUsers()) {
          if (matchPattern(relu, m_Op<RealReluOp>())) {
            std::vector<Value *> operands;
	    operands.push_back(op->getOperand(0));
	    operands.push_back(op->getOperand(1));
	    operands.push_back(bias->getOperand(1));
            rewriter.replaceOpWithNewOp<fxpmath::RealConv2DBiasReluOp>(relu, relu->getResult(0)->getType(), operands, op->getAttrs());
            return matchSuccess();
	  }
	}
      }
    }

    return matchFailure();
  }
};

template <typename OpTy>
class QuantizedConv2DBiasReluRequantize : public RewritePattern {
public:
  QuantizedConv2DBiasReluRequantize (MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 2, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto result = op->getResult(0);

    if (!result->hasOneUse())
      return matchFailure();
  
    for (auto *bias : result->getUsers()) {
      if (matchPattern(bias, m_Op<RealBiasOp>())) {
        // convert conv2d + qcast -> fused Conv2DRequantize
        for (auto *relu : bias->getResult(0)->getUsers()) {
          if (matchPattern(relu, m_Op<RealReluOp>())) {
            for (auto *qcast : relu->getResult(0)->getUsers()) {
              if (matchPattern(qcast, m_Op<QuantizeCastOp>())) {
                std::vector<Value *> operands;
                operands.push_back(op->getOperand(0));
	        operands.push_back(op->getOperand(1));
	        operands.push_back(bias->getOperand(1));
                rewriter.replaceOpWithNewOp<fxpmath::RealConv2DBiasReluRequantizeOp>(qcast, qcast->getResult(0)->getType(), operands, op->getAttrs());
                return matchSuccess();
	      }
	    }
	  }
	}
      }
    }

    return matchFailure();
  }
};

template <typename OpTy>
class QuantizedConv2DBiasSumRelu : public RewritePattern {
public:
  QuantizedConv2DBiasSumRelu (MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto result = op->getResult(0);

    if (!result->hasOneUse())
      return matchFailure();

    for (auto *bias : result->getUsers()) {
      if (matchPattern(bias, m_Op<RealBiasOp>())) {
        // convert conv2d + qcast -> fused Conv2DRequantize
        for (auto *sum : bias->getResult(0)->getUsers()) {
          if (matchPattern(sum, m_Op<RealAddOp>())) {
            for (auto *relu : sum->getResult(0)->getUsers()) {
              if (matchPattern(relu, m_Op<RealReluOp>())) {
                std::vector<Value *> operands;
   	        operands.push_back(op->getOperand(0));
  	        operands.push_back(op->getOperand(1));
	        operands.push_back(bias->getOperand(1));
	        operands.push_back(sum->getOperand(1));
                rewriter.replaceOpWithNewOp<fxpmath::RealConv2DBiasSumReluOp>(sum, sum->getResult(0)->getType(), operands, op->getAttrs());
                return matchSuccess();
	      }
	    }
	  }
	}
      }
    }

    return matchFailure();
  }
};

template <typename OpTy>
class QuantizedConv2DBiasSumReluRequantize : public RewritePattern {
public:
  QuantizedConv2DBiasSumReluRequantize (MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 2, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto result = op->getResult(0);

    if (!result->hasOneUse())
      return matchFailure();
  
    for (auto *bias : result->getUsers()) {
      if (matchPattern(bias, m_Op<RealBiasOp>())) {
        // convert conv2d + qcast -> fused Conv2DRequantize
        for (auto *sum : bias->getResult(0)->getUsers()) {
          if (matchPattern(sum, m_Op<RealAddOp>())) {
            for (auto *relu : sum->getResult(0)->getUsers()) {
              if (matchPattern(relu, m_Op<RealReluOp>())) {
                for (auto *qcast : relu->getResult(0)->getUsers()) {
                  if (matchPattern(qcast, m_Op<QuantizeCastOp>())) {
                    std::vector<Value *> operands;
   	            operands.push_back(op->getOperand(0));
  	            operands.push_back(op->getOperand(1));
	            operands.push_back(bias->getOperand(1));
	            operands.push_back(sum->getOperand(1));
                    rewriter.replaceOpWithNewOp<fxpmath::RealConv2DBiasSumReluRequantizeOp>(qcast, qcast->getResult(0)->getType(), operands, op->getAttrs());
                    return matchSuccess();
		  }
		}
	      }
	    }
	  }
	}
      }
    }

    return matchFailure();
  }
};

} // end anonymous namespace

void Conv2dFusionPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto &func = getFunction();
  auto *context = &getContext();
  patterns.push_back(
      llvm::make_unique<QuantizedConv2DRequantize<RealConv2DOp>>(context));
  patterns.push_back(
      llvm::make_unique<QuantizedConv2DRelu<RealConv2DOp>>(context));
  patterns.push_back(
      llvm::make_unique<QuantizedConv2DReluRequantize<RealConv2DOp>>(context));
  patterns.push_back(
      llvm::make_unique<QuantizedConv2DBiasRelu<RealConv2DOp>>(context));
  patterns.push_back(
      llvm::make_unique<QuantizedConv2DBiasReluRequantize<RealConv2DOp>>(context));
  patterns.push_back(
      llvm::make_unique<QuantizedConv2DBiasSumRelu<RealConv2DOp>>(context));
  patterns.push_back(
      llvm::make_unique<QuantizedConv2DBiasSumReluRequantize<RealConv2DOp>>(context));
  applyPatternsGreedily(func, std::move(patterns));
}

FunctionPassBase *mlir::quantizer::createConv2dFusionPass() {
  return new Conv2dFusionPass();
}

static PassRegistration<Conv2dFusionPass>
    pass("quantizer-conv2d-fusion",
         "Fuse Conv2D with bias/add/relu/qcast for execution on Intel CPUs");
