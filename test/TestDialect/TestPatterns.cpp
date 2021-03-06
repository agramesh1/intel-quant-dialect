//===- TestPatterns.cpp - Test dialect pattern driver ---------------------===//
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

#include "TestDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;

namespace {
#include "TestPatterns.inc"

struct TestPatternDriver : public FunctionPass<TestPatternDriver> {
  void runOnFunction() override;
};
} // end anonymous namespace

void TestPatternDriver::runOnFunction() {
  mlir::OwningRewritePatternList patterns;
  populateWithGenerated(&getContext(), &patterns);

  // Verify named pattern is generated with expected name.
  RewriteListBuilder<TestNamedPatternRule>::build(patterns, &getContext());

  applyPatternsGreedily(getFunction(), std::move(patterns));
}

static mlir::PassRegistration<TestPatternDriver>
    pass("test-patterns", "Run test dialect patterns");
