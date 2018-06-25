//===- Lexer.h - MLIR Lexer Interface ---------------------------*- C++ -*-===//
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
// This file declares the MLIR Lexer class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_PARSER_LEXER_H
#define MLIR_LIB_PARSER_LEXER_H

#include "mlir/Parser.h"
#include "Token.h"

namespace mlir {

/// This class breaks up the current file into a token stream.
class Lexer {
  llvm::SourceMgr &sourceMgr;
  const SMDiagnosticHandlerTy &errorReporter;

  StringRef curBuffer;
  const char *curPtr;

  Lexer(const Lexer&) = delete;
  void operator=(const Lexer&) = delete;
public:
  explicit Lexer(llvm::SourceMgr &sourceMgr,
                 const SMDiagnosticHandlerTy &errorReporter);

  llvm::SourceMgr &getSourceMgr() { return sourceMgr; }

  Token lexToken();

  /// Change the position of the lexer cursor.  The next token we lex will start
  /// at the designated point in the input.
  void resetPointer(const char *newPointer) { curPtr = newPointer; }
private:
  // Helpers.
  Token formToken(Token::TokenKind kind, const char *tokStart) {
    return Token(kind, StringRef(tokStart, curPtr-tokStart));
  }

  Token emitError(const char *loc, const Twine &message);

  // Lexer implementation methods.
  Token lexComment();
  Token lexBareIdentifierOrKeyword(const char *tokStart);
  Token lexAtIdentifier(const char *tokStart);
  Token lexNumber(const char *tokStart);
};

} // end namespace mlir

#endif  // MLIR_LIB_PARSER_LEXER_H