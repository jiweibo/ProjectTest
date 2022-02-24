#include "rt_ops.h"
#include "types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

#include <iostream>

namespace rt {
using namespace mlir;

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//
static ParseResult parseReturnOp(OpAsmParser& parser, OperationState& result) {
  llvm::SmallVector<OpAsmParser::OperandType, 2> opInfo;
  llvm::SmallVector<Type, 2> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return mlir::failure(
      parser.parseOperandList(opInfo) ||
      (!opInfo.empty() && parser.parseColonTypeList(types)) ||
      parser.resolveOperands(opInfo, types, loc, result.operands));
}

static void print(OpAsmPrinter& p, ReturnOp& op) {
  if (op.getNumOperands() > 0) {
    p << ' ';
    p.printOperands(op.getOperands());
    p << " : ";
    llvm::interleaveComma(op.getOperandTypes(), p);
  }
}

static LogicalResult verify(ReturnOp op) {
  // The parent is often a 'func' but not always.
  auto function = llvm::dyn_cast<FuncOp>(op->getParentOp());

  // We allow rt.return in arbitrary control flow structures.
  if (!function)
    return success();

  // The operand number and types must match the function signature.
  auto results = function.getType().getResults();
  if (op->getNumOperands() != results.size())
    return op.emitOpError("has ")
           << op.getNumOperands()
           << " operands, but enclosing function returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (op->getOperand(i).getType() != results[i])
      return op.emitError()
             << "type of return operand " << i << " ("
             << op.getOperand(i).getType()
             << ") doesn't match function result type (" << results[i] << ")";

  return success();
}

} // namespace rt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "rt_ops.cpp.inc"