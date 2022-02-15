#include "rt_base.h"
#include "rt_ops.h"
#include "types.h"
#include "llvm/Support/ErrorHandling.h"

namespace rt {

//===----------------------------------------------------------------------===//
// RTDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void RTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "rt_ops.cpp.inc"
      >();
  addTypes<ChainType, ComplexType>();
}

mlir::Type RTDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (spec == "chain") return rt::ChainType::get(getContext());
  if (auto type = mlir::Dialect::parseType(parser)) return type;

  mlir::Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  mlir::emitError(loc) << "unknown tfrt type " << spec;
  return {};
}

void RTDialect::printType(mlir::Type type,
                          mlir::DialectAsmPrinter &printer) const {
  if (type.isa<rt::ChainType>()) {
    printer << "chain";
  } else {
    llvm_unreachable("unknown rt type");
  }
}

}  // namespace rt
