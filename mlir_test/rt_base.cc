#include "rt_base.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "rt_ops.h"
#include "rt_types.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"

#include "rt_base.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "rt_types.cpp.inc"

namespace rt {

//===----------------------------------------------------------------------===//
// RTDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void RTDialect::initialize() {
  //   addTypes<
  // #define GET_TYPEDEF_LIST
  // #include "rt_types.cpp.inc"
  //       >();

  addTypes<ChainType, RTFixedVectorType, RTScalableVectorType>();

  addOperations<
#define GET_OP_LIST
#include "rt_ops.cpp.inc"
      >();
}

mlir::Type RTDialect::parseType(mlir::DialectAsmParser& parser) const {
  return detail::parseType(parser);
}

void RTDialect::printType(mlir::Type type,
                          mlir::DialectAsmPrinter& printer) const {
  return detail::printType(type, printer);
}

} // namespace rt
