#include "rt_base.h"

#include "rt_base.cpp.inc"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

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
}


}  // namespace rt
