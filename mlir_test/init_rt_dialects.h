#pragma once

#include "mlir/IR/Dialect.h"

namespace rt {

// Registers dialects that can be used in executed MLIR functions (functions
// with operations that will be translated to BEF kernel calls).
void RegisterRTDialects(mlir::DialectRegistry& registry);

} // namespace rt