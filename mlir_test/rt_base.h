#ifndef RT_BASE_H_
#define RT_BASE_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"

#include "rt_base.h.inc"

#define GET_TYPEDEF_CLASSES
#include "rt_types.h.inc"

#endif // RT_BASE_H_