#ifndef RT_BASE_H_
#define RT_BASE_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "rt_base.h.inc"

#define GET_OP_CLASSES
#include "rt_ops.h.inc"

#endif // RT_BASE_H_