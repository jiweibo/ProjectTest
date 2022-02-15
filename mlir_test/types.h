#ifndef RT_TYPES_H_
#define RT_TYPES_H_

#include "mlir/IR/Types.h"

namespace rt {

class ChainType : public mlir::Type::TypeBase<ChainType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
};

} // namespace rt

#endif // RT_TYPES_H_