#ifndef RT_BASE_H_
#define RT_BASE_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

namespace rt {

class RTDialect : public ::mlir::Dialect {
  explicit RTDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<RTDialect>()) {

    initialize();
  }

  /// Parse a type registered to this dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(mlir::Type, mlir::DialectAsmPrinter &) const override;

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~RTDialect() override = default;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("rt");
  }
};

} // namespace rt

#endif // RT_BASE_H_