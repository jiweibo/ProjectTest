#include "rt_base.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "rt_ops.h"
#include "types.h"

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

  addTypes<ChainType>();

  addOperations<
#define GET_OP_LIST
#include "rt_ops.cpp.inc"
      >();
}

mlir::Type RTDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword)) return nullptr;

  if (keyword == "chain") return rt::ChainType::get(getContext());
  // // parse complex type, for exampe complex<1, 1>.
  // if (keyword == "complex") {
  //   // parse "<"
  //   if (parser.parseLess()) return nullptr;
  //   int32_t first, second;
  //   // parse first integer
  //   if (parser.parseInteger(first)) return nullptr;
  //   // parse ","
  //   if (parser.parseComma()) return nullptr;
  //   // parse second integer
  //   if (parser.parseInteger(second)) return nullptr;
  //   // parse ">"
  //   if (parser.parseGreater()) return nullptr;

  //   return rt::ComplexType::get(getContext(), first,
  //   mlir::IntegerType::get(getContext(), second));
  // }

  // if (keyword == "pair") {
  //   // parse "<"
  //   if (parser.parseLess()) return nullptr;
  //   int32_t first, second;
  //   // parse first integer
  //   if (parser.parseInteger(first)) return nullptr;
  //   // parse ","
  //   if (parser.parseComma()) return nullptr;
  //   // parse second integer
  //   if (parser.parseInteger(second)) return nullptr;
  //   // parse ">"
  //   if (parser.parseGreater()) return nullptr;

  //   return rt::PairType::get(getContext(), first, second);
  // }

  if (auto type = mlir::Dialect::parseType(parser)) return type;

  mlir::Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  mlir::emitError(loc) << "unknown tfrt type " << keyword;
  return {};
}

void RTDialect::printType(mlir::Type type,
                          mlir::DialectAsmPrinter &printer) const {
  if (type.isa<rt::ChainType>()) {
    printer << "chain";
  // } else if (type.isa<rt::ComplexType>()) {
  //   auto complexType = type.cast<rt::ComplexType>();
  //   printer << "complex<" << complexType.getParameter() << ", " <<
  //   complexType.getParameterType() << ">";
  // } else if (type.isa<rt::PairType>()) {
  //   auto pairType = type.cast<rt::PairType>();
  //   printer << "pair<" << pairType.getFirst() << ", " << pairType.getSecond()
  //   << ">" ;
  } else {
    llvm_unreachable("unknown rt type");
  }
}

// ::mlir::Type PairType::parse(::mlir::AsmParser& parser) {
//   llvm::StringRef keyword;
//   parser.parseKeyword(&keyword);
//   if (keyword == "pair") {
//     // parse "<"
//     if (parser.parseLess())
//       return nullptr;
//     int32_t first, second;
//     // parse first integer
//     if (parser.parseInteger(first))
//       return nullptr;
//     // parse ","
//     if (parser.parseComma())
//       return nullptr;
//     // parse second integer
//     if (parser.parseInteger(second))
//       return nullptr;
//     // parse ">"
//     if (parser.parseGreater())
//       return nullptr;

//     return rt::PairType::get(parser.getContext(), first, second);
//   }
//   return mlir::Type();
// }
// void PairType::print(::mlir::AsmPrinter& printer) const {
//   printer << "pair<" << getFirst() << ", " << getSecond() << ">";
// }

} // namespace rt
