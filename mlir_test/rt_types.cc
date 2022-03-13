#include "rt_types.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <iostream>

namespace rt {
namespace detail {

//===----------------------------------------------------------------------===//
// Printing.
//===----------------------------------------------------------------------===//

/// If the given type is compatible with the RT dialect, prints it using
/// internal functions to avoid getting a verbose `!rt` prefix. Otherwise
/// prints it as usual.
static void dispatchPrint(mlir::AsmPrinter& printer, Type type) {
  if (isCompatibleType(type) && !type.isa<mlir::IntegerType, mlir::FloatType, mlir::VectorType>())
    return detail::printType(type, printer);
  printer.printType(type);
}

/// Retuns the keyword to use for the given type.
static llvm::StringRef getTypeKeyword(Type type) {
  return mlir::TypeSwitch<Type, llvm::StringRef>(type)
    .Case<RTFixedVectorType, RTScalableVectorType>([&](Type) {return "vec";})
    .Case<ChainType>([&](Type){return "chain";})
    .Default([](Type) -> llvm::StringRef {
      llvm_unreachable("unexpected 'rt' type kind");
    });
}

/// Prints a type containing a fixed number of elements.
template <typename TypeTy>
static void printArrayOrVectorType(mlir::AsmPrinter& printer, TypeTy type) {
  printer << '<' << type.getNumElements() << " x ";
  dispatchPrint(printer, type.getElementType());
  printer << '>';
}

/// Prints the given RT dialect type recursively. This leverages closeness of
/// the RT dialect type systems to avoid printing the dialect prefix
/// repeatedly.
void printType(Type type, mlir::AsmPrinter& printer) {
  if (!type) {
    printer << "NULL - TYPE>>";
    return;
  }

  printer << getTypeKeyword(type);

  if (auto vectorType = type.dyn_cast<RTFixedVectorType>())
    return printArrayOrVectorType(printer, vectorType);
  
  if (auto vectorType = type.dyn_cast<RTScalableVectorType>()) {
    printer << "<? x " << vectorType.getMinNumElements() << " x ";
    dispatchPrint(printer, vectorType.getElementType());
    printer << '>';
    return;
  }
}

//===----------------------------------------------------------------------===//
// Parsing.
//===----------------------------------------------------------------------===//

static mlir::ParseResult dispatchParse(mlir::AsmParser &parser, mlir::Type &type);

/// Parse a RT dialect vector type.
///   rt-type ::= `vec<` `? x`? integer `x` rt-type `>`
/// Support both fixed and scalable vectors.
static Type parseVectorType(mlir::AsmParser& parser) {
  llvm::SmallVector<int64_t, 2> dims;
  mlir::SMLoc dimPos, typePos;
  Type elementType;
  mlir::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLess() || parser.getCurrentLocation(&dimPos) || 
      parser.parseDimensionList(dims, /*allowDynamic*/true) ||
      parser.getCurrentLocation(&typePos) ||
      dispatchParse(parser, elementType) || parser.parseGreater())
    return mlir::Type();
  
  // We parsed a generic dimension list, but vector only support two forms:
  //  - single non-dynamic entry in the list (fixed vector);
  //  - two elements, the first dynamic (indicated by -1) and the second
  //    non-dynamic (scalable vector).
  if (dims.empty() || dims.size() > 2 || 
      ((dims.size() == 2) ^ (dims[0] == -1)) ||
      (dims.size() == 2 && dims[1] == -1)) {
    parser.emitError(dimPos) << "expected '? x <integer> x <type>' or '<integer> x <type>'";
    return Type();
  }

  bool isScalble = dims.size() == 2;
  if (isScalble)
    return parser.getChecked<RTScalableVectorType>(loc, elementType, dims[1]);
  if (elementType.isSignlessIntOrFloat()) {
    parser.emitError(typePos) << "cannot use !rt.vec for built-in primitives, use 'vector' instead";
    return Type();
  }
  return parser.getChecked<RTFixedVectorType>(loc, elementType, dims[0]);
}

/// Parses a type appearing inside another RT dialect-compatible type. This
/// will try to parse any type in full form (including with the `!rt` prefix),
/// and on failure fall back to parsing the shord-hand version of the RT
/// dialect types without the `!rt` prefix.
static Type dispatchParse(mlir::AsmParser& parser, bool allowAny = true) {
  mlir::SMLoc keyLoc = parser.getCurrentLocation();

  // Try parsing any MLIR type.
  Type type;
  mlir::OptionalParseResult result = parser.parseOptionalType(type);
  if (result.hasValue()) {
    if (mlir::failed(result.getValue()))
      return nullptr;
    if (!allowAny) {
      parser.emitError(keyLoc) << "unexpected type, expected keyword";
      return nullptr;
    }
    return type;
  }

  // If no type found, fallback to the shorthand form.
  mlir::StringRef key;
  if (mlir::failed(parser.parseKeyword(&key)))
    return Type();
  mlir::MLIRContext* ctx = parser.getContext();
  return mlir::StringSwitch<llvm::function_ref<Type()>>(key)
    .Case("chain", [&] { return rt::ChainType::get(ctx); })
    .Case("vec", [&] { return parseVectorType(parser); })
    .Default([&] {
      parser.emitError(keyLoc) << "unknown RT type: " << key;
      return Type();
    })();
}

/// Helper to use in parse lists.
static mlir::ParseResult dispatchParse(mlir::AsmParser& parser, mlir::Type& type) {
  type = dispatchParse(parser);
  return mlir::success(type != nullptr);
}

Type parseType(mlir::DialectAsmParser& parser) {
  mlir::SMLoc loc = parser.getCurrentLocation();
  Type type = dispatchParse(parser, /*allowAny=*/false);
  if (!type)
    return type;
  if (!isCompatibleOuterType(type)) {
    parser.emitError(loc) << "unexpected type, expected keyword";
    return nullptr;
  }
  return type;
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// Vector types.
//===----------------------------------------------------------------------===//

/// Verifies that the type about to be constructed is well-formed.
template <typename VecTy>
static LogicalResult verifyVectorConstructionInvariants(function_ref<InFlightDiagnostic()> emitError, Type elementType, unsigned numElements) {
  if (numElements == 0) 
    return emitError() << "the number of vector elements must be positive";

  if (!VecTy::isValidElementType(elementType))
    return emitError() << "invalid vector element type";

  return mlir::success();
}

RTFixedVectorType RTFixedVectorType::get(Type elementType, unsigned int numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), elementType, numElements);
}

RTFixedVectorType RTFixedVectorType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                                Type elementType, unsigned numElements) {
  assert(elementType && "expected non-null subtype");
  return Base::getChecked(emitError, elementType.getContext(), elementType, numElements);
}

Type RTFixedVectorType::getElementType() {
  return static_cast<detail::RTTypeAndSizeStorage*>(impl)->elementType;
}

unsigned RTFixedVectorType::getNumElements() {
  return getImpl()->numElements;
}

bool RTFixedVectorType::isValidElementType(Type type) {
  // TODO(wilber): add valid type.
  // return type.isa<>();
  return true;
}

LogicalResult RTFixedVectorType::verify(function_ref<InFlightDiagnostic ()> emitError, Type elementType, unsigned int numElements) {
  return verifyVectorConstructionInvariants<RTFixedVectorType>(emitError, elementType, numElements);
}

//===----------------------------------------------------------------------===//
// RTScalableVectorType.
//===----------------------------------------------------------------------===//

RTScalableVectorType RTScalableVectorType::get(Type elementType, unsigned minNumElements) {
  assert(elementType && "expected non-null subtype");
  return Base::get(elementType.getContext(), elementType, minNumElements);
}

RTScalableVectorType RTScalableVectorType::getChecked(function_ref<InFlightDiagnostic()> emitError, Type elementType, unsigned minNumElements) {
  assert(elementType && "expected non-null subtype");
  return Base::getChecked(emitError, elementType.getContext(), elementType, minNumElements);
}

Type RTScalableVectorType::getElementType() {
  return static_cast<detail::RTTypeAndSizeStorage*>(impl)->elementType;
}

unsigned RTScalableVectorType::getMinNumElements() {
  return getImpl()->numElements;
}

bool RTScalableVectorType::isValidElementType(Type type) {
  // if (auto intType = type.dyn_cast<mlir::IntegerType>())
  //   return intType.isSignless();
  // return isCompatibleFloatingPointType(type);
  return true;
}

LogicalResult RTScalableVectorType::verify(function_ref<InFlightDiagnostic ()> emitError, Type elementType, unsigned int minNumElements) {
  return verifyVectorConstructionInvariants<RTScalableVectorType>(emitError, elementType, minNumElements);
}

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

bool isCompatibleOuterType(Type type) {
  if (type.isa<
      ChainType,
      mlir::Float16Type,
      mlir::Float32Type,
      mlir::Float64Type,
      mlir::Float128Type,
      RTFixedVectorType,
      RTScalableVectorType
      >()) {
    return true;
  }

  // Only signless integers are compatible.
  if (auto intType = type.dyn_cast<mlir::IntegerType>())
    return intType.isSignless();
  
  // 1D vector types are compatible.
  if (auto vecType = type.dyn_cast<mlir::VectorType>())
    return vecType.getRank() == 1;

  return false;
}

bool isCompatibleType(Type type) {
  return true;
}

bool isCompatibleFloatingPointType(Type type) {
  return type.isa<mlir::Float16Type, mlir::Float32Type, mlir::Float64Type, mlir::Float128Type>();
}

bool isCompatibleVectorType(Type type) {
  if (type.isa<RTFixedVectorType, RTScalableVectorType>()) 
    return true;
  
  if (auto vecType = type.dyn_cast<mlir::VectorType>()) {
    if (vecType.getRank() != 1)
      return false;
    Type elementType = vecType.getElementType();
    if (auto intType = elementType.dyn_cast<mlir::IntegerType>())
      return intType.isSignless();
    return elementType.isa<mlir::Float16Type, mlir::Float32Type, mlir::Float64Type, mlir::Float128Type>();
  }
  return false;
}

Type getVectorElementType(Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
    .Case<RTFixedVectorType, RTScalableVectorType, mlir::VectorType>([](auto ty) { return ty.getElementType(); })
    .Default([](Type) -> Type {
      llvm_unreachable("incompatible with RT vector type");
    });
}

}  // namespace rt
