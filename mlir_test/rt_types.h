#ifndef RT_TYPES_H_
#define RT_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <limits>
#include <tuple>
#include <utility>

using mlir::Type;
using mlir::LogicalResult;
using mlir::function_ref;
using mlir::InFlightDiagnostic;
using mlir::TypeStorage;

namespace rt {
namespace detail {

//===----------------------------------------------------------------------===//
// RTTypeAndSizeStorage.
//===----------------------------------------------------------------------===//

/// Common storage used for RT dialect types that need an element type and a
/// number: arrays, fixed and scalable vectors. The actual semantics of the
/// type is defined by its kind.

struct RTTypeAndSizeStorage : public TypeStorage {
  using KeyTy = std::tuple<Type, unsigned>;

  RTTypeAndSizeStorage(const KeyTy& key) 
    : elementType(std::get<0>(key)), numElements(std::get<1>(key)) {}
  
  static RTTypeAndSizeStorage *construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key) {
    return new (allocator.allocate<RTTypeAndSizeStorage>()) RTTypeAndSizeStorage(key);
  }

  bool operator==(const KeyTy& key) const {
    return std::make_tuple(elementType, numElements) == key;
  }

  Type elementType;
  unsigned numElements;
};

//===----------------------------------------------------------------------===//
// Printing and parsing.
//===----------------------------------------------------------------------===//

/// Parses an RT dialect type.
Type parseType(mlir::DialectAsmParser& parser);

/// Prints an RT dialect type.
void printType(Type type, mlir::AsmPrinter& printer);
}  // namespace detail


class ChainType
    : public mlir::Type::TypeBase<ChainType, Type, mlir::TypeStorage> {
public:
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// RTVectorType.
//===----------------------------------------------------------------------===//

/// RT dialect vector type, represents a sequence of elements that can be
/// processed as one. This is a base class for fixed and scalable vectors.
class RTVectorType : public Type {
 public:
  /// Inherit base constructor.
  using Type::Type;

  /// Support type casting functionality.
  static bool classof(Type type);

  /// Returns the element type of the vector.
  Type getElementType();

  // TODO(wilber): support ElementCount type.
  /// Returns the number of elements in the vector.
  int getElementCount();

  /// Verifies that the type about to be constructed is well-formed.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              Type elementType, unsigned numElements);
};

//===----------------------------------------------------------------------===//
// RTFixedVectorType.
//===----------------------------------------------------------------------===//

/// RT dialect fixed vector type, represents a sequence of elements of known
/// length that can be processed as one.
class RTFixedVectorType : public Type::TypeBase<RTFixedVectorType, Type, detail::RTTypeAndSizeStorage> {
 public:
  /// Inherit base constructor.
  using Base::Base;
  using Base::getChecked;

  /// Gets or creates a fixed vector type containing `numElements` of
  /// `elementType` in the same context as `elementType`.
  static RTFixedVectorType get(Type elementType, unsigned numElements);
  static RTFixedVectorType getChecked(function_ref<InFlightDiagnostic()> emitError, Type elementType, unsigned numElements);

  static bool isValidElementType(Type type);

  Type getElementType();

  unsigned getNumElements();

  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError, Type elementType, unsigned numElements);
};

//===----------------------------------------------------------------------===//
// RTScalableVectorType.
//===----------------------------------------------------------------------===//

/// RT dialect scalable vector type, represents a sequence of elements of
/// unknown length that is known to be divisible by some constant.
class RTScalableVectorType : public Type::TypeBase<RTScalableVectorType, Type, detail::RTTypeAndSizeStorage> {
 public:
  /// Inherit base constructor.
  using Base::Base;
  using Base::getChecked;

  /// Gets or creates a scalable vector type containing a non-zero multiple of
  /// `minNumElements` of `elementType` in the same context as `elementType`.
  static RTScalableVectorType get(Type elementType, unsigned minNumElements);
  static RTScalableVectorType getChecked(function_ref<InFlightDiagnostic()> emitError, Type elementType, unsigned minNumElements);

  /// Checks if the given type can be used in vector type.
  static bool isValidElementType(Type type);

  /// Returns the element type of the vector.
  Type getElementType();

  /// Returns the scaling factor of the number of elements in the vector. The
  /// vector contains at least the resulting number of elements, or any non-zero
  /// multiple of this number.
  unsigned getMinNumElements();

  /// Verifies that the type about to be constructed is well-formed.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError, Type elementType, unsigned minNumElements);
};

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

/// Returns `true` if the given type is compatible with the RT dialect.
bool isCompatibleType(Type type);

/// Returns `true` if the given outter type is compatible with the RT dialect
/// without checking its potential nested types such as struct elements.
bool isCompatibleOuterType(Type type);

/// Returns `true` if the given type is a floating-point type compatible with
/// the RT dialect.
bool isCompatibleFloatingPointType(Type type);

/// Returns `true` if the given type is a vector type compatible with the RT
/// dialect. Compatible types include 1D built-in vector types of built-in
/// integers and floating-point values, RT dialect fixed vector types and
/// RT dialect scalable vector types.
bool isCompatibleVectorType(Type type);

/// Returns the element type of any vector type compatible with the RT dialect.
Type getVectorElementType(Type type);

}  // namespace rt

#endif // RT_TYPES_H_