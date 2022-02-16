#ifndef RT_TYPES_H_
#define RT_TYPES_H_

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"

#include <utility>

namespace rt {

class ChainType : public mlir::Type::TypeBase<ChainType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
};

/// Here we define a storage class for a ComplexType, that holds a non-zero
/// integer and an integer type.
struct ComplexTypeStorage : public mlir::TypeStorage {
  ComplexTypeStorage(unsigned nonZeroParam, mlir::IntegerType integerType) : nonZeroParam(nonZeroParam), integerType(integerType) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = std::pair<unsigned, mlir::IntegerType>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(nonZeroParam, integerType);
  }

  /// Define a hash function for the key type.
  /// Note: This isn't necessary because std::pair, unsigned, and Type all have
  /// hash functions already available.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  /// Define a construction function for the key type.
  /// Note: This isn't necessary because KeyTy can be directly constructed with
  /// the given parameters.
  static KeyTy getKey(unsigned nonZeroParam, mlir::IntegerType integerType) {
    return KeyTy(nonZeroParam, integerType);
  }

  /// Define a construction method for creating a new instance of this storage.
  static ComplexTypeStorage* construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<ComplexTypeStorage>())
        ComplexTypeStorage(key.first, key.second);
  }

  /// The parametric data held by the storage class.
  unsigned nonZeroParam;
  mlir::IntegerType integerType;
};

/// This class defines a parametric type. All derived types must inherit from
/// the CRTP class 'Type::TypeBase'. It takes as template parameters the
/// concrete type (ComplexType), the base class to use (Type), the storage
/// class (ComplexTypeStorage), and an optional set of traits and
/// interfaces(detailed below).
class ComplexType : public mlir::Type::TypeBase<ComplexType, mlir::Type, ComplexTypeStorage> {
 public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This method is used to get an instance of the 'ComplexType'. This method
  /// asserts that all of the construction invariants were satisfied. To
  /// gracefully handle failed construction, getChecked should be used instead.
  static ComplexType get(::mlir::MLIRContext *context, unsigned param, mlir::IntegerType type) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. All parameters to the storage class are passed after the
    // context.
    return Base::get(context, param, type);
  }

  /// This method is used to get an instance of the 'ComplexType'. If any of the
  /// construction invariants are invalid, errors are emitted with the provided
  /// `emitError` function and a null type is returned.
  /// Note: This method is completely optional.
  static ComplexType getChecked(mlir::function_ref<mlir::InFlightDiagnostic()> emitError,
                                unsigned param, mlir::IntegerType type) {
    // Call into a helper 'getChecked' method in 'TypeBase' to get a uniqued
    // instance of this type. All parameters to the storage class are passed
    // after the context.
    return Base::getChecked(emitError, type.getContext(), param, type);
  }

  /// This method is used to verify the construction invariants passed into the
  /// 'get' and 'getChecked' methods. Note: This method is completely optional.
  static mlir::LogicalResult verify(mlir::function_ref<mlir::InFlightDiagnostic()> emitError,
                              unsigned param, mlir::IntegerType type) {
    // Our type only allows non-zero parameters.
    if (param == 0)
      return emitError() << "non-zero parameter passed to 'ComplexType'";
    // Our type also expects an integer type.
    if (!type.isa<mlir::IntegerType>())
      return emitError() << "non integer-type passed to 'ComplexType'";
    return mlir::success();
  }

  /// Return the parameter value.
  unsigned getParameter() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->nonZeroParam;
  }

  /// Return the integer parameter type.
  mlir::IntegerType getParameterType() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->integerType;
  }
};


} // namespace rt

#endif // RT_TYPES_H_