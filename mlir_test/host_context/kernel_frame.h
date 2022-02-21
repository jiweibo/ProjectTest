#pragma  once

#include "attribute_utils.h"

#include "llvm/ADT/ArrayRef.h"
#include <cassert>

namespace rt {
class Value;

// SyncKernelFrame captures the states associated with a kernel invocation,
// including the input arguments, attributes, result values, and the execution
// context. SyncKernelFrame is constructed by the kernel caller (currently only
// BEFInterpreter) using the SyncKernelFrameBuilder subclass. The kernel
// implementation is passed a pointer to a SyncKernelFrame object for them to
// access the inputs and attributes, and return result values.
class KernelFrame {
 public:

  // Get the number of arguments.
  int GetNumArgs() const {
    return argument_indices_.size();
  }

  // Get the argument at the given index as type T.
  template <typename T>
  T& GetArgAt(int index) const {
    return GetArgAt(index)->get<T>();
  }

  // Get the argument at the given index as Value*.
  Value* GetArgAt(int index) const {
    assert(index < GetNumArgs());
    return registers_[argument_indices_[index]];
  }

  // Get all arguments.
  llvm::ArrayRef<uint32_t> GetArguments() const { return argument_indices_; }

  // Get the number of attributes.
  int GetNumAttributes() const { return attributes_.size(); }

  const void* GetAttributeAt(int index) const {
    assert(index < GetNumAttributes());
    return attributes_[index];
  }

  // Get the attribute at the given index as type T.
  template <typename T>
  Attribute<T> GetAttributeAt(int index) const {
    return Attribute<T>(GetAttributeAt(index));
  }

  // Get the number of results.
  int GetNumResults() const {
    return result_indices_.size();
  }

  // Emplace construct the result at given index.
  template <typename T, typename... Args>
  void EmplaceResultAt(int index, Args&&... args) {
    assert(index < GetNumResults() && "Invalid result index");
    Value* result = GetResultAt(index);
    assert(!result->HasValue() && "Result value is non-empty");
    result->emplace<T>(std::forward<Args>(args)...);
  }

  // Get result at the given index.
  Value* GetResultAt(int index) const {
    assert(index < result_indices_.size());
    return registers_[result_indices_[index]];
  }

 protected:
  KernelFrame(llvm::ArrayRef<Value*> registers) : registers_(registers) {}

  // These are indices into `registers_`
  llvm::ArrayRef<uint32_t> argument_indices_;
  llvm::ArrayRef<const void*> attributes_;
  // These are indices into `registers_`
  llvm::ArrayRef<uint32_t> result_indices_;

  const llvm::ArrayRef<Value*> registers_;
};

// SyncKernelFrameBuilder is used by the kernel caller to construct a
// SyncKernelFrame object without exposing the builder methods to the kernel
// implementation.
//
// As an optimization, SyncKernelFrame stores arguments, attributes, and results
// in a single SmallVector. As a result, to initialize a SyncKernelFrame, this
// class requires that the client performs the following actions in order:
// 1. Adds the arguments (using AddArg())
// 2. Add the attributes (using AddAttribute())
// 3. Add the results (using AddResult())
class KernelFrameBuilder : public KernelFrame {
 public:
  explicit KernelFrameBuilder(llvm::ArrayRef<Value*> registers) : KernelFrame(registers) {}

  void SetArguments(llvm::ArrayRef<uint32_t> argument_indices) {
    argument_indices_ = argument_indices_;
  }
  void SetAttributes(llvm::ArrayRef<const void*> attributes) {
    attributes_ = attributes;
  }
  void SetResults(llvm::ArrayRef<uint32_t> result_indices) {
    result_indices_ = result_indices;
  }
};

}  // namespace rt