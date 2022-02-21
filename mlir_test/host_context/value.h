#pragma once

#include <utility>

namespace rt {

// Value is a type-erased data type for representing synchronous values and  is
// used for defining synchronous kernels and in TFRT interpreter.
class Value {
 public:
  
  // Value is default contructable. The payload is unset in the default
  // constructed Value.
  Value() = default;

  // Value is not copyable or copy-assignable.
  Value(const Value&) = delete;
  Value& operator=(const Value&) = delete;

  // Value is movable and move-assignable.
  Value(Value&&);
  Value& operator=(Value&&);

  // Construct Value and store `t` as the payload.
  template <typename T>
  explicit Value(T&& t);

  ~Value();

  // get() function returns the payload of the Value object in the requested
  // type.
  //
  // Dynamic type checking is performed in the debug model.
  template <typename T>
  T& get();

  template <typename T>
  const T& get() const;

  // Check if value contains a payload.
  bool HasValue() const {
    return traits_;
  }

 private:
  template <typename T, typename... Args>
  void fill(Args&&... args);

 private:
  void* value_; // Always point to the payload.
};

namespace internal {

struct TypeTraits {
  using ClearFn = void (*)(Value*);
  using MoveConstructFn = void (*)(Value*, Value*);

  template <typename T>
  TypeTraits(TypeTag<T>) {
    
  }
};

}  // namespace internal


}  // namespace rt