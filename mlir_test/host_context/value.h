#pragma once

#include "support/type_traits.h"
#include <cassert>
#include <type_traits>
#include <utility>

namespace rt {
namespace internal {
class TypeTraits;
template <typename T>
class OutOfPlaceTypeTraits;
class PointerPayloadTypeTraits;
}  // namespace internal

// Value is a type-erased data type for representing synchronous values and  is
// used for defining synchronous kernels and in TFRT interpreter.
class Value {
public:
  struct PointerPayload {};

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

  // Construct Value that stores a pointer to the payload. With Value(ptr,
  // PointerPayload{}), Value::get() returns a ref to the pointee object. This
  // is unlike Value(ptr) where Value::get() returns a ref to the pointer.
  template <typename T>
  explicit Value(T* t, PointerPayload);

  ~Value();

  // get() function returns the payload of the Value object in the requested
  // type.
  //
  // Dynamic type checking is performed in the debug model.
  template <typename T>
  T& get();

  template <typename T>
  const T& get() const;

  // emplace() constructs the payload object of type T in place with the given
  // args.
  template <typename T, typename... Args>
  void emplace(Args&&... args);

  // set() stores the argument `t` as the payload of Value.
  template <typename T>
  void set(T&& t);

  template <typename T>
  void set(T* t, PointerPayload);

  // Reset the Value object to empty.
  void reset();

  // Check if value contains a payload.
  bool HasValue() const { return traits_; }

  // Check if Value contains object of type T.
  template <typename T>
  bool IsType() const;

private:
  template <typename T, typename... Args>
  void fill(Args&&... args);

private:
  // template <typename T>
  // friend class internal::InPlaceTypetraits;

  template <typename T>
  friend class internal::OutOfPlaceTypeTraits;

  friend class internal::PointerPayloadTypeTraits;

  const internal::TypeTraits* traits_{nullptr};
  void* value_; // Always point to the payload.
};

namespace internal {

// template <typename T>
// struct InPlaceTypetraits {
//   // Clear the payload in `v`. `v` should be non-empty.
//   static void Clear(Value* v) {
//     assert(v->HasValue());

//     T& t = v->get<T>();
//     t.~T();
//     v->traits_ = nullptr;
//   }

//   // Move construct `from` to `to`. `to` shoule be an empty Value and `from`
//   // shoule be a non-empty Value.
//   static void MoveConstruct(Value* to, Value* from) {
//     assert(!to->HasValue() && from->HasValue());

//     T& t = from->get<T>();
//     new (&to) T(std::move(t));

//     to->traits_ = from->traits_;

//     t.~T();
//     from->traits_ = nullptr;
//   }
// };

template <typename T>
struct OutOfPlaceTypeTraits {
  // Clear the payload in `v`. `v` should be non-empty.
  static void Clear(Value* v) {
    assert(v->HasValue());

    T& t = v->get<T>();
    delete &t;
    v->traits_ = nullptr;
  }

  // Move construct `from` to `to`. `to` should be an empty Value and `from`
  // should be a non-empty Value.
  static void MoveConstruct(Value* to, Value* from) {
    assert(!to->HasValue() && from->HasValue());

    T& t = from->get<T>();
    to->value_ = &t;
    to->traits_ = from->traits_;
    from->traits_ = nullptr;
  }
};

struct PointerPayloadTypeTraits {
  // Clear the payload in `v`. `v` should be non-empty.
  static void Clear(Value* v) {
    assert(v->HasValue());
    v->traits_ = nullptr;
  }

  // Move construct `from` to `to`. `to` should be an empty Value and `from`
  // should be a non-empty Value.
  static void MoveConstruct(Value* to, Value* from) {
    assert(!to->HasValue() && from->HasValue());

    to->value_ = from->value_;
    to->traits_ = from->traits_;
    from->traits_ = nullptr;
  }
};

struct TypeTraits {
  using ClearFn = void (*)(Value*);
  using MoveConstructFn = void (*)(Value*, Value*);

  template <typename T>
  TypeTraits(TypeTag<T>) {
    using TypeTraitsFns = OutOfPlaceTypeTraits<T>;
    clear = &TypeTraitsFns::Clear;
    move_construct = &TypeTraitsFns::MoveConstruct;
    is_polymorphic = std::is_polymorphic<T>::value;
    is_pointer_payload = false;
  }

  template <typename T>
  TypeTraits(TypeTag<T>, Value::PointerPayload) {
    using TypeTraitsFns = PointerPayloadTypeTraits;
    clear = &TypeTraitsFns::Clear;
    move_construct = &TypeTraitsFns::MoveConstruct;
    is_polymorphic = std::is_polymorphic<T>::value;
    is_pointer_payload = true;
  }

  ClearFn clear;
  MoveConstructFn move_construct;
  bool is_polymorphic;
  bool is_pointer_payload;
};

template <typename T>
TypeTraits* GetTypeTraits() {
  static TypeTraits* traits = new TypeTraits(TypeTag<T>());
  return traits;
}

template <typename T>
TypeTraits* GetTypeTraits(Value::PointerPayload) {
  static TypeTraits* traits =
      new TypeTraits(TypeTag<T>(), Value::PointerPayload{});
  return traits;
}

} // namespace internal

inline Value::Value(Value&& v) {
  if (v.HasValue()) {
    v.traits_->move_construct(this, &v);
  }
}

inline Value& Value::operator=(Value&& v) { 
  reset();
  if (v.HasValue()) {
    v.traits_->move_construct(this, &v);
  }
  return *this;
}

template <typename T>
Value::Value(T&& t) {
  fill<T>(std::forward<T>(t));
}

template <typename T>
Value::Value(T* t, PointerPayload)
  : value_(t),
    traits_(internal::GetTypeTraits<T>(PointerPayload{})) {}

inline Value::~Value() { reset(); };

template <typename T>
T& Value::get() {
  return *static_cast<T*>(value_);
}

template <typename T>
const T& Value::get() const {
  return *static_cast<const T*>(value_);
}

template <typename T, typename... Args>
void Value::emplace(Args&&... args) {
  reset();
  fill<T>(std::forward<Args>(args)...);
}

template <typename T>
void Value::set(T&& t) {
  emplace<T>(std::forward<T>(t));
}

template <typename T>
void Value::set(T* t, PointerPayload) {
  reset();
  value_ = t;
  traits_ = internal::GetTypeTraits<T>(PointerPayload{});
}

inline void Value::reset() {
  if (!traits_)
    return;
  traits_->clear(this);
}

template <typename T>
bool Value::IsType() const {
  if (traits_->is_pointer_payload) {
    return internal::GetTypeTraits<T>(PointerPayload{}) == traits_;
  } else {
    return internal::GetTypeTraits<T>() == traits_;
  }
}

template <typename T, typename... Args>
void Value::fill(Args&&... args) {
  traits_ = internal::GetTypeTraits<T>();
  value_ = new T(std::forward<Args>(args)...);
}

} // namespace rt