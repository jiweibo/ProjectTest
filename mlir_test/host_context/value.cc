#include "value.h"

namespace rt {

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

}  // namespace rt
