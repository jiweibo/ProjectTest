#include "value.h"

namespace rt {

template <typename T, typename... Args>
void Value::fill(Args&&... args) {
  value_ = new T(std::forward<Args>(args)...);
}

template <typename T>
Value::Value(T&& t) {
  fill<T>(std::forward<T>(t));
}

inline Value::Value(Value&& v) {
  // if (v.HasValue()) v.traits_->move_construct(this, &v);
}

inline Value& operator=(Value&& v) {
  
}


}  // namespace rt
