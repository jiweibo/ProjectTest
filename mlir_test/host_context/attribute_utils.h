#pragma once

#include <string>
#include "host_context/value.h"

namespace rt {

template <typename T>
class Attribute {
 public:
  explicit Attribute(const void* value) : value_(*reinterpret_cast<const T*>(value)) {}

  const T& get() const { return value_; }
  const T* operator->() const { return &value_; }
  const T& operator*() const { return value_; }

 private:
  static_assert(!std::is_same<T, std::string>, "Use StringAttribute instead of Attribute<std::string>");
  const T& value_;
};

}  // namespace rt