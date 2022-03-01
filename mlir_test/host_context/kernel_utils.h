#pragma once

#include "host_context/attribute_utils.h"
#include "host_context/kernel_frame.h"
#include "host_context/value.h"
#include <type_traits>

namespace rt {

// ===----------------------------------------------------------===//
// Registration helpers used to make kernels eaier to define.
// ===----------------------------------------------------------===//

#define RT_KERNEL(...)                                                         \
  ::rt::RtKernelImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Invoke;

// Kernels should use this so we know the kernel has an argument.
template <typename T>
struct Argument {
public:
  explicit Argument(Value* value) : value_(value) {}

  Value* value() const { return value_; }

  T& get() const { return value_->template get<T>(); }
  T* operator->() const { return &get(); }
  T& operator*() const { return get(); }

private:
  Value* value_;
};

namespace internal {

template <typename F, F f>
struct RtKernelImpl;

template <typename Return, typename... Args, Return (*impl_fn)(Args...)>
struct RtKernelImpl<Return (*)(Args...), impl_fn> {
  // This is the main entry point that gets registered as a kernel.
  static void Invoke(KernelFrame* frame) {
    KernelCallHelper<Args..., TypeTag<int>>::template Invoke<0, 0>(frame);
  }

private:
  // Check whether a type T has an internal UnderlyingT type.
  template <typnemae T>
  using UnderlyingT = typename T::UnderlyingT;

  template <typename T>
  using IsViewT = is_detected<UnderlyingT, T>;

  // Helper that introspects the kernel arguments to derive the signature and
  // cast parts of the KernelFrame to their appropriate type before passing
  // them to impl_fn. Works by recursively unpacking the arguments.
  template <typename... RemainingArgs>
  struct KernelCallHelper;

  // Specialization to cast a single attribute (Head).
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Attribute<Head>, Tail...> {
    template <int arg_idx, int attr_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      Attribute<Head> arg = frame->GetAttributeAt<Head>(attr_idx);
      KernelCallHelper<Tail...>::template Invoke<arg_idx, attr_idx + 1>(
          frame, pargs..., arg);
    }
  };

  // Specialization to cast a single input argument (Head).
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Argument<Head>, Tail...> {
    template <int arg_idx, int attr_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "Arguments and results should appear before atttibutes.");
      Argument<Head> arg(frame->GetArgAt(arg_idx));
      KernelCallHelper<Tail...>::template Invoke<arg_idx + 1, attr_idx>(
          frame, pargs..., arg);
    }
  };

  // Treat other pointer as an Argument.
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Head*, Tail...> {
    template <int arg_idx, int attr_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(attr_idx == 0,
                    "Arguments and results should appear before attributes.");
      auto* arg = &frame->GetArgAt<Head>(arg_idx);
      KernelCallHelper<Tail...>::template Invoke<arg_idx + 1, attr_idx>(
          frame, pargs..., arg);
    }
  };

  // Treat any other type as an Argument.
  template <typename Head, typename... Tail>
  struct KernelCallHelper<Head, Tail...> {
    using ArgT = std::decay_t<Head>;

    template <typename T>
    static T GetArg(Value* value, std::true_type) {
      return T(&value->template get<typename ArgT::UnderlyingT>());
    }

    template <typename T>
    static T& GetArg(Value* value, std::false_type) {
      return value->get<ArgT>();
    }

    template <int arg_idx, int attr_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(
          arg_idx != -1,
          "Do not place Arguments after RemainingArguments/RepeatedArguments");
      static_assert(attr_idx == 0,
                    "Arguments and results should appear before attributes.");
      auto* value = frame->GetArgAt(arg_idx);
      auto&& arg = GetArg<ArgT>(value, IsViewT<ArgT>());

      KernelCallHelper<Tail...>::template Invoke<arg_idx + 1, attr_idx>(
          frame, pargs..., arg);
    }
  };

  // Base case: No arguments left.
  template <typename T>
  struct KernelCallHelper<TypeTag<T>> {
    template <int arg_idx, int attr_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      assert((arg_idx == -1 || arg_idx == frame->GetNumArgs()) &&
             "Extra arguments passed to kernel.");
      assert(attr_idx == frame->GetNumAttributes() &&
             "Extra attributes passed to kernel.");
      // TODO(wilber): return helper.
      // KernelReturnHelper<Return>::Invoke()
    }
  };
};

} // namespace internal

} // namespace rt