#pragma once

#include "host_context/attribute_utils.h"
#include "host_context/kernel_frame.h"
#include "host_context/value.h"

#include <llvm/ADT/ArrayRef.h>

#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>

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

// RemainingArguments collects all remaining arguments in an ArrayRef. There
// can be at most one RemainingArguments, and it must appear after all other
// Arguments.
class RemainingArguments {
 public:
  RemainingArguments(llvm::ArrayRef<uint32_t> reg_indices,
                     llvm::ArrayRef<Value*> registers) : reg_indices_{reg_indices}, registers_{registers} {}

  size_t size() const { return reg_indices_.size(); }
  Value* operator[](size_t i) { return registers_[reg_indices_[i]]; }
 private:
  llvm::ArrayRef<uint32_t> reg_indices_;
  llvm::ArrayRef<Value*> registers_;
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
  // template <typename T>
  // using UnderlyingT = typename T::UnderlyingT;

  // template <typename T>
  // using IsViewT = is_detected<UnderlyingT, T>;

  // Casts the return value of the kernel, if non-void. Otherwise ignores the
  // return value
  template <typename T, typename Enbale = void>
  struct KernelReturnHelper {
    static void Invoke(KernelFrame* frame, const Args&... args) {
      HandleReturn(frame, impl_fn(args...));
    }
  };

  // Specialize for the case when T is void.
  template <typename T>
  struct KernelReturnHelper<T, std::enable_if_t<std::is_same<T, void>::value>> {
    static void Invoke(KernelFrame* frame, const Args&... args) {
      impl_fn(args...);
    }
  };

  // Store result as a Value output in KernelFrame.
  template <typename T>
  static void StoreResultAt(KernelFrame* frame, int index, T&& t) {
    frame->EmplaceResultAt<std::decay_t<T>>(index, std::forward<T>(t));
  }

  // Store the function result back to the output Value in the
  // KernelFrame.
  template <typename T> 
  static void HandleReturn(KernelFrame* frame, T&& t) {
    assert(frame->GetNumResults() == 1 && "Extra results passes to kernel.");
    StoreResultAt(frame, index, std::forward<T>(t));
  }

  // For kernel function that return std::pair<>, store the result as the first
  // and second output Value in the KernelFrame.
  template <typename T1, typename T2> 
  static void HandleReturn(KernelFrame* frame, std::pair<T1, T2>&& t) {
    assert(frame->GetNumResults() == 2 && "Incorrect number of results passes to kernel.");
    StoreResultAt(frame, 0, std::move(t.first));
    StoreResultAt(frame, 1, std::move(t.second));
  }

  // For kernel function that return std::tuple<>, store the results in order
  // as the output Values in the KernelFrame.
  template <typename... T>
  static void HandleReturn(KernelFrame* frame, std::tuple<T...>&& t) {
    assert(frame->GetNumResults() == sizeof...(T) && "Incorrect number of results passes to kernel.");
    EmplaceTupleResult(frame, std::move(t), std::make_index_sequence<sizeof...(T)>{});
  }

  // Helper function for storing multiple return values in std::tuple<> as 
  // output Value in KernelFrame.
  template <typename TupleT, size_t... I>
  static void EmplaceTupleResult(KernelFrame* frame, TupleT&& result, std::index_sequence<I...>) {
    // Use braced-init-list to retrieve the results in the tuple sequence.
    // TODO(wilber): why?
    std::ignore = std::initializer_list<int>{
      (StoreResultAt(frame, I, std::get<I>(std::forward<TupleT>(result))), 0)...
    };
  }

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
      // auto&& arg = GetArg<ArgT>(value, IsViewT<ArgT>());
      auto&& arg = GetArg<ArgT>(value, std::false_type());

      KernelCallHelper<Tail...>::template Invoke<arg_idx + 1, attr_idx>(
          frame, pargs..., arg);
    }
  };

  // RemainingArguments provides an ArrayRef<Value*> containing all 
  // remaining arguments. Useful for variadic kernels.
  template <typename... Tail>
  struct KernelCallHelper<RemainingArguments, Tail...> {
    template <int arg_idx, int attr_idx, typename... PreviousArgs>
    static void Invoke(KernelFrame* frame, const PreviousArgs&... pargs) {
      static_assert(
          arg_idx != -1,
          "Do not use more than one RemainingArguments/RepeatedArguments");
      static_assert(attr_idx == 0,
                    "Arguments and results should appear before attributes.");

      RemainingArguments remaining_arguments(frame->GetArguments().drop_front(arg_idx), frame->GetRegisters());
      KernelCallHelper<Tail...>::template Invoke<-1, attr_idx>(frame, pargs..., remaining_arguments);
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
      KernelReturnHelper<Return>::Invoke(frame, pargs...);
    }
  };
};

} // namespace internal

} // namespace rt