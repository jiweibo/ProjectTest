#pragma once

#include <memory>
#include <functional>

#include "llvm/ADT/StringRef.h"

namespace rt {

class KernelFrame;

using KernelImplementation = std::function<void(KernelFrame* frame)>;

// This represents a mapping between the names of the MLIR opcodes to the
// implementations of those functions, along with type mappings.
class KernelRegistry {
 public:
  KernelRegistry();
  ~KernelRegistry();

  void AddKernel(llvm::StringRef name, KernelImplementation fn);
  KernelImplementation GetKernel(llvm::StringRef name) const;
 private:
  // KernelRegistry();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Use this macro to add a function that will register kernels that are
// statically linked in the binary. FUNC should be a function pointer with the
// prototype given by the rt::KernelRegistration alias.
#define RT_STATIC_KERNEL_REGISTRATION(FUNC) \
  RT_STATIC_KERNEL_REGISTRATION_(FUNC, __COUNTER__)
#define RT_STATIC_KERNEL_REGISTRATION_(FUNC, N) \
  static bool rt_static_kernel_##N##_registered_ = []() { \
    ::rt::AddStaticKernelRegistration(FUNC);              \
    return true;                                          \
  }()

// The type for kernel registration functions. This is the same as the
// prototype for the entry point function for dynamic plugins.
using KernelRegistration = void (*)(KernelRegistry*);

// This is called to register all the statically linked kernels in the given
// registry.
void RegisterStaticKernels(KernelRegistry* kernel_reg);

// Adds a kernel to the registry. This should not be used directly; use
// RT_STATIC_KERNEL_REGISTRATION instead.
void AddStaticKernelRegistration(KernelRegistration func);

} // namespace rt