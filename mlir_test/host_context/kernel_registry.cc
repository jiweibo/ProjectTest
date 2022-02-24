#include "kernel_registry.h"

#include <cassert>
#include <memory>
#include <mutex>

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace rt {

struct KernelRegistry::Impl {
  llvm::StringMap<KernelImplementation> implementations;
  // llvm::StringSet<> 
  // std::mutex mu;
};

KernelRegistry::KernelRegistry() : impl_(std::make_unique<Impl>()) {}

KernelRegistry::~KernelRegistry() = default;

void KernelRegistry::AddKernel(llvm::StringRef kernel_name, KernelImplementation fn) {
  bool added = impl_->implementations.try_emplace(kernel_name, fn).second;
  assert(added && "Re-registered existing kernel_name for kernel");
}

KernelImplementation KernelRegistry::GetKernel(llvm::StringRef name) const {
  auto it = impl_->implementations.find(name);
  return it == impl_->implementations.end() ? KernelImplementation() : it->second;
}

static std::vector<KernelRegistration>* GetStaticKernelRegistrations() {
  static std::vector<KernelRegistration>* ret = new std::vector<KernelRegistration>;
  return ret;
}

void AddStaticKernelRegistration(KernelRegistration func) {
  GetStaticKernelRegistrations()->push_back(func);
}

void RegisterStaticKernels(KernelRegistry* kernel_reg) {
  for (auto func : *GetStaticKernelRegistrations()) {
    func(kernel_reg);
  }
}

} // namespace rt
