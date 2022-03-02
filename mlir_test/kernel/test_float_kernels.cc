#include "host_context/value.h"
#include "kernel/float_kernels.h"

#include "host_context/kernel_frame.h"
#include "host_context/kernel_registry.h"
#include "llvm/ADT/SmallVector.h"
#include <bits/stdint-uintn.h>

int main() {

  rt::KernelRegistry registry;
  rt::RegisterFloatKernels(&registry);

  rt::KernelImplementation kernel_impl = registry.GetKernel("rt.print.f32");


  rt::Value in;
  in.set<float>(6.f);
  rt::KernelFrameBuilder kf({});
  rt::Value out;
  llvm::SmallVector<rt::Value*, 4> registers;
  rt::Value in2;
  in2.set(&kf, rt::Value::PointerPayload{});

  registers.push_back(&in);
  registers.push_back(&in2);
  registers.push_back(&out);

  rt::KernelFrameBuilder kernel_frame(registers);
  std::vector<uint32_t> args{0, 1};
  std::vector<uint32_t> res{2};
  kernel_frame.SetArguments(args);
  kernel_frame.SetResults(res);

  kernel_impl(&kernel_frame);

  return 0;

}