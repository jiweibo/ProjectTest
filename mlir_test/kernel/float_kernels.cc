#include "kernel/float_kernels.h"
#include "host_context/kernel_utils.h"
#include "host_context/kernel_registry.h"
#include "host_context/chain.h"
#include "rt_types.h"
#include <cstdio>


namespace rt {

// float kernels
static Chain RTPrintF32(Argument<float> arg, KernelFrame* frame) {
  printf("f32 = %f\n", *arg);
  std::fflush(stdout);
  return Chain();
}

// Registration
void RegisterFloatKernels(KernelRegistry *registry) {
  registry->AddKernel("rt.print.f32", RT_KERNEL(RTPrintF32));
}

} // namespace rt