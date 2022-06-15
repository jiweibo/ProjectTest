#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <torch/types.h>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
    module.to(at::kCUDA);

    std::vector<torch::jit::IValue> inputs;
    // FP16 inference
    auto option =
        at::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    // FP32 inference
    // auto option = at::TensorOptions().device(torch::kCUDA, 0);
    inputs.push_back(torch::ones({1, 3, 224, 224}).to(option));

    at::Tensor output = module.forward(inputs).toTensor(); //.to(at::kCPU);
    std::cout << output.slice(1, 0, 5) << "\n";
  } catch (const c10::Error& e) {
    std::cerr << "error.\n";
    return -1;
  }

  std::cout << "ok\n";
}
