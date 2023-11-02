#include "onnxruntime/core/session/experimental_onnxruntime_cxx_api.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <nvToolsExt.h>

std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; ++i)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Usage: ./ort_sample <onnx_model.onnx>" << std::endl;
  }
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::string model_file = argv[1];

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort-sample");
  Ort::SessionOptions session_options;
  OrtCUDAProviderOptions cuda_options;
  cuda_options.do_copy_in_default_stream = 1;
  cuda_options.has_user_compute_stream = 1;
  cuda_options.user_compute_stream = stream;
  session_options.AppendExecutionProvider_CUDA(cuda_options);

  Ort::Session session = Ort::Session(env, model_file.c_str(), session_options);
  Ort::IoBinding binding(session);
  Ort::AllocatorWithDefaultOptions ort_alloc;
  Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  // Ort::Allocator ort_alloc(session, memory_info);

  // print name/shape of inputs
  std::vector<const char*> input_names;
  std::vector<std::vector<int64_t>> input_shapes;
  for (size_t i = 0; i < session.GetInputCount(); ++i) {
    char* t = session.GetInputName(i, ort_alloc);
    input_names.push_back(strdupa(t));
    input_shapes.push_back(
        session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    ort_alloc.Free(t);
  }
  std::cout << "Input Node Name/Shape (" << input_names.size()
            << "):" << std::endl;
  for (size_t i = 0; i < input_names.size(); ++i) {
    std::cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i])
              << std::endl;
  }

  // print name/shape of outputs
  std::vector<const char*> output_names;
  std::vector<std::vector<int64_t>> output_shapes;
  for (size_t i = 0; i < session.GetOutputCount(); ++i) {
    char* t = session.GetOutputName(i, ort_alloc);
    output_names.push_back(strdupa(t));
    output_shapes.push_back(
        session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    ort_alloc.Free(t);
  }
  std::cout << "Output Node Name/Shape (" << output_names.size()
            << "):" << std::endl;
  for (size_t i = 0; i < output_names.size(); ++i) {
    std::cout << "\t" << output_names[i] << " : "
              << print_shape(output_shapes[i]) << std::endl;
  }

  assert(input_names.size() == 1 && output_names.size() == 1);

  auto input_shape = input_shapes[0];
  int total_number_elements = std::accumulate(
      input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
  std::vector<float> input_tensor_values(total_number_elements);
  std::generate(input_tensor_values.begin(), input_tensor_values.end(),
                [&] { return rand() % 255; });
  // cudaHostRegister(input_tensor_values.data(), input_tensor_values.size() *
  // sizeof(float), cudaHostRegisterDefault);

  Ort::RunOptions run_options;
  try {
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(nullptr);
    std::vector<Ort::Value> output_tensors;
    output_tensors.emplace_back(nullptr);

    for (size_t i = 0; i < 1; ++i) {
      nvtxRangePushA("ort run method 1");
      input_tensors[0] = Ort::Value::CreateTensor<float>(
          memory_info, input_tensor_values.data(), input_tensor_values.size(),
          input_shape.data(), input_shape.size());
      assert(input_tensors[0].IsTensor() &&
             input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() ==
                 input_shape);
      session.Run(run_options, input_names.data(), input_tensors.data(),
                  input_tensors.size(), output_names.data(),
                  output_tensors.data(), output_tensors.size());
      assert(output_tensors.size() == output_names.size() &&
             output_tensors[0].IsTensor());
      std::cout << "output_tensor_shape: "
                << print_shape(
                       output_tensors[0].GetTensorTypeAndShapeInfo().GetShape())
                << std::endl;
      std::cout << "run method 1 done" << std::endl;
      nvtxRangePop();
    }

    for (size_t i = 0; i < 1; ++i) {
      nvtxRangePushA("ort run method 2");

      input_tensors[0] = Ort::Value::CreateTensor<float>(
          memory_info, input_tensor_values.data(), input_tensor_values.size(),
          input_shape.data(), input_shape.size());
      std::vector<Ort::Value> out = session.Run(
          run_options, input_names.data(), input_tensors.data(),
          input_tensors.size(), output_names.data(), output_tensors.size());
      std::cout << "output_tensor_shape: "
                << print_shape(out[0].GetTensorTypeAndShapeInfo().GetShape())
                << std::endl;
      std::cout << "run method 2 done" << std::endl;
      nvtxRangePop();
    }

    for (size_t i = 0; i < 1; ++i) {
      nvtxRangePushA("ort run method 3");
      input_tensors[0] = Ort::Value::CreateTensor<float>(
          memory_info, input_tensor_values.data(), input_tensor_values.size(),
          input_shape.data(), input_shape.size());
      binding.BindInput(input_names[0], input_tensors[0]);
      binding.BindOutput(output_names[0], memory_info);
      // binding.BindOutput(output_names[0], output_tensors[0]);
      session.Run(run_options, binding);
      std::vector<Ort::Value> out = binding.GetOutputValues();
      std::cout << "output_tensor_shape: "
                << print_shape(out[0].GetTensorTypeAndShapeInfo().GetShape())
                << std::endl;
      std::cout << "run method 3 done" << std::endl;
      nvtxRangePop();
    }

  } catch (const Ort::Exception& exception) {
    std::cerr << "ERROR running model inference: " << exception.what()
              << std::endl;
  }

  return 0;
}