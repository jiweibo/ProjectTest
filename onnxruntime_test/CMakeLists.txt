# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Copyright wilber.

cmake_minimum_required(VERSION 3.13)

project(ort_samples CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")

# onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

if(NOT ONNXRUNTIME_ROOTDIR)
  include_directories("/usr/local/include/onnxruntime")
endif()

include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA)
  include_directories("/usr/local/cuda/include")
  link_directories("/usr/local/cuda/lib64")
endif()
if(onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
  include_directories(
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session/
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/providers/tensorrt/
  )
endif()

add_executable(ort_sample main.cc)
target_link_libraries(ort_sample onnxruntime)

if(onnxruntime_USE_CUDA)
target_link_libraries(ort_sample nvToolsExt cudart)
endif()