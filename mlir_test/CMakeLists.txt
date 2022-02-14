project(RTMLIR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required(VERSION 3.16)

set(LLVM_PATH "/wilber/repo/llvm-project/build")
set(LLVM_DIR ${LLVM_PATH}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_PATH}/lib/cmake/mlir)
find_package(LLVM REQUIRED CONFIG HINTS ${LLVM_DIR})
find_package(MLIR REQUIRED CONFIG HINTS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS rt_ops.td)
mlir_tablegen(rt_ops.h.inc -gen-op-decls)
mlir_tablegen(rt_ops.cpp.inc -gen-op-defs)
mlir_tablegen(rt_base.h.inc -gen-dialect-decls -dialect=rt)
mlir_tablegen(rt_base.cpp.inc -gen-dialect-defs -dialect=rt)
add_public_tablegen_target(MLIRRTIncGen)
add_custom_target(rt_base_inc DEPENDS MLIRRTIncGen)

add_library(rt_base rt_base.cc)
add_dependencies(rt_base rt_base_inc)
