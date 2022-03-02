project(RTMLIR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required(VERSION 3.16)

set(LLVM_PATH "" CACHE PATH "LLVM_PATH")

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
get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)

llvm_map_components_to_libnames(llvm_libs Support Core irreader
        X86 executionengine orcjit mcjit all codegen)

set(LLVM_TARGET_DEFINITIONS rt_ops.td)
mlir_tablegen(rt_ops.h.inc -gen-op-decls)
mlir_tablegen(rt_ops.cpp.inc -gen-op-defs)
mlir_tablegen(rt_base.h.inc -gen-dialect-decls -dialect=rt)
mlir_tablegen(rt_base.cpp.inc -gen-dialect-defs -dialect=rt)
mlir_tablegen(rt_types.h.inc -gen-typedef-decls)
mlir_tablegen(rt_types.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRRTIncGen)

set(LLVM_TARGET_DEFINITIONS rt_pass.td)
mlir_tablegen(rt_pass.inc -gen-rewriters)
add_public_tablegen_target(MLIRRTPassIncGen)
add_custom_target(rt_base_inc DEPENDS MLIRRTIncGen MLIRRTPassIncGen)


set(rt_srcs rt_base.cc rt_ops.cc rt_pass.cc)
add_library(rt SHARED ${rt_srcs})
target_link_libraries(rt ${mlir_libs})
add_dependencies(rt rt_base_inc)

add_executable(rtopt opt.cc init_rt_dialects.cc)
target_link_libraries(rtopt rt)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(host_context)
add_subdirectory(kernel)

#if (WITH_TESTING)
#  add_library(gtest_main gtest_main.cc)
#  target_link_libraries(gtest_main gtest)
#endif()