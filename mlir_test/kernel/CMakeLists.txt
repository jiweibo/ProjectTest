add_library(kernel float_kernels.cc)

add_executable(test_float_kernels test_float_kernels.cc)
target_link_libraries(test_float_kernels kernel host_context ${llvm_libs})