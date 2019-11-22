cmake_minimum_required(VERSION 3.10)
project(gflags-example CXX)
find_package(gflags REQUIRED)

add_executable(gflags-test test-gflag.cc)

target_link_libraries(gflags-test gflags)
