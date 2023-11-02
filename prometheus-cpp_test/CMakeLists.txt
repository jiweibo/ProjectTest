cmake_minimum_required(VERSION 3.10)

project(prometheus_test CXX)

set(PROMETHEUSCPP_DIR "" CACHE PATH "prometheus-cpp install dir")

include_directories(${PROMETHEUSCPP_DIR}/include)
link_directories(${PROMETHEUSCPP_DIR}/lib)

add_executable(demo demo.cc)
target_link_libraries(demo prometheus-cpp-pull prometheus-cpp-core pthread z)