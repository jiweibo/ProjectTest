cmake_minimum_required(VERSION 3.16)
project(http_server CXX)

file(GLOB SRCS "*.cc")

add_executable(http_server ${SRCS})
target_link_libraries(http_server pthread)
