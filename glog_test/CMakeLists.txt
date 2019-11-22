cmake_minimum_required(VERSION 3.10)

project(glog_demo)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(GLOG REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

add_executable(glog_demo glog_main.cc)
target_link_libraries(glog_demo ${GLOG_LIBRARIES})
