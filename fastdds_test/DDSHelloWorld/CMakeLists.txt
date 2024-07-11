cmake_minimum_required(VERSION 3.20)

project(DDSHelloWorld)

set(fastcdr_ROOT CACHE PATH "" "fastcdr root dir")
set(fastrtps_ROOT CACHE PATH "" "fastdds root dir")
set(foonathan_memory_vendor_ROOT CACHE PATH "" "foonathan_memory_vendor root dir")

include_directories(${fastcdr_ROOT}/include)
include_directories(${fastrtps_ROOT}/include)
include_directories(${foonathan_memory_vendor_ROOT}/include)

find_package(fastcdr 2 REQUIRED)
find_package(fastrtps 2 REQUIRED)

set(CMAKE_CXX_STANDARD 11)

file(GLOB DDS_HELLOWORLD_SOURCES_CXX "src/*.cxx")

add_executable(DDSHelloWorldPublisher src/HelloWorldPublisher.cpp ${DDS_HELLOWORLD_SOURCES_CXX})
add_executable(DDSHelloWorldSubscriber src/HelloWorldSubscriber.cpp ${DDS_HELLOWORLD_SOURCES_CXX})

target_link_libraries(DDSHelloWorldPublisher fastrtps fastcdr)
target_link_libraries(DDSHelloWorldSubscriber fastrtps fastcdr)
