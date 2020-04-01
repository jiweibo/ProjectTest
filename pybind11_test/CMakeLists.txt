cmake_minimum_required(VERSION 2.8)
project(cmake_example)

find_package(pybind11 REQUIRED)
pybind11_add_module(cmake_example src/main.cpp)
