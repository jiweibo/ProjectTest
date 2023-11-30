cmake_minimum_required(VERSION 3.10)
project(jpeg_demo CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

set(LIBJPEGTURBO_DIR "" CACHE PATH "libjpeg-turbo path")
set(OPENCV_DIR "" CACHE PATH "opencv path")

include_directories(${LIBJPEGTURBO_DIR}/include)
link_directories(${LIBJPEGTURBO_DIR}/lib)

include_directories(${OPENCV_DIR}/include)
link_directories(${OPENCV_DIR}/lib)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_executable(demo demo.cc nvjpeg_decoder.cc)
target_link_libraries(demo jpeg  pthread tbb cudart z nvjpeg opencv_cudaarithm opencv_imgcodecs opencv_imgproc png opencv_core pthread tiff webp)