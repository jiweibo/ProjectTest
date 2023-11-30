#include "jpeg_decoder.h"
#include "opencv2/core/mat.hpp"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <ios>
#include <ratio>
#include <utility>

#include "nvjpeg_decoder.h"
#include "opencv2/imgcodecs.hpp"

std::pair<void*, size_t> ReadImage(const std::string& file) {
  std::ifstream fs(file, std::ios::in | std::ios::binary);
  if (!fs) {
    std::cerr << "read file failed " << std::endl;
  }
  fs.seekg(0, fs.end);
  size_t length = fs.tellg();
  std::cout << "length is " << length << std::endl;
  fs.seekg(0, fs.beg);
  char* buffer = new char[length];
  fs.read(buffer, length);

  return std::make_pair(buffer, length);
}

int main(int agrc, char** argv) {
  auto [buffer, length] = ReadImage(argv[1]);
  JpegDecoder decoder;
  cv::Mat mat;
  auto res = decoder.Decode(static_cast<const uint8_t*>(buffer), length, &mat);
  if (!res) {
    std::cerr << "Decode failed." << std::endl;
  }

  int repeats = 20;
  int warmup = 5;

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeats; ++i) {
    res = decoder.Decode(static_cast<const uint8_t*>(buffer), length, &mat);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time: "
            << std::chrono::duration<double, std::milli>(t2 - t1).count() /
                   repeats
            << "ms" << std::endl;

  NvJpegDecoder nv_decoder(NVJPEG_OUTPUT_BGRI);

  cv::Mat mat2;
  for (int i = 0; i < warmup; ++i) {
    nv_decoder.Decode(static_cast<const uint8_t*>(buffer), length, &mat2);
  }

  t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeats; ++i) {
    nv_decoder.Decode(static_cast<const uint8_t*>(buffer), length, &mat2);
  }
  t2 = std::chrono::high_resolution_clock::now();
  std::cout << "time: "
            << std::chrono::duration<double, std::milli>(t2 - t1).count() * 1. /
                   repeats
            << "ms" << std::endl;

  cv::imwrite("test_cpu.jpg", mat);
  cv::imwrite("test_gpu.jpg", mat2);
  return 0;
}