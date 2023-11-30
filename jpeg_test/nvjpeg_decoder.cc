#include "nvjpeg_decoder.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"

#include <chrono>

#include <cuda_runtime_api.h>

namespace {
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::cout << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__     \
                << ":" << __LINE__ << std::endl;                               \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CHECK_NVJPEG(call)                                                     \
  {                                                                            \
    nvjpegStatus_t _e = (call);                                                \
    if (_e != NVJPEG_STATUS_SUCCESS) {                                         \
      std::cout << "NVJPEG failure: '#'" << _e << "' at " << __FILE__ << ":"   \
                << __LINE__ << std::endl;                                      \
    }                                                                          \
  }
} // namespace

NvJpegDecoder::NvJpegDecoder(nvjpegOutputFormat_t fmt, int device_id)
    : fmt_(fmt), device_id_(device_id) {

  CHECK_CUDA(cudaSetDevice(device_id_));
  dev_allocator_ = {&NvJpegDecoder::dev_malloc, &NvJpegDecoder::dev_free};
  pinned_allocator_ = {&NvJpegDecoder::host_malloc, &NvJpegDecoder::host_free};
  CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator_,
                              &pinned_allocator_, NVJPEG_FLAGS_DEFAULT,
                              &nvjpeg_handle_));

  CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle_, &nvjpeg_state_));

  CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle_, NVJPEG_BACKEND_DEFAULT,
                                   &nvjpeg_decoder_));
  CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpeg_handle_, nvjpeg_decoder_,
                                        &nvjpeg_decoupled_state_));
  CHECK_NVJPEG(
      nvjpegBufferPinnedCreate(nvjpeg_handle_, nullptr, &pinned_buffers_[0]));
  CHECK_NVJPEG(
      nvjpegBufferPinnedCreate(nvjpeg_handle_, nullptr, &pinned_buffers_[1]));
  CHECK_NVJPEG(
      nvjpegBufferDeviceCreate(nvjpeg_handle_, nullptr, &device_buffer_));
  CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_streams_[0]));
  CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_streams_[1]));
  CHECK_NVJPEG(
      nvjpegDecodeParamsCreate(nvjpeg_handle_, &nvjpeg_decode_params_));

  CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

  for (int c = 0; c < NVJPEG_MAX_COMPONENT; ++c) {
    decode_out_.channel[c] = nullptr;
    decode_out_.pitch[c] = 0;
    alloc_size_.pitch[c] = 0;
  }
}

NvJpegDecoder::~NvJpegDecoder() {
  CHECK_CUDA(cudaSetDevice(device_id_));
  CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params_));
  CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_streams_[0]));
  CHECK_NVJPEG(nvjpegJpegStreamDestroy(jpeg_streams_[1]));
  CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffers_[0]));
  CHECK_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffers_[1]));
  CHECK_NVJPEG(nvjpegBufferDeviceDestroy(device_buffer_));
  CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoupled_state_));
  CHECK_NVJPEG(nvjpegDecoderDestroy(nvjpeg_decoder_));
  CHECK_CUDA(cudaStreamDestroy(stream_));
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; ++c) {
    if (decode_out_.channel[c]) {
      CHECK_CUDA(cudaFree(decode_out_.channel[c]));
    }
    alloc_size_.pitch[c] = 0;
  }
}

bool NvJpegDecoder::Decode(const uint8_t* buffer, size_t buffer_size,
                           cv::Mat* image) {
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;
  nvjpegGetImageInfo(nvjpeg_handle_, buffer, buffer_size, &channels,
                     &subsampling, widths, heights);
  // std::cout << "Image is " << channels << " channels." << std::endl;
  // for (int c = 0; c < channels; c++) {
  //   std::cout << "Channel #" << c << " size: " << widths[c] << " x "
  //             << heights[c] << std::endl;
  // }
  // switch (subsampling) {
  // case NVJPEG_CSS_444:
  //   std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
  //   break;
  // case NVJPEG_CSS_440:
  //   std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
  //   break;
  // case NVJPEG_CSS_422:
  //   std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
  //   break;
  // case NVJPEG_CSS_420:
  //   std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
  //   break;
  // case NVJPEG_CSS_411:
  //   std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
  //   break;
  // case NVJPEG_CSS_410:
  //   std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
  //   break;
  // case NVJPEG_CSS_GRAY:
  //   std::cout << "Grayscale JPEG " << std::endl;
  //   break;
  // case NVJPEG_CSS_UNKNOWN:
  //   std::cout << "Unknown chroma subsampling" << std::endl;
  //   return EXIT_FAILURE;
  // }

  int mul = 1;
  // in the case of interleaved RGB output, write only to single channel, but
  // 3 samples at once
  if (fmt_ == NVJPEG_OUTPUT_RGBI || fmt_ == NVJPEG_OUTPUT_BGRI) {
    channels = 1;
    mul = 3;
  } else if (fmt_ == NVJPEG_OUTPUT_RGB || fmt_ == NVJPEG_OUTPUT_BGR) {
    channels = 3;
    widths[1] = widths[2] = widths[0];
    heights[1] = heights[2] = heights[0];
  }

  image->create(heights[0], widths[0], CV_8UC3);

  for (int c = 0; c < channels; ++c) {
    int aw = mul * widths[c];
    int ah = heights[c];
    int sz = aw * ah;
    decode_out_.pitch[c] = aw;
    if (sz > alloc_size_.pitch[c]) {
      if (decode_out_.channel[c]) {
        CHECK_CUDA(cudaFree(decode_out_.channel[c]));
      }
      CHECK_CUDA(cudaMalloc((void**)&(decode_out_.channel[c]), sz));
      alloc_size_.pitch[c] = sz;
    }
  }

  const bool pipeline = false;
  if (pipeline) {
    CHECK_NVJPEG(
        nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state_, device_buffer_));
    int buffer_index = 0;
    CHECK_NVJPEG(
        nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params_, fmt_));
    CHECK_NVJPEG(nvjpegJpegStreamParse(nvjpeg_handle_, buffer, buffer_size, 0,
                                       0, jpeg_streams_[buffer_index]));
    CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state_,
                                               pinned_buffers_[buffer_index]));
    CHECK_NVJPEG(nvjpegDecodeJpegHost(
        nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decoupled_state_,
        nvjpeg_decode_params_, jpeg_streams_[buffer_index]));
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(
        nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decoupled_state_,
        jpeg_streams_[buffer_index], stream_));
    CHECK_NVJPEG(nvjpegDecodeJpegDevice(nvjpeg_handle_, nvjpeg_decoder_,
                                        nvjpeg_decoupled_state_, &decode_out_,
                                        stream_));
  } else {
    CHECK_NVJPEG(nvjpegDecode(nvjpeg_handle_, nvjpeg_state_, buffer,
                              buffer_size, fmt_, &decode_out_, stream_));
  }

  if (fmt_ == NVJPEG_OUTPUT_BGRI || fmt_ == NVJPEG_OUTPUT_RGBI) {
    cudaMemcpyAsync(image->data, decode_out_.channel[0],
                    heights[0] * widths[0] * 3, cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronize(stream_);
  } else if (fmt_ == NVJPEG_OUTPUT_BGR || fmt_ == NVJPEG_OUTPUT_RGB) {
    cudaStreamSynchronize(stream_);
    cv::cuda::GpuMat g1(heights[0], widths[0], CV_8UC1, decode_out_.channel[0]);
    cv::cuda::GpuMat g2(heights[0], widths[0], CV_8UC1, decode_out_.channel[1]);
    cv::cuda::GpuMat g3(heights[0], widths[0], CV_8UC1, decode_out_.channel[2]);
    std::vector<cv::cuda::GpuMat> channel_mats;
    channel_mats.push_back(g1);
    channel_mats.push_back(g2);
    channel_mats.push_back(g3);
    cv::cuda::GpuMat result(heights[0], widths[0], CV_8UC3);
    cv::cuda::merge(channel_mats, result);
    result.download(*image);
  } else {
    std::cerr << "Not supported format";
  }
  return true;
}

int NvJpegDecoder::dev_malloc(void** p, size_t s) {
  return (int)cudaMalloc(p, s);
}

int NvJpegDecoder::dev_free(void* p) { return (int)cudaFree(p); }

int NvJpegDecoder::host_malloc(void** p, size_t s, unsigned int f) {
  return (int)cudaHostAlloc(p, s, f);
}

int NvJpegDecoder::host_free(void* p) { return (int)cudaFreeHost(p); }
