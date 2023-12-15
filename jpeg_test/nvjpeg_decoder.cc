#include "nvjpeg_decoder.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"

#include <chrono>

#include <cstddef>
#include <cuda_runtime_api.h>
#include <thread>

namespace {
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::cout << "CUDA Runtime failure: '#" << _e << " "                     \
                << cudaGetErrorString(_e) << "' at " << __FILE__ << ":"        \
                << __LINE__ << std::endl;                                      \
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
    : device_id_(device_id), fmt_(fmt) {
  CHECK_CUDA(cudaSetDevice(device_id_));
  dev_allocator_ = {&NvJpegDecoder::dev_malloc, &NvJpegDecoder::dev_free};
  pinned_allocator_ = {&NvJpegDecoder::host_malloc, &NvJpegDecoder::host_free};
  CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator_,
                              &pinned_allocator_, NVJPEG_FLAGS_DEFAULT,
                              &nvjpeg_handle_));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; ++c) {
    decode_out_.channel[c] = nullptr;
    decode_out_.pitch[c] = 0;
    alloc_size_.pitch[c] = 0;
  }
}

NvJpegDecoder::~NvJpegDecoder() {
  CHECK_CUDA(cudaSetDevice(device_id_));
  for (auto& it : nvjpeg_per_thread_data_) {
    DestroyParams(it.second);
  }
  nvjpeg_per_thread_data_.clear();
  CHECK_CUDA(cudaStreamSynchronize(stream_));
  CHECK_CUDA(cudaStreamDestroy(stream_));
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; ++c) {
    if (decode_out_.channel[c]) {
      CHECK_CUDA(cudaFree(decode_out_.channel[c]));
    }
    alloc_size_.pitch[c] = 0;
  }
}

void NvJpegDecoder::InitParams(DecodePerThreadParams& params) {
  CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle_, NVJPEG_BACKEND_GPU_HYBRID,
                                   &params.nvjpeg_dec_gpu));
  CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle_, NVJPEG_BACKEND_HYBRID,
                                   &params.nvjpeg_dec_cpu));
  CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpeg_handle_, params.nvjpeg_dec_gpu,
                                        &params.dec_state_gpu));
  CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpeg_handle_, params.nvjpeg_dec_cpu,
                                        &params.dec_state_cpu));
  CHECK_NVJPEG(
      nvjpegBufferDeviceCreate(nvjpeg_handle_, NULL, &params.device_buffer));
  CHECK_NVJPEG(
      nvjpegBufferPinnedCreate(nvjpeg_handle_, NULL, &params.pinned_buffer));
  CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle_, &params.jpeg_stream));
  CHECK_NVJPEG(
      nvjpegDecodeParamsCreate(nvjpeg_handle_, &params.nvjpeg_decode_params));
  CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(params.dec_state_cpu,
                                             params.device_buffer));
  CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(params.dec_state_gpu,
                                             params.device_buffer));
}

void NvJpegDecoder::DestroyParams(DecodePerThreadParams& params) {
  CHECK_NVJPEG(nvjpegDecodeParamsDestroy(params.nvjpeg_decode_params));
  CHECK_NVJPEG(nvjpegJpegStreamDestroy(params.jpeg_stream));
  CHECK_NVJPEG(nvjpegBufferPinnedDestroy(params.pinned_buffer));
  CHECK_NVJPEG(nvjpegBufferDeviceDestroy(params.device_buffer));
  CHECK_NVJPEG(nvjpegJpegStateDestroy(params.dec_state_gpu));
  CHECK_NVJPEG(nvjpegDecoderDestroy(params.nvjpeg_dec_gpu));
}

bool NvJpegDecoder::Decode(const uint8_t* buffer, size_t buffer_size,
                           cv::Mat* image) {
  if (buffer_size == 0)
    return false;

  auto tid = std::this_thread::get_id();
  if (!nvjpeg_per_thread_data_.count(tid)) {
    nvjpeg_per_thread_data_[tid] = DecodePerThreadParams();
    InitParams(nvjpeg_per_thread_data_[tid]);
  }

  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;
  CHECK_NVJPEG(nvjpegGetImageInfo(nvjpeg_handle_, buffer, buffer_size,
                                  &channels, &subsampling, widths, heights));
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

  auto& per_thread_params = nvjpeg_per_thread_data_[tid];
  CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(
      per_thread_params.nvjpeg_decode_params, fmt_));
  CHECK_NVJPEG(nvjpegJpegStreamParse(nvjpeg_handle_,
                                     (const unsigned char*)buffer, buffer_size,
                                     0, 0, per_thread_params.jpeg_stream));
  nvjpegJpegDecoder_t& decoder = per_thread_params.nvjpeg_dec_gpu;
  nvjpegJpegState_t& decoder_state = per_thread_params.dec_state_gpu;
  CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(decoder_state,
                                             per_thread_params.pinned_buffer));
  CHECK_NVJPEG(nvjpegDecodeJpegHost(nvjpeg_handle_, decoder, decoder_state,
                                    per_thread_params.nvjpeg_decode_params,
                                    per_thread_params.jpeg_stream));
  CHECK_CUDA(cudaStreamSynchronize(stream_));
  CHECK_NVJPEG(
      nvjpegDecodeJpegTransferToDevice(nvjpeg_handle_, decoder, decoder_state,
                                       per_thread_params.jpeg_stream, stream_));
  CHECK_NVJPEG(nvjpegDecodeJpegDevice(nvjpeg_handle_, decoder, decoder_state,
                                      &decode_out_, stream_));

  if (fmt_ == NVJPEG_OUTPUT_BGRI || fmt_ == NVJPEG_OUTPUT_RGBI) {
    CHECK_CUDA(cudaMemcpyAsync(image->data, decode_out_.channel[0],
                               heights[0] * widths[0] * 3,
                               cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaStreamSynchronize(stream_));
  } else if (fmt_ == NVJPEG_OUTPUT_BGR || fmt_ == NVJPEG_OUTPUT_RGB) {
    CHECK_CUDA(cudaStreamSynchronize(stream_));
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
