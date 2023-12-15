#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"
#include <nvjpeg.h>

#include <iostream>
#include <map>
#include <thread>

class NvJpegDecoder final {
private:
  struct DecodePerThreadParams {
    nvjpegJpegState_t dec_state_cpu;
    nvjpegJpegState_t dec_state_gpu;
    nvjpegBufferPinned_t pinned_buffer;
    nvjpegBufferDevice_t device_buffer;
    nvjpegJpegStream_t jpeg_stream;
    nvjpegDecodeParams_t nvjpeg_decode_params;
    nvjpegJpegDecoder_t nvjpeg_dec_cpu;
    nvjpegJpegDecoder_t nvjpeg_dec_gpu;
  };

public:
  NvJpegDecoder(nvjpegOutputFormat_t fmt, int device_id = 0);

  ~NvJpegDecoder();

  // Decode single image.
  bool Decode(const uint8_t* buffer, size_t buffer_size, cv::Mat* image);

private:
  NvJpegDecoder(const NvJpegDecoder&) = delete;
  NvJpegDecoder& operator=(const NvJpegDecoder&) = delete;
  static int dev_malloc(void** p, size_t s);
  static int dev_free(void* p);
  static int host_malloc(void** p, size_t s, unsigned int f);
  static int host_free(void* p);

  void InitParams(DecodePerThreadParams& params);

  void DestroyParams(DecodePerThreadParams& params);

private:
  int device_id_;

  cudaStream_t stream_;
  nvjpegHandle_t nvjpeg_handle_;
  nvjpegOutputFormat_t fmt_;
  std::map<std::thread::id, DecodePerThreadParams> nvjpeg_per_thread_data_;
  nvjpegImage_t decode_out_;
  nvjpegImage_t alloc_size_;
  nvjpegDevAllocator_t dev_allocator_;
  nvjpegPinnedAllocator_t pinned_allocator_;
};