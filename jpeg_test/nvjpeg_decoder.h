#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"
#include <nvjpeg.h>

#include <iostream>

class NvJpegDecoder final {
public:
  NvJpegDecoder(nvjpegOutputFormat_t fmt, int device_id = 0);

  ~NvJpegDecoder();

  // Decode single image.
  bool Decode(const uint8_t* buffer, size_t buffer_size, cv::Mat* image);

  // TODO
  bool Decode(const std::vector<uint8_t*>& images,
              const std::vector<size_t>& lengths, cv::OutputArray& dst);

private:
  static int dev_malloc(void** p, size_t s);
  static int dev_free(void* p);
  static int host_malloc(void** p, size_t s, unsigned int f);
  static int host_free(void* p);

private:
  bool hw_decode_available_;
  int batch_size_;
  int device_id_;

  nvjpegImage_t decode_out_;
  nvjpegImage_t alloc_size_;

  nvjpegDevAllocator_t dev_allocator_;
  nvjpegPinnedAllocator_t pinned_allocator_;
  nvjpegJpegState_t nvjpeg_state_;
  nvjpegHandle_t nvjpeg_handle_;
  cudaStream_t stream_;

  nvjpegJpegState_t nvjpeg_decoupled_state_;
  nvjpegBufferPinned_t pinned_buffers_[2]; // 2 buffers for pipeline
  nvjpegBufferDevice_t device_buffer_;
  nvjpegJpegStream_t jpeg_streams_[2]; // 2 streams for pipeline
  nvjpegDecodeParams_t nvjpeg_decode_params_;
  nvjpegJpegDecoder_t nvjpeg_decoder_;

  nvjpegOutputFormat_t fmt_;
};