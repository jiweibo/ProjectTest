#include <iostream>

#include "opencv2/core.hpp"
#include <jpeglib.h>
#include <setjmp.h>

void ErrorExit(j_common_ptr common_struct_ptr) {
  (*common_struct_ptr->err->output_message)(common_struct_ptr);
  jmp_buf* setjmp_buffer =
      static_cast<jmp_buf*>(common_struct_ptr->client_data);
  longjmp(*setjmp_buffer, 1);
}

void ErrorOutputMessage(j_common_ptr common_struct_ptr) {
  char message_buffer[JMSG_LENGTH_MAX];
  (*common_struct_ptr->err->format_message)(common_struct_ptr, message_buffer);
  // LOG(ERROR) << message_buffer;
  std::cerr << message_buffer << std::endl;
}

class JpegDecoder {
public:
  JpegDecoder() {
    decompress_struct_.err = jpeg_std_error(&error_manager_);
    error_manager_.error_exit = ErrorExit;
    error_manager_.output_message = ErrorOutputMessage;
    jpeg_create_decompress(&decompress_struct_);
  }

  ~JpegDecoder() { jpeg_destroy_decompress(&decompress_struct_); }

  bool Decode(const uint8_t* buffer, int buffer_size, cv::Mat* image) {
    if (buffer_size == 0) {
      return false;
    }

    jmp_buf setjmp_buffer;
    decompress_struct_.client_data = &setjmp_buffer;

    if (setjmp(setjmp_buffer)) {
      jpeg_abort_decompress(&decompress_struct_);
      return false;
    }

    jpeg_mem_src(&decompress_struct_, const_cast<uint8_t*>(buffer),
                 buffer_size);

    // Read start of the datastream to obtain image information.
    if (!jpeg_read_header(&decompress_struct_, FALSE))
      return false;

    // Set the output image colorspace to rgb.
    decompress_struct_.out_color_space = JCS_EXT_BGR;

    if (!jpeg_start_decompress(&decompress_struct_))
      return false;

    int row_stride =
        decompress_struct_.output_width * decompress_struct_.output_components;
    image->create(decompress_struct_.output_height,
                  decompress_struct_.output_width, CV_8UC3);
    unsigned char* image_buffer = image->data;
    if (!image_buffer)
      return false;

    while (decompress_struct_.output_scanline <
           decompress_struct_.output_height) {
      jpeg_read_scanlines(&decompress_struct_, &image_buffer,
                          1 /* number of lines */);
      image_buffer += row_stride;
    }

    if (!jpeg_finish_decompress(&decompress_struct_))
      return false;

    return true;
  }

private:
  struct jpeg_error_mgr error_manager_;
  struct jpeg_decompress_struct decompress_struct_;
};
