#pragma once

#include <string>
#include <vector>

#include "header.h"

namespace http {
namespace server {

/// A request received from a client.
struct Request {
  std::string method;
  std::string uri;
  int http_version_major;
  int http_version_minor;
  std::vector<header> headers;
};

} // namespace server
} // namespace http