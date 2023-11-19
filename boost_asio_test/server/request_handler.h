#pragma once

#include "reply.h"
#include "request.h"
#include <string>

namespace http {
namespace server {

class RequestHandler {
public:
  RequestHandler(const RequestHandler&) = delete;
  RequestHandler& operator=(const RequestHandler&) = delete;

  /// Construct with a directory containing files to be served.
  explicit RequestHandler(const std::string& doc_root);

  /// Handle a request and produce a reply
  void HandleRequest(const Request& req, Reply& rep);

private:
  /// The directory containing the files to be served.
  std::string doc_root_;

  /// Perform URL-decoding on a string. Returns false if the encoding was
  /// invalid.
  static bool url_decode(const std::string& in, std::string& out);
};

} // namespace server
} // namespace http