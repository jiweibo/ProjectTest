#pragma once

#include "reply.h"
#include "request.h"
#include "request_handler.h"
#include "request_parser.h"
#include <boost/asio/ip/tcp.hpp>
#include <memory>

#include <boost/asio.hpp>

namespace http {
namespace server {

class ConnectionManager;

class Connection : public std::enable_shared_from_this<Connection> {
public:
  Connection(const Connection&) = delete;
  Connection& operator=(const Connection&) = delete;

  /// Construct a connection with the given socket.
  explicit Connection(boost::asio::ip::tcp::socket socket,
                      ConnectionManager& manager, RequestHandler& handler);

  /// Start the first asynchronous operation for the connection.
  void Start();

  /// Stop all asynchronous operations associated with the connection.
  void Stop();

private:
  /// Perform an asynchronous read operation.
  void do_read();

  /// Perform an asynchronous write operation.
  void do_write();

private:
  /// Socket for the connection.
  boost::asio::ip::tcp::socket socket_;
  /// The manager for this connection.
  ConnectionManager& connection_manager_;
  /// Buffer for incoming data.
  std::array<char, 8192> buffer_;
  /// The handler used to process the incoming request.
  RequestHandler& request_handler_;
  /// The incoming request.
  Request request_;
  /// The parser for the incoming request.
  RequestParser request_parser_;
  /// The reply to be sent back to the client.
  Reply reply_;
};

using ConnectionPtr = std::shared_ptr<Connection>;

} // namespace server
} // namespace http