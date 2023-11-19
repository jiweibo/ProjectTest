
#include "connection.h"
#include "connection_manager.h"
#include "reply.h"
#include "request_handler.h"
#include "request_parser.h"

#include <boost/asio/error.hpp>
#include <boost/asio/write.hpp>
#include <boost/system/detail/error_code.hpp>
#include <tuple>
#include <iostream>

namespace http {
namespace server {

Connection::Connection(boost::asio::ip::tcp::socket socket,
                       ConnectionManager& manager, RequestHandler& handler)
    : socket_(std::move(socket)), connection_manager_(manager),
      request_handler_(handler) {}

void Connection::Start() { do_read(); }

void Connection::Stop() { socket_.close(); }

void Connection::do_read() {
  auto self(shared_from_this());
  socket_.async_read_some(
      boost::asio::buffer(buffer_),
      [this, self](boost::system::error_code ec,
                   std::size_t bytes_transferred) {
        if (!ec) {
          RequestParser::ResultType result;
          std::cout << "bytes_transferred " << bytes_transferred << std::endl;
          std::cout.write(buffer_.data(), bytes_transferred);
          std::tie(result, std::ignore) = request_parser_.Parse(
              request_, buffer_.data(), buffer_.end() + bytes_transferred);
          std::cout << "req is " << request_.method << ", " << request_.uri << ", " << request_.http_version_major << ", " << request_.http_version_minor << std::endl;
          for (auto h : request_.headers) {
            std::cout << h.name << ", " << h.value << std::endl;
          }
          if (result == RequestParser::ResultType::GOOD) {
            request_handler_.HandleRequest(request_, reply_);
            do_write();
          } else if (result == RequestParser::ResultType::BAD) {
            reply_ = Reply::stock_reply(Reply::StatusType::bad_request);
            do_write();
          } else {
            do_read();
          }
        } else if (ec != boost::asio::error::operation_aborted) {
          connection_manager_.Stop(shared_from_this());
        }
      });
}

void Connection::do_write() {
  auto self(shared_from_this());
  boost::asio::async_write(
      socket_, reply_.to_buffers(),
      [this, self](boost::system::error_code ec, std::size_t) {
        if (!ec) {
          // Initiate graceful connection closure.
          boost::system::error_code ignored_ec;
          socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both,
                           ignored_ec);
        }
        if (ec != boost::asio::error::operation_aborted) {
          connection_manager_.Stop(shared_from_this());
        }
      });
}

} // namespace server
} // namespace http