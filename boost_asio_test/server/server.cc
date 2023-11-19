#include "server.h"
#include "connection_manager.h"
#include "request_handler.h"
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/system/detail/error_code.hpp>

namespace http {
namespace server {

Server::Server(const std::string& address, const std::string& port,
               const std::string& doc_root)
    : io_context_(1), signals_(io_context_), acceptor_(io_context_),
      connection_manager_(), request_handler_(doc_root) {
  // Register to handle the signals that indicate when the server should exit.
  // It is safe to register for the same signal multiple times in a program,
  // provided all registration for the specified signal is made through Asio.
  signals_.add(SIGINT);
  signals_.add(SIGTERM);
  signals_.add(SIGQUIT);

  do_await_stop();

  boost::asio::ip::tcp::resolver resolver(io_context_);
  boost::asio::ip::tcp::endpoint endpoint =
      *resolver.resolve(address, port).begin();
  acceptor_.open(endpoint.protocol());
  acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
  acceptor_.bind(endpoint);
  acceptor_.listen();

  do_accept();
}

void Server::Run() { io_context_.run(); }

void Server::do_accept() {
  acceptor_.async_accept([this](boost::system::error_code ec,
                                boost::asio::ip::tcp::socket socket) {
    if (!acceptor_.is_open()) {
      return;
    }
    if (!ec) {
      connection_manager_.Start(std::make_shared<Connection>(
          std::move(socket), connection_manager_, request_handler_));
    }
    do_accept();
  });
}

void Server::do_await_stop() {
  signals_.async_wait([this](boost::system::error_code, int) {
    /// The server is stopped by cancelling all outstanding asynchronous
    /// operations. Once all operations have finished the io_context::run() call
    /// will exit.
    acceptor_.close();
    connection_manager_.StopAll();
  });
}

} // namespace server
} // namespace http