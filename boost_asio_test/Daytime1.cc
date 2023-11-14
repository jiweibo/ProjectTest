#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>

using boost::asio::ip::tcp;

int main(int argc, char** argv) {
  try {
    //if (argc != 2) {
    //  std::cerr << "Usage: client <host>" << std::endl;
    //  return 1;
    //}
    boost::asio::io_context io_context;
    tcp::resolver resolver(io_context);
    
    tcp::resolver::query query("localhost", "8013");
    tcp::resolver::iterator iter = resolver.resolve(query);
    tcp::resolver::iterator end;
    boost::system::error_code error = boost::asio::error::host_not_found;
    tcp::endpoint endpoint;
    while (error && iter != end) {
      endpoint = *iter++;
      std::cout << endpoint << std::endl;
    }
    tcp::socket socket(io_context);
    //boost::asio::connect(socket, endpoint);
    socket.connect(endpoint, error);

    //while (iter != end) {
    //  endpoints = *iter++;
    //}
    //boost::asio::ip::address add;
    //add.from_string("127.0.0.1");
    //tcp::endpoint endpoint(add, 8013);
    //tcp::socket socket(io_context);
    //boost::system::error_code error = boost::asio::error::host_not_found;
    //socket.connect(endpoint, error);

    for (;;) {
      boost::array<char, 128> buf;
      boost::system::error_code error;
      size_t len = socket.read_some(boost::asio::buffer(buf), error);

      if (error == boost::asio::error::eof)
        break;
      else if (error)
        throw boost::system::system_error(error);
      
      std::cout.write(buf.data(), len);
    }
  } catch(std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
}
