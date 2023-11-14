#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>

using boost::asio::ip::udp;

int main(int argc, char** argv) {

  try {
  boost::asio::io_context io_context;
  udp::resolver resolver(io_context);

  boost::asio::ip::address add;
  add.from_string("127.0.0.1");
  udp::endpoint receiver_endpoint(add, 8013);

  udp::socket socket(io_context);
  socket.open(udp::v4());

  boost::array<char, 1> send_buf = {{0}};
  std::cout << "begin call send_to " << receiver_endpoint << std::endl;
  socket.send_to(boost::asio::buffer(send_buf), receiver_endpoint);

  boost::array<char, 128> recv_buf;
  udp::endpoint sender_endpoint;
  std::cout << "begin call receive_from "; 
  size_t len = socket.receive_from(boost::asio::buffer(recv_buf), sender_endpoint);
  std::cout << sender_endpoint << std::endl;
  std::cout.write(recv_buf.data(), len);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
