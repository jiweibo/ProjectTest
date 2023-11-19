#include "server.h"
#include <boost/asio.hpp>
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    // Check command line arguments
    if (argc != 4) {
      std::cerr << "Usage: http_server <adress> <port> <doc_root>\n";
      std::cerr << "  For IPv4, try: \n";
      std::cerr << "    receiver 0.0.0.0 8022. \n";
      std::cerr << "  For IPv6, try: \n";
      std::cerr << "    receiver 0::0 8022. \n";
      return 1;
    }

    // Initialize the he server.
    http::server::Server s(argv[1], argv[2], argv[3]);

    // Run the server until stopped.
    s.Run();
  } catch (std::exception& e) {
    std::cerr << "exception " << e.what() << "\n" << std::endl;
  }

  return 0;
}