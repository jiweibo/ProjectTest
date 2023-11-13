#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/thread/thread.hpp>

class Printer {
 public:
  Printer(boost::asio::io_context& io)
    : strand_(boost::asio::make_strand(io)),
      timer1_(io, boost::asio::chrono::seconds(1)),
      timer2_(io, boost::asio::chrono::seconds(1)),
      count_(0) {
    timer1_.async_wait(boost::asio::bind_executor(strand_, boost::bind(&Printer::Print1, this)));
    timer2_.async_wait(boost::asio::bind_executor(strand_, boost::bind(&Printer::Print2, this)));
  }
  ~Printer() {
    std::cout << "Final count is " << count_ << std::endl;
  }

  void Print1() {
    if (count_ < 10) {
      std::cout << "Timer1: " << count_  << " " << boost::this_thread::get_id() << std::endl;
      ++count_;
      timer1_.expires_at(timer1_.expiry() + boost::asio::chrono::seconds(1));
      timer1_.async_wait(boost::asio::bind_executor(strand_, boost::bind(&Printer::Print1, this)));
    }
  }

  void Print2() {
    if (count_ < 10) {
      std::cout << "Timer2: " << count_ << " " << boost::this_thread::get_id() << std::endl;
      ++count_;
      timer2_.expires_at(timer2_.expiry() + boost::asio::chrono::seconds(1));
      timer2_.async_wait(boost::asio::bind_executor(strand_, boost::bind(&Printer::Print2, this)));
    }
  }
 private:
  boost::asio::strand<boost::asio::io_context::executor_type> strand_;
  boost::asio::steady_timer timer1_;
  boost::asio::steady_timer timer2_;
  int count_;
};


int main() {
  boost::asio::io_context io;
  Printer p(io);
  boost::thread t(boost::bind(&boost::asio::io_context::run, &io));
  io.run();
  t.join();

  return 0;
}
