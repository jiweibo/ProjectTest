#include <boost/python.hpp>
#include <iostream>
#include <string>

using namespace boost::python;

class HelloWorld {
public:
  HelloWorld(const std::string &name, int age);

  void printInfo();

private:
  std::string name_;
  int age_;
};

HelloWorld::HelloWorld(const std::string &name, int age)
    : name_(name), age_(age) {}

void HelloWorld::printInfo() {
  std::cout << "I am " << name_ << ", "
            << " my age is " << age_ << std::endl;
}

void test_func() { std::cout << "for test!" << std::endl; }

BOOST_PYTHON_MODULE(helloworld) {
  class_<HelloWorld, boost::noncopyable>("helloworld",
                                         init<const std::string &, int>())
      .def("printinfo", &HelloWorld::printInfo);

  def("test_func", &test_func);
}