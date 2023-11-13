cmake_minimum_required(VERSION 3.16)
project(asio_test CXX)

add_executable(timer1 Timer1.cc)
target_link_libraries(timer1 pthread)

add_executable(timer2 Timer2.cc)
target_link_libraries(timer2 pthread)

add_executable(timer3 Timer3.cc)
target_link_libraries(timer3 pthread)

add_executable(timer4 Timer4.cc)
target_link_libraries(timer4 pthread)

add_executable(timer5 Timer5.cc)
target_link_libraries(timer5 pthread boost_thread)
