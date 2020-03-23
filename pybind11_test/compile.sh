g++ -O3 -Wall -shared -std=c++11 -fPIC -I${HOME}/anaconda3/include/python3.7m -L${HOME}/anaconda3/lib -lpython3.7m example.cpp -o python_example`python3-config --extension-suffix`
