#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

int main(int argc, char** argv) {
	Tensor<float, 3> input(20, 30, 50);

  //array<ptrdiff_t, 3> S = {{1, 2, 0}};
  array<int, 3> S({1, 2, 0});

  Tensor<float, 3> output = input.shuffle(S);
  //std::cout << output << std::endl;
  std::cout << "num dim: " << output.NumDimensions << std::endl;
  std::cout << "tensor size: " << output.size() << std::endl;
  const Tensor<float, 3>::Dimensions& d = output.dimensions();
  std::cout << "Dim size: " << d.size()  << ", dim 0: " << d[0] << " dim 1: " << d[1] << " dim 2: " << d[2] << std::endl;
}
