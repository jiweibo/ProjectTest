#include <iostream>
#include <vector>
#include <cblas.h>

int main()
{
  const int M = 3;
  const int N = 4;
  const int K = 2;
  std::vector<float> x(M * K, 0);
  std::vector<float> y(K * N, 0);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = i % 10 * 0.1;
  }
  for (size_t i = 0; i < y.size(); ++i) {
    y[i] = i % 10 * 0.2;
  }
  std::vector<float> out(M * N, 0);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, x.data(), K,
                y.data(), N, 0.0, out.data(), N);
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      std::cout << out[i * N + j] << "\t";
    }
    std::cout << std::endl;
  }
  return 0;
}
