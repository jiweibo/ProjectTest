module {
  toy.func @main() -> tensor<2x2xf64>{

    // %0 = toy.constant dense<1.0> : tensor<2x3xf64>
    // %1 = toy.constant dense<2.0> : tensor<2x3xf64>
    // %2 = toy.add %0, %1 : tensor<2x3xf64>

    %0 = toy.constant dense<1.0> : tensor<2x3xf64>
    %2 = toy.constant dense<2.0> : tensor<2x3xf64>
    %w = toy.constant dense<3.0> : tensor<3x2xf64>
    %b = toy.constant dense<1.0> : tensor<2x2xf64>
    %4 = "toy.matmul"(%2, %w) {} : (tensor<2x3xf64>, tensor<3x2xf64>) -> tensor<2x2xf64>
    %5 = "toy.add"(%4, %b) {} : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>

    %6 = "toy.matmul"(%2, %w) {} : (tensor<2x3xf64>, tensor<3x2xf64>) -> tensor<2x2xf64>
    %7 = "toy.add"(%6, %5) {} : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>


    toy.return %7 : tensor<2x2xf64>
  }
}

