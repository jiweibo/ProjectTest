//===- Mlir-Opt utility ---------------------------------------------------===//
//
// Load MLIR and apply required passes on it.

#include "init_rt_dialects.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  rt::RegisterRTDialects(registry);
  mlir::registerCanonicalizerPass();
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "RT pass driver\n", registry));
}