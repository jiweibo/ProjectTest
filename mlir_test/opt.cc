//===- Mlir-Opt utility ---------------------------------------------------===//
//
// Load MLIR and apply required passes on it.

#include "mlir/IR/Dialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MlirOptMain.h"
#include "init_rt_dialects.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  rt::RegisterRTDialects(registry);
  return mlir::failed(mlir::MlirOptMain(argc, argv, "RT pass driver\n", registry));
}