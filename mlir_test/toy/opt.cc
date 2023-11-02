//===- Mlir-Opt utility ---------------------------------------------------===//
//
// Load MLIR and apply required passes on it.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "toy/dialect.h"
#include "toy/Pass.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <system_error>

int LoadMLIR(llvm::SourceMgr& sourceMgr, mlir::MLIRContext& context, mlir::OwningOpRef<mlir::ModuleOp>& module) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN("../toy/test.mlir");
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << "../toy/test.mlir\n";
    return -1;
  }

  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Could not open input file: " << "../toy/test.mlir\n";
    return -1;
  }
  return 0;
}

int main(int argc, char** argv) {
  // mlir::DialectRegistry registry;
  // registry.insert<mlir::toy::ToyDialect>();
  // mlir::registerCanonicalizerPass();
  // registerTestFusion();
  // return mlir::failed(
  //     mlir::MlirOptMain(argc, argv, "RT pass driver\n", registry));

  mlir::MLIRContext context;
  // Load Dialect
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;

  if (int err = LoadMLIR(sourceMgr, context, module)) {
    return err;
  }
  llvm::outs() << "Src module:\n";
  module->dump();

  mlir::PassManager pm(&context);
  mlir::OpPassManager& nestFuncPM = pm.nest<mlir::toy::FuncOp>();
  nestFuncPM.addPass(CreateTestFusionPass());
  nestFuncPM.addPass(mlir::createCanonicalizerPass());

  if (mlir::failed(pm.run(*module))) {
    return 4;
  }
  llvm::outs() << "Dst module:\n";
  module->dump();
  return 0;
}