
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "toy/dialect.h"
#include "toy/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <memory>

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "toy/Pass.inc"
}  // namespace

class MyPatternRewriter : public mlir::PatternRewriter {
 public:
  MyPatternRewriter(mlir::MLIRContext* ctx) : mlir::PatternRewriter(ctx) {}
};

void applyMyPatternDriver(mlir::Operation* op, const mlir::FrozenRewritePatternSet& patterns) {
  llvm::outs() << "in applyMyPatternDriver " << op->getName()  << "\n";
  MyPatternRewriter rewriter(op->getContext());

  mlir::PatternApplicator applicator(patterns);
  applicator.applyCostModel([](const mlir::Pattern& pattern) {
    return pattern.getBenefit();
  });

  mlir::LogicalResult result = applicator.matchAndRewrite(op, rewriter);
  // if (mlir::failed(result))
  // return;
}


struct TestFusionPass : public mlir::PassWrapper<TestFusionPass, mlir::OperationPass<::mlir::toy::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFusionPass)
  llvm::StringRef getArgument() const final { return "test-fusion-pass"; }
  llvm::StringRef getDescription() const final { return "Test Fusion"; }

  void runOnOperation() override {
    std::cout << "in TestFusionPass::runOnOperation " << std::endl;
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    populateWithGenerated(patterns);



    llvm::MutableArrayRef<mlir::Region> regions = getOperation()->getRegions();
    mlir::FrozenRewritePatternSet frozen_patterns(std::move(patterns));

    for (auto& region : regions) {
      // Bottom-up
      region.walk([&](mlir::Operation* op) {
        addToWorklist(op);
      });

      // // Top-down
      //   region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      //       worklist.push_back(op);
      //       return mlir::WalkResult::advance();
      //   });
    }

    // std::reverse(worklist.begin(), worklist.end());
    // // Remember the reverse index.
    // for (size_t i = 0, e = worklist.size(); i != e; ++i)
    //   worklistMap[worklist[i]] = i;

    mlir::PatternApplicator pa(frozen_patterns);
    pa.applyDefaultCostModel();
    MyPatternRewriter rewriter(getOperation()->getContext());

    bool changed = false;
    while (!worklist.empty()) {
      auto* op = popFromWorklist();
      if (op == nullptr) continue;

      llvm::outs() << "Processing " << op->getName() << "\n";

      pa.matchAndRewrite(op, rewriter);

      getOperation().dump();
    }

    // mlir::PatternRewriter rr(context);
    // mlir::PatternApplicator ap;


    // applyMyPatternDriver(/*mlir::toy::AddOp()*/, mlir::FrozenRewritePatternSet(std::move(patterns)));
    
    // if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    //   return signalPassFailure();
  }

 protected:
  llvm::DenseMap<mlir::Operation*, unsigned> worklistMap;
  std::vector<mlir::Operation*> worklist;

  void addToWorklist(mlir::Operation* op) {
    if (worklistMap.count(op)) return;

    worklistMap[op] = worklist.size();
    worklist.push_back(op);
  }

  mlir::Operation* popFromWorklist() {
    auto* op = worklist.back();
    worklist.pop_back();

    if (op) worklistMap.erase(op);
    return op;
  }

};


void registerTestFusion() {
  mlir::PassRegistration<TestFusionPass>();
}

std::unique_ptr<mlir::Pass> CreateTestFusionPass() {
  return std::make_unique<TestFusionPass>();
}