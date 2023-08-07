#include "mlir/Pass/Pass.h"
#include <memory>

void registerTestFusion();

std::unique_ptr<mlir::Pass> CreateTestFusionPass();