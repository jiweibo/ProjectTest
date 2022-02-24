#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "rt_ops.h.inc"

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "rt_pass.inc"
} // namespace

namespace rt {

/// Register our patterns as "canonicalization" patterns on the AddOp so
/// that they can be picked up by the Canonicalization framework.
void AddI32::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                         ::mlir::MLIRContext* context) {
  results.add<TestAddPattern>(context);
}

} // namespace rt