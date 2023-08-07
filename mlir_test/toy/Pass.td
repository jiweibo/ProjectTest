#ifndef TOY_PASS
#define TOY_PASS

include "mlir/IR/PatternBase.td"
include "toy/Ops.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// Basic Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

// Matmul + Add -> Fc
def FcOptPattern : Pat<(AddOp(MatmulOp $x, $w) ,$b), (FcOp $x, $w, $b)>;

#endif // TOY_PASS