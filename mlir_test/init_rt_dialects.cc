#include "init_rt_dialects.h"

#include "rt_base.h"

namespace rt {

void RegisterRTDialects(mlir::DialectRegistry& registry) {
  registry.insert<rt::RTDialect>();
}

} // namespace rt