#ifndef RT_BASE
#define RT_BASE

include "mlir/IR/OpBase.td"

def RT_Dialect : Dialect {
  let name = "rt";
  let cppNamespace = "::rt";
  let description = [{
    ...
  }];
}

#endif // RT_BASE