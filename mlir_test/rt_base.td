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

def RT_ChainType :
  Type<CPred<"$_self.isa<rt::ChainType>()">, "!rt.chain type">,
  BuildableType<"$_builder.getType<rt::ChainType>()">;

#endif // RT_BASE