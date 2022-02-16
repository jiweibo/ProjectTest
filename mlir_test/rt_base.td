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

class RT_Type<string name, list<Trait> traits = [],
              string baseCppClass = "::mlir::Type">
      : TypeDef<RT_Dialect, name, traits, baseCppClass> {
}

def RT_PairType : RT_Type<"Pair"> {
  let summary = "rt::PairType";
  // Define a mnemonic to allow the dialect's parser hook to call into the
  // generated parser.
  let mnemonic = "pair";
  let parameters = (ins "int":$first, "int":$second);
}

#endif // RT_BASE