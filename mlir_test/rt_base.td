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
    DialectType<RT_Dialect, CPred<"$_self.isa<rt::ChainType>()">, "!rt.chain type">,
    BuildableType<"$_builder.getType<rt::ChainType>()">;

// class RT_Type<string name, list<Trait> traits = [],
//              string baseCppClass = "::mlir::Type">
//      : TypeDef<RT_Dialect, name, traits, baseCppClass> {
// }

// def RT_PairType : RT_Type<"Pair"> {
//   let summary = "rt::PairType";
//   // Define a mnemonic to allow the dialect's parser hook to call into the
//   // generated parser.
//   let mnemonic = "pair";
//   let parameters = (ins "int":$first, "int":$second);
// }

// Type constraint accepting any RT vector type.
def RT_AnyVector : Type<CPred<"::rt::isCompatibleVectorType($_self)">, "RT dialect-compatible vector type">;

class RT_VectorOf<Type element> : Type<
  And<[RT_AnyVector.predicate,
       SubstLeaves<
       "$_self",
       "::rt::getVectorElementType($_self)",
       element.predicate>]>,
  "RT dialect-compatible vector of " # element.summary>;

// Type constraint accepting a constrained type, or a vector of such types.
class RT_ScalarOrVectorOf<Type element> : AnyTypeOf<[element, RT_VectorOf<element>]>;

#endif // RT_BASE