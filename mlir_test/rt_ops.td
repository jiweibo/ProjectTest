#ifndef RT_OPS
#define RT_OPS

include "rt_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class RT_Op<string mnemonic, list<Trait> traits = []>
    : Op<RT_Dialect, mnemonic, traits> {

  // Each registered op in the rt namespace needs to provide all of a printer,
  // parser and verifier.
  let parser = [{ return rt::parse$cppClass(parser, result); }];
  let printer = [{ return rt::print(p, *this); }];
  let verifier = [{ return rt::verify(*this); }];
}

def ReturnOp : RT_Op<"return", [Terminator]> {
  let summary = "rt.return operation";
  let description = [{
    The "rt.return" operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation. For example:

      func @foo() : (i32, f8) {
        ...
        rt.return %0, %1 : i32, f8
  }];
  let arguments = (ins Variadic<AnyType>:$operands);

  // let builders = [
  //   OpBuilder<(ins),
  //   [{ build($_builder, $_state, llvm::None); }]>];
}

class ConstantOp<string suffix, Type baseType, Attr attr>
  : RT_Op<"constant." # suffix, [NoSideEffect]> {
  let summary = "host executor constant value constructor";

  let arguments = (ins attr:$value);
  let results = (outs baseType);
  let assemblyFormat = "$value attr-dict";
  let verifier = ?;
}

def ConstantI1Op  : ConstantOp<"i1", I1, I1Attr>;
def ConstantI32Op : ConstantOp<"i32", I32, I32Attr>;
def ConstantI64Op : ConstantOp<"i64", I64, I64Attr>;
def ConstantF32Op : ConstantOp<"f32", F32, F32Attr>;
def ConstantF64Op : ConstantOp<"f64", F64, F64Attr>;
def ConstantUI32Op : ConstantOp<"ui32", UI32, UI32Attr>;
def ConstantUI64Op : ConstantOp<"ui64", UI64, UI64Attr>;

#endif // RT_OPS