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

def NewChainOp : RT_Op<"new.chain", [NoSideEffect]> {
  let summary = "host executor chain constructor";
  let description = [{
    The "rt.new.chain" operation returns a !rt.chain value, which can be fed
    into a side-effecting operation that does not depend on anything.
    For example:

      func @foo() {
        ...
        %x = rt.new.chain
        "side.effecting.thing"(%x) : (!rt.chain) -> ()
      }
  }];
  let arguments = (ins);
  let results = (outs RT_ChainType);
  let assemblyFormat = "attr-dict";
  let verifier = ?;
  // let parser = ?;
  // let printer = ?;
}

def NewVectorChainOp : RT_Op<"new.chains", [NoSideEffect]> {
  let summary = "";
  let description = [{

  }];
  let arguments = (ins I64Attr:$num);
  let results = (outs RT_VectorOf<RT_ChainType>:$chains);
  let parser = ?;
  let printer = ?;
  let verifier = ?;
}

class AddOp<string suffix, Type type>
  : RT_Op<"add." # suffix, [NoSideEffect]> {
  let summary = "rt.add operation";
  let description = [{
    An operation that takes two inputs and returns their sum as the result.

    Example:
      %2 = rt.add.i32 %0, %1
  }];
  let arguments = (ins type, type);
  let results = (outs type);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}
def AddI32 : AddOp<"i32", I32> {
  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;
}
def AddI64 : AddOp<"i64", I64>;
def AddF32 : AddOp<"f32", F32>;
def AddF64 : AddOp<"f64", F64>;

class MulOp<string suffix, Type type>
  : RT_Op<"mul." # suffix, [NoSideEffect]> {
  let summary = "rt.mul operation";
  let description = [{
    An operation that takes two inputs and returns their product.

    Example:
      %2 = rt.mul.i32 %0, %1
  }];
  let arguments = (ins type, type);
  let results = (outs type);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

def MulI32Op : MulOp<"i32", I32>;
def MulI64Op : MulOp<"i64", I64>;

class PrintOp<string suffix, Type type>
  : RT_Op<"print." # suffix> {
  let summary = "rt.print operation";
  let description = [{
    An operation takes a number input and a chain input.
    It prints the number to stdout and returns a chain output.
    The chain input must be the second operand.

    Example:
      %2 = rt.print.i3 %0, %1
  }];
  let arguments = (ins type, RT_ChainType);
  let results = (outs RT_ChainType);
  // let assemblyFormat = "operands attr-dict";
  let verifier = ?;
  let parser = ?;
  let printer = ?;
}
def PrintI1Op : PrintOp<"i1", I1>;
def PrintI32Op : PrintOp<"i32", I32>;
def PrintI64Op : PrintOp<"i64", I64>;
def PrintF32Op : PrintOp<"f32", F32>;
def PrintF64Op : PrintOp<"f64", F64>;

#endif // RT_OPS