
RT MLIR and Runtime.

LLVM version: llvmorg-14.0.0-rc1

Build:
cmake .. -DLLVM_PATH=${LLVM_PATH}
make

Run opt:
./rtopt ../rt.mlir

Run exec:
TODO
