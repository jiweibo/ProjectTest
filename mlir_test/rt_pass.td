#ifndef RT_PASS
#define RT_PASS

include "rt_ops.td"

// rt.add.i32 -> rt.mul.i32
def TestAddPattern : Pat<(AddI32 $arg1,$arg2), (MulI32Op $arg1,$arg2)>;

#endif // RT_PASS