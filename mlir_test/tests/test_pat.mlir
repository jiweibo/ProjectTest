// rtopt --canonicalize %s
func @func() -> i32 {
  %0 = rt.constant.i32 2
  %1 = rt.add.i32 %0, %0
  rt.return %1 : i32
}
