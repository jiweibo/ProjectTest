// rtopt %s
func @func() {
  %0 = rt.constant.i32 2
  %1 = rt.add.i32 %0, %0
  rt.return
}
