// rtopt %s
func @main()-> f32 {
  %0 = "rt.constant.f32"() {value = 1.000000e+00 : f32} : () -> f32
  %1 = rt.constant.f32 2.0
  %2 = "rt.add.f32"(%0, %1) {} : (f32, f32) -> f32
  %3 = "rt.new.chain"() {} : () -> !rt.chain
  "rt.print.f32"(%2, %3) {} : (f32, !rt.chain) -> !rt.chain

  %4 = "rt.new.chains"() {num = 3:i64} : () -> !rt.vec<?x3x!rt.chain>
  rt.return %2 : f32
}
