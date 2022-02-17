// rtopt %s
func @main()-> (f32, f32) {
  %0 = "rt.constant.f32"() {value = 1.000000e+00 : f32} : () -> f32
  %1 = rt.constant.f32 2.0
  rt.return %0, %1 : f32, f32
}
