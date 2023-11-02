# Install onnxruntime

```
git clone https://github.com/microsoft/onnxruntime.git && cd onnxruntime
git checkout v1.11.1

# modify your trt home.
TENSORRT_HOME=...

./build.sh --config RelWithDebInfo --build_shared_lib --parallel --skip_tests --build_wheel  --use_cuda  --cuda_home /usr/local/cuda/ --cudnn_home /usr/ --use_tensorrt --tensorrt_home ${TENSORRT_HOME} --cmake_extra_defines=CMAKE_INSTALL_PREFIX=$PWD/build/install

make install
```


# Run demo
```
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOTDIR=${onnxruntime_ROOT}/build/install/ -Donnxruntime_USE_CUDA=ON -Donnxruntime_USE_TENSORRT=ON
wget https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.0-7.onnx
./ort_sample ./squeezenet1.0-7.onnx
```