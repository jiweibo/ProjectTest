

```bash
cd src 
fastddsgen HelloWorld.idl
cd ..
mkdir build && cd build
cmake .. -Dfastcdr_ROOT=/wilber/repo/Fast-DDS/Fast-CDR/build/install \
  -Dfastdds_ROOT=/wilber/repo/Fast-DDS/Fast-DDS/build/install/ \
  -Dfoonathan_memory_vendor_ROOT=/wilber/repo/Fast-DDS/foonathan_memory_vendor/build/install
make -j4

./DDSHelloWorldPublisher &
./DDSHelloWorldSubscriber &
```