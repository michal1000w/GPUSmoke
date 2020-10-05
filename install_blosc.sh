git submodule update --init --recursive --depth 1

cd third_party/c-blosc/
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_GENERATOR_PLATFORM=x64
cmake --build . --config Release --parallel 16
cmake --build . --config Release --parallel 16 --target install
