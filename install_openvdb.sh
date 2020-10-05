#git submodule update --init --recursive --depth 2

C:\\vcpkg\\vcpkg.exe install zlib:x64-windows
C:\\vcpkg\\vcpkg.exe install blosc:x64-windows
C:\\vcpkg\\vcpkg.exe install openexr:x64-windows
C:\\vcpkg\\vcpkg.exe install tbb:x64-windows
C:\\vcpkg\\vcpkg.exe install boost-iostreams:x64-windows
C:\\vcpkg\\vcpkg.exe install boost-system:x64-windows
C:\\vcpkg\\vcpkg.exe install boost-any:x64-windows
C:\\vcpkg\\vcpkg.exe install boost-algorithm:x64-windows
C:\\vcpkg\\vcpkg.exe install boost-uuid:x64-windows
C:\\vcpkg\\vcpkg.exe install boost-interprocess:x64-windows


cd third_party/openvdb/
rm -rf build
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=C:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -A x64 ..
cmake --build . --parallel 16 --config Release --target install
