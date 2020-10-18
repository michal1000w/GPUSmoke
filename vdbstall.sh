
#C:\\vcpkg\\vcpkg.exe install --triplet x64-windows ilmbase blosc boost zlib openexr tbb boost-python cppunit openvdb
#C:\\vcpkg\\vcpkg.exe integrate install

rm -rf third_party/openvdb/build
mkdir third_party/openvdb/build
cd third_party/openvdb/build

cmake -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake  -G"Visual Studio 16 2019" -DOPENVDB_ABI_VERSION_NUMBER=6 -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"  -DOPENVDB_BUILD_PYTHON_MODULE=ON -DOPENVDB_BUILD_UNITTESTS=ON -DOPENEXR_ROOT="C:/vcpkg/installed/x64-windows" -DILMBASE_ROOT=”C:/vcpkg/installed/x64-windows/lib” -DTBB_ROOT="C:/vcpkg/installed/x64-windows" -DUSE_BLOSC=OFF -DCPPUNIT_ROOT="C:/vcpkg/installed/x64-windows" -DBOOST_ROOT="C:/vcpkg/installed/x64-windows" -DBOOST-PYTHON_ROOT="C:/vcpkg/installed/x64-windows" -DBOOST_LIBRARYDIR="C:/vcpkg/installed/x64-windows" -DZLIB_ROOT="C:/vcpkg/installed/x64-windows"  -DVCPKG_TARGET_TRIPLET=x64-windows  ..

cmake --build . --parallel 4 --config Release --target install

#cmake -S"C:/vcpkg/openvdb" -G"Visual Studio 16 2019" -DOPENVDB_ABI_VERSION_NUMBER=5 -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"  -DOPENVDB_BUILD_PYTHON_MODULE=ON -DOPENVDB_BUILD_UNITTESTS=ON -DOPENEXR_ROOT="C:/vcpkg/installed/x64-windows" -DILMBASE_ROOT=”C:/vcpkg/installed/x64-windows/lib” -DTBB_ROOT="C:/vcpkg/installed/x64-windows" -DUSE_BLOSC=OFF -DCPPUNIT_ROOT="C:/vcpkg/installed/x64-windows" -DBOOST_ROOT="C:/vcpkg/installed/x64-windows" -DBOOST-PYTHON_ROOT="C:/vcpkg/installed/x64-windows" -DBOOST_LIBRARYDIR="C:/vcpkg/installed/x64-windows" -DZLIB_ROOT="C:/vcpkg/installed/x64-windows" ..

#cmake \ -G"Visual Studio 16 2019 Win64" \ -DCMAKE_CONFIGURATION_TYPES=%CONFIGURATION% \ -DCMAKE_INSTALL_PREFIX=%APPVEYOR_BUILD_FOLDER%\install \ -DCMAKE_TOOLCHAIN_FILE=\vcpkg\scripts\buildsystems\vcpkg.cmake \ -DOPENVDB_ABI_VERSION_NUMBER=6 \ -DOPENVDB_BUILD_BINARIES=OFF \-DOPENVDB_BUILD_PYTHON_MODULE=OFF \ -DOPENVDB_BUILD_UNITTESTS=ON \ -DOPENEXR_ROOT=C:\vcpkg\installed\x64-windows \ -DILMBASE_ROOT=C:\vcpkg\installed\x64-windows \ -DTBB_ROOT=C:\vcpkg\installed\x64-windows \ -DBLOSC_ROOT=C:\vcpkg\installed\x64-windows \ -DUSE_BLOSC=ON \ -DCPPUNIT_ROOT=C:\vcpkg\installed\x64-windows \ -DBOOST_ROOT=C:\Libraries\boost_1_67_0 \ -DBOOST_LIBRARYDIR=C:\Libraries\boost_1_67_0\lib64-msvc-14.0 \ ..