#!/bin/bash

clang-format -i src/*.h
clang-format -i src/var/*.cpp src/var/*.h src/var/functions/*.cpp src/var/functions/*.h
clang-format -i src/tensor/*.cpp src/tensor/*.h src/tensor/*.inl.h src/tensor/kernels/*.cu src/tensor/kernels/*.h
clang-format -i tests/*.cpp tests/include/*.h
clang-format -i examples/*.cpp

cmake-format -c=.cmake-format CMakeLists.txt -o CMakeLists.txt
cmake-format -c=.cmake-format src/CMakeLists.txt -o src/CMakeLists.txt
cmake-format -c=.cmake-format src/var/CMakeLists.txt -o src/var/CMakeLists.txt
cmake-format -c=.cmake-format src/tensor/CMakeLists.txt -o src/tensor/CMakeLists.txt
cmake-format -c=.cmake-format tests/CMakeLists.txt -o tests/CMakeLists.txt
cmake-format -c=.cmake-format examples/CMakeLists.txt -o examples/CMakeLists.txt
