#!/bin/bash

clang-format -i src/*.cpp src/*.h src/*.h.inl src/functions/*.cpp src/functions/*.h src/kernels/*.cu src/kernels/*.h
clang-format -i tests/*.cpp tests/include/*.h
clang-format -i examples/*.cpp

cmake-format -c=.cmake-format CMakeLists.txt -o CMakeLists.txt
cmake-format -c=.cmake-format src/CMakeLists.txt -o src/CMakeLists.txt
cmake-format -c=.cmake-format tests/CMakeLists.txt -o tests/CMakeLists.txt
cmake-format -c=.cmake-format examples/CMakeLists.txt -o examples/CMakeLists.txt
