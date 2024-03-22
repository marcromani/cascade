#!/bin/bash

clang-format-11 -i src/*.cpp src/include/*.h
clang-format-11 -i tests/*.cpp tests/include/*.h

cmake-format -c=.cmake-format CMakeLists.txt -o CMakeLists.txt
cmake-format -c=.cmake-format src/CMakeLists.txt -o src/CMakeLists.txt
cmake-format -c=.cmake-format tests/CMakeLists.txt -o tests/CMakeLists.txt
