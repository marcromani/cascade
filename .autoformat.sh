#!/bin/bash

clang-format -i src/*.cpp src/*.h src/functions/*.cpp src/functions/*.h
clang-format -i tests/*.cpp tests/include/*.h

cmake-format -c=.cmake-format CMakeLists.txt -o CMakeLists.txt
cmake-format -c=.cmake-format src/CMakeLists.txt -o src/CMakeLists.txt
cmake-format -c=.cmake-format tests/CMakeLists.txt -o tests/CMakeLists.txt
