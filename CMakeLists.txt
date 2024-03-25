cmake_minimum_required(VERSION 3.16)

project(cascade)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

option(BUILD_SHARED "Build as shared library" ON)

if(MSVC)
  add_compile_options(/Wall)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(src)
add_subdirectory(tests)
