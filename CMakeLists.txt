cmake_minimum_required(VERSION 3.16)

project(
  cascade
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

option(BUILD_SHARED "Build as shared library" ON)
option(CODE_COVERAGE "Build with code coverage flags, for GCC" OFF)
option(ENABLE_CUDA "Enable GPU support, if available" ON)

find_package(CUDA)

if(CUDA_FOUND AND ENABLE_CUDA)
  enable_language(CUDA)
  set(CUDA_ENABLED 1)
else()
  set(CUDA_ENABLED 0)
endif()

if(MSVC)
  add_compile_options(/Wall /WX-)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  add_compile_options(-Wall -Wextra -Wpedantic)
elseif(CMAKE_CUDA_COMPILER)
  add_compile_options()
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
