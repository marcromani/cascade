cmake_minimum_required(VERSION 3.16)

project(cascade)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

option(BUILD_SHARED "Build as shared library" ON)
option(CODE_COVERAGE "Build with code coverage flags, for GCC" ON)

if(MSVC)
  add_compile_options(/Wall /WX-)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  add_compile_options(-Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CODE_COVERAGE)
  add_compile_options(-Wall -Wextra -Wpedantic -Werror -fprofile-arcs
                      -ftest-coverage)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

add_subdirectory(src)
add_subdirectory(tests)
