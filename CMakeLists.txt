cmake_minimum_required(VERSION 3.16)

project(cascade)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

option(BUILD_SHARED "Build as shared library" ON)

if(MSVC)
  add_compile_options(/Wall /WX-)
else()
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_compile_options(-Wall -Wextra -Wpedantic)
  else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
  endif()
endif()

add_subdirectory(src)
add_subdirectory(tests)
