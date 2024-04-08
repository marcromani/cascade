cmake_minimum_required(VERSION 3.16)

set(HEADERS # cmake-format: sort
            cascade.h functions/functions.h tensor.h var.h)

set(SOURCES # cmake-format: sort
            node.cpp node_var.cpp tensor.cpp util.cpp var.cpp)

file(GLOB FUNCTION_SOURCES functions/*.cpp)
list(APPEND SOURCES ${FUNCTION_SOURCES})

set(CUDA_SOURCES # cmake-format: sort
                 tensor.cu)
list(APPEND SOURCES ${CUDA_SOURCES})

find_package(CUDA REQUIRED)

add_library(cascade_static STATIC ${SOURCES})
set_target_properties(cascade_static PROPERTIES OUTPUT_NAME cascade_static)
target_include_directories(
  cascade_static
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/functions
         ${CUDA_INCLUDE_DIRS})
target_compile_definitions(cascade_static PRIVATE CUDA_ENABLED)

if(BUILD_SHARED)
  add_library(cascade_shared SHARED ${SOURCES})
  set_target_properties(cascade_shared PROPERTIES OUTPUT_NAME cascade_shared)
  target_include_directories(
    cascade_shared
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/functions
           ${CUDA_INCLUDE_DIRS})
  target_compile_definitions(cascade_shared PRIVATE CUDA_ENABLED)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CODE_COVERAGE)
  add_library(cascade_with_coverage STATIC ${SOURCES})
  set_target_properties(cascade_with_coverage PROPERTIES OUTPUT_NAME
                                                         cascade_with_coverage)
  target_include_directories(
    cascade_with_coverage
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/functions
           ${CUDA_INCLUDE_DIRS})
  target_compile_definitions(cascade_with_coverage PRIVATE CUDA_ENABLED)
  target_compile_options(cascade_with_coverage PRIVATE -fprofile-arcs
                                                       -ftest-coverage)
endif()

install(
  TARGETS cascade_static
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

if(TARGET cascade_shared)
  install(
    TARGETS cascade_shared
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
endif()

install(FILES ${HEADERS} DESTINATION include)
