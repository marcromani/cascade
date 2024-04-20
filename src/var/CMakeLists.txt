cmake_minimum_required(VERSION 3.16)

set(HEADERS # cmake-format: sort
            functions/functions.h var.h)

foreach(header ${HEADERS})
  list(APPEND PUBLIC_HEADERS var/${header})
endforeach()

set(PUBLIC_HEADERS
    ${PUBLIC_HEADERS}
    PARENT_SCOPE)

set(SOURCES # cmake-format: sort
            node.cpp node_var.cpp util.cpp var.cpp)

file(GLOB FUNCTION_SOURCES functions/*.cpp)
list(APPEND SOURCES ${FUNCTION_SOURCES})

add_library(var_object OBJECT ${SOURCES})
set_target_properties(var_object PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  var_object PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/functions)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CODE_COVERAGE)
  target_compile_options(var_object PRIVATE -fprofile-arcs -ftest-coverage)
endif()
