cmake_minimum_required(VERSION 3.16)

set(HEADERS # cmake-format: sort
            functions/functions.h util.h var.h)

set(SOURCES # cmake-format: sort
            node.cpp node_var.cpp util.cpp var.cpp)

file(GLOB FUNCTION_SOURCES functions/*.cpp)
list(APPEND SOURCES ${FUNCTION_SOURCES})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
endif()

add_library(cascade_static STATIC ${SOURCES})
set_target_properties(cascade_static PROPERTIES OUTPUT_NAME cascade_static)
target_include_directories(
  cascade_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                        ${CMAKE_CURRENT_SOURCE_DIR}/functions)

if(BUILD_SHARED)
  add_library(cascade_shared SHARED ${SOURCES})
  set_target_properties(cascade_shared PROPERTIES OUTPUT_NAME cascade_shared)
  target_include_directories(
    cascade_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                          ${CMAKE_CURRENT_SOURCE_DIR}/functions)
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
