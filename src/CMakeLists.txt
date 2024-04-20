cmake_minimum_required(VERSION 3.16)

set(PUBLIC_HEADERS cascade.h)

add_subdirectory(tensor)
add_subdirectory(var)

get_target_property(TENSOR_INCLUDE_DIRS tensor_object
                    INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(VAR_INCLUDE_DIRS var_object INTERFACE_INCLUDE_DIRECTORIES)

add_library(cascade_static STATIC $<TARGET_OBJECTS:tensor_object>
                                  $<TARGET_OBJECTS:var_object>)
set_target_properties(cascade_static PROPERTIES OUTPUT_NAME cascade_static)
set_target_properties(cascade_static PROPERTIES PUBLIC_HEADER
                                                "${PUBLIC_HEADERS}")
target_include_directories(
  cascade_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${TENSOR_INCLUDE_DIRS}
                        ${VAR_INCLUDE_DIRS})

if(BUILD_SHARED)
  add_library(cascade_shared SHARED $<TARGET_OBJECTS:tensor_object>
                                    $<TARGET_OBJECTS:var_object>)
  set_target_properties(cascade_shared PROPERTIES OUTPUT_NAME cascade_shared)
  set_target_properties(cascade_shared PROPERTIES PUBLIC_HEADER
                                                  "${PUBLIC_HEADERS}")
  target_include_directories(cascade_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CODE_COVERAGE)
  add_library(cascade_with_coverage STATIC $<TARGET_OBJECTS:tensor_object>
                                           $<TARGET_OBJECTS:var_object>)
  set_target_properties(cascade_with_coverage PROPERTIES OUTPUT_NAME
                                                         cascade_with_coverage)
  set_target_properties(cascade_with_coverage PROPERTIES PUBLIC_HEADER
                                                         "${PUBLIC_HEADERS}")
  target_include_directories(cascade_with_coverage
                             PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(cascade_with_coverage PRIVATE -fprofile-arcs
                                                       -ftest-coverage)
endif()

install(
  TARGETS cascade_static
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include)

if(TARGET cascade_shared)
  install(
    TARGETS cascade_shared
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include)
endif()
