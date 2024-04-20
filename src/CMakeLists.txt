cmake_minimum_required(VERSION 3.16)

add_subdirectory(var)
# add_subdirectory(tensor)

set(HEADERS cascade.h)
list(APPEND HEADERS ${VAR_HEADERS})

add_library(cascade_static STATIC $<TARGET_OBJECTS:var_object>)
set_target_properties(cascade_static PROPERTIES OUTPUT_NAME cascade_static)

if(BUILD_SHARED)
  add_library(cascade_shared SHARED $<TARGET_OBJECTS:var_object>)
  set_target_properties(cascade_shared PROPERTIES OUTPUT_NAME cascade_shared)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CODE_COVERAGE)
  add_library(cascade_with_coverage STATIC $<TARGET_OBJECTS:var_object>)
  set_target_properties(cascade_with_coverage PROPERTIES OUTPUT_NAME
                                                         cascade_with_coverage)
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
