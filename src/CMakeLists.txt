cmake_minimum_required(VERSION 3.16)

set(HEADERS
    # cmake-format: sort
    include/functions.h
    include/node.h
    include/node_add.h
    include/node_cos.h
    include/node_div.h
    include/node_mul.h
    include/node_sin.h
    include/node_sub.h
    include/node_tan.h
    include/node_var.h
    include/util.h
    include/var.h)

set(SOURCES
    # cmake-format: sort
    functions.cpp
    node.cpp
    node_add.cpp
    node_cos.cpp
    node_div.cpp
    node_mul.cpp
    node_sin.cpp
    node_sub.cpp
    node_tan.cpp
    node_var.cpp
    util.cpp
    var.cpp)

add_library(cascade_static STATIC ${SOURCES})
set_target_properties(cascade_static PROPERTIES OUTPUT_NAME cascade_static)
target_include_directories(cascade_static
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(BUILD_SHARED)
  add_library(cascade_shared SHARED ${SOURCES})
  set_target_properties(cascade_shared PROPERTIES OUTPUT_NAME cascade_shared)
  target_include_directories(cascade_shared
                             PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
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
