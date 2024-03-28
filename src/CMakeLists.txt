cmake_minimum_required(VERSION 3.16)

set(HEADERS
    # cmake-format: sort
    functions/functions.h
    functions/node_acos.h
    functions/node_add.h
    functions/node_asin.h
    functions/node_atan.h
    functions/node_cos.h
    functions/node_div.h
    functions/node_mul.h
    functions/node_sin.h
    functions/node_sub.h
    functions/node_tan.h
    node.h
    node_var.h
    util.h
    var.h)

set(SOURCES
    # cmake-format: sort
    functions/functions.cpp
    functions/node_acos.cpp
    functions/node_add.cpp
    functions/node_asin.cpp
    functions/node_atan.cpp
    functions/node_cos.cpp
    functions/node_div.cpp
    functions/node_mul.cpp
    functions/node_sin.cpp
    functions/node_sub.cpp
    functions/node_tan.cpp
    node.cpp
    node_var.cpp
    util.cpp
    var.cpp)

add_library(cascade_static STATIC ${SOURCES})
set_target_properties(cascade_static PROPERTIES OUTPUT_NAME cascade_static)
target_include_directories(cascade_static
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/functions)

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
