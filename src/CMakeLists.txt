cmake_minimum_required(VERSION 3.16)

set(HEADERS
    # cmake-format: sort
    include/math.h
    include/node.h
    include/node_add.h
    include/node_cos.h
    include/node_div.h
    include/node_mul.h
    include/node_sin.h
    include/node_sub.h
    include/node_var.h
    include/util.h
    include/var.h)

set(SOURCES
    # cmake-format: sort
    math.cpp
    node.cpp
    node_add.cpp
    node_cos.cpp
    node_div.cpp
    node_mul.cpp
    node_sin.cpp
    node_sub.cpp
    node_var.cpp
    util.cpp
    var.cpp)

if(BUILD_SHARED)
  add_library(cascade SHARED ${SOURCES})
else()
  add_library(cascade STATIC ${SOURCES})
endif()

set_target_properties(cascade PROPERTIES OUTPUT_NAME cascade)

target_include_directories(cascade PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(
  TARGETS cascade
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES ${HEADERS} DESTINATION include)
