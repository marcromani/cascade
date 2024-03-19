cmake_minimum_required(VERSION 3.16)

project(cascade)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)

set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

option(BUILD_SHARED "Build as shared library" ON)

set(HEADERS
    # cmake-format: sort
    src/include/node-add.h
    src/include/node-div.h
    src/include/node-mul.h
    src/include/node-sub.h
    src/include/node-var.h
    src/include/node.h
    src/include/var.h)

set(SOURCES
    # cmake-format: sort
    src/node-add.cpp
    src/node-div.cpp
    src/node-mul.cpp
    src/node-sub.cpp
    src/node-var.cpp
    src/node.cpp
    src/var.cpp)

if(BUILD_SHARED)
  add_library(cascade SHARED ${SOURCES})
else()
  add_library(cascade STATIC ${SOURCES})
endif()

set_target_properties(cascade PROPERTIES OUTPUT_NAME cascade)

target_include_directories(cascade
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)

install(
  TARGETS cascade
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES ${HEADERS} DESTINATION include)

add_executable(main src/main.cpp)

target_link_libraries(main PRIVATE cascade)
