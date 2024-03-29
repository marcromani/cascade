cmake_minimum_required(VERSION 3.16)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)

set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)

set(BUILD_GMOCK
    OFF
    CACHE BOOL "" FORCE)

set(INSTALL_GTEST
    OFF
    CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

set(SOURCES
    # cmake-format: sort
    covariance_tests.cpp derivative_tests.cpp matrix_tests.cpp var_tests.cpp)

add_executable(run_tests ${SOURCES})

enable_testing()

target_include_directories(run_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CODE_COVERAGE)
  target_link_libraries(run_tests cascade_static GTest::gtest_main -lgcov
                        --coverage)
else()
  target_link_libraries(run_tests cascade_static GTest::gtest_main)
endif()

include(GoogleTest)
gtest_discover_tests(run_tests)

install(TARGETS run_tests RUNTIME DESTINATION bin)
