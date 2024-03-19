cmake_minimum_required(VERSION 3.16)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)

set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(SOURCES # cmake-format: sort
            covariance-tests.cpp derivative-tests.cpp matrix-tests.cpp)

add_executable(run_tests ${SOURCES})

enable_testing()

target_link_libraries(run_tests cascade GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(run_tests)
