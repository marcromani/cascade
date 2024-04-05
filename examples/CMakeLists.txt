cmake_minimum_required(VERSION 3.16)

add_executable(example_covariances example_covariances.cpp)
add_executable(example_derivatives example_derivatives.cpp)
add_executable(example_optimization example_optimization.cpp)

target_link_libraries(example_covariances cascade_static)
target_link_libraries(example_derivatives cascade_static)
target_link_libraries(example_optimization cascade_static)

install(TARGETS example_covariances example_derivatives example_optimization
        RUNTIME DESTINATION bin)
