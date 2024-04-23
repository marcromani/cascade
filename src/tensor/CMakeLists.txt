cmake_minimum_required(VERSION 3.16)

set(HEADERS # cmake-format: sort
            proxy_value.h tensor.h tensor.h.inl tensor_data.h)

foreach(header ${HEADERS})
  list(APPEND PUBLIC_HEADERS tensor/${header})
endforeach()

set(PUBLIC_HEADERS
    ${PUBLIC_HEADERS}
    PARENT_SCOPE)

set(SOURCES # cmake-format: sort
            tensor.cpp)

set(CUDA_SOURCES # cmake-format: sort
                 kernels/kernel_add.cu kernels/kernel_mul.cu)
list(APPEND SOURCES ${CUDA_SOURCES})

add_library(tensor_object OBJECT ${SOURCES})
set_target_properties(tensor_object PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  tensor_object PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                       $<$<BOOL:${CUDA_ENABLED}>:${CUDA_INCLUDE_DIRS}>)
target_compile_definitions(tensor_object PUBLIC CUDA_ENABLED=${CUDA_ENABLED})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CODE_COVERAGE)
  add_library(tensor_object_with_coverage OBJECT ${SOURCES})
  set_target_properties(tensor_object_with_coverage
                        PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(
    tensor_object_with_coverage
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
           $<$<BOOL:${CUDA_ENABLED}>:${CUDA_INCLUDE_DIRS}>)
  target_compile_definitions(tensor_object_with_coverage
                             PUBLIC CUDA_ENABLED=${CUDA_ENABLED})
  target_compile_options(tensor_object_with_coverage PRIVATE -fprofile-arcs
                                                             -ftest-coverage)
endif()
