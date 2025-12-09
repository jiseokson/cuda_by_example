cmake_minimum_required(VERSION 3.18)
project(cuda_by_example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_library(utils STATIC
    src/utils/cuda_check.cu
)

target_include_directories(utils PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_subdirectory(examples/ch03/01_first_kernel)
add_subdirectory(examples/ch03/02_dev_info)
