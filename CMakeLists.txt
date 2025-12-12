cmake_minimum_required(VERSION 3.18)
project(cuda_by_example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_library(utils STATIC
    src/utils/cuda_check.cu
    src/utils/image.cpp
)

target_include_directories(utils PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_subdirectory(examples/ch03/01_first_kernel)
add_subdirectory(examples/ch03/02_dev_info)
add_subdirectory(examples/ch03/03_choose_dev)
add_subdirectory(examples/ch04/01_add)
add_subdirectory(examples/ch04/02_cpu_julia_set)
add_subdirectory(examples/ch04/03_gpu_julia_set)
add_subdirectory(examples/ch05/01_add_threads)
add_subdirectory(examples/ch05/02_add_grid_stride)
add_subdirectory(examples/ch05/03_ripple)
add_subdirectory(examples/ch05/04_dot_prod)
add_subdirectory(examples/ch05/05_shared_png)
