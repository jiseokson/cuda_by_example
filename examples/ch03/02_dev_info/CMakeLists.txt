add_executable(ch03_02 main.cu)

target_link_libraries(ch03_02 PRIVATE utils)

set_target_properties(ch03_02 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
