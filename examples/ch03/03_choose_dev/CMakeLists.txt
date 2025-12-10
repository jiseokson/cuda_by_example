add_executable(ch03_03 main.cu)

target_link_libraries(ch03_03 PRIVATE utils)

set_target_properties(ch03_03 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
