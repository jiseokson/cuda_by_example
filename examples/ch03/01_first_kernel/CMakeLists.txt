add_executable(ch03_01 main.cu)

target_link_libraries(ch03_01 PRIVATE utils)

set_target_properties(ch03_01 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
