add_executable(ch05_03 main.cu)

target_link_libraries(ch05_03 PRIVATE utils)

set_target_properties(ch05_03 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
