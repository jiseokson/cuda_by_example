add_executable(ch05_02 main.cu)

target_link_libraries(ch05_02 PRIVATE utils)

set_target_properties(ch05_02 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
