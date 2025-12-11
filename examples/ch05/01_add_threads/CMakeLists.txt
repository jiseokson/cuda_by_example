add_executable(ch05_01 main.cu)

target_link_libraries(ch05_01 PRIVATE utils)

set_target_properties(ch05_01 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
