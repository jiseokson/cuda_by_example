add_executable(ch04_02 main.cu)

target_link_libraries(ch04_02 PRIVATE utils)

set_target_properties(ch04_02 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
