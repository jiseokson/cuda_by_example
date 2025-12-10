add_executable(ch04_01 main.cu)

target_link_libraries(ch04_01 PRIVATE utils)

set_target_properties(ch04_01 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
