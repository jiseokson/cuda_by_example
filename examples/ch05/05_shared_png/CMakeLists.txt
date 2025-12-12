add_executable(ch05_05 main.cu)

target_link_libraries(ch05_05 PRIVATE utils)

set_target_properties(ch05_05 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
