add_executable(ch05_04 main.cu)

target_link_libraries(ch05_04 PRIVATE utils)

set_target_properties(ch05_04 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
