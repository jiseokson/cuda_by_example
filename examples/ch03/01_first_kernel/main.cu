#include <cstdio>
#include <cuda_runtime.h>
#include "utils/cuda_check.cuh"

__global__ void add(int *c, int a, int b)
{
    *c = a + b;
}

int main()
{
    int c;
    int *dev_c;
    CUDA_CHECK(cudaMalloc((void**)&dev_c, sizeof(c)));

    int a = 2, b = 9;
    add<<<1, 1>>>(dev_c, a, b);

    CUDA_CHECK(cudaMemcpy(&c, dev_c, sizeof(c), cudaMemcpyDeviceToHost));

    printf("%d + %d = %d\n", a, b, c);
}