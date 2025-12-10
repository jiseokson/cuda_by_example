#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include "utils/cuda_check.cuh"

int main()
{
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    printf("ID of current CUDA device                 : %d\n", dev);
    
    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.major = 1;
    prop.minor = 3;

    CUDA_CHECK(cudaChooseDevice(&dev, &prop));
    printf("ID of CUDA device closest to revision 1.3 : %d\n", dev);
}