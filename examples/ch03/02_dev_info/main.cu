#include <cstdio>
#include <cuda_runtime.h>
#include "utils/cuda_check.cuh"

int main()
{
    cudaDeviceProp prop;

    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    
    for (int i = 0; i < count; ++i) {
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("================================\n");
        printf("    Information for device %d    \n", i);
        printf("================================\n");
        putchar('\n');

        printf("--- General Information ---\n");
        printf("Name                     : %s\n", prop.name);
        printf("Cupatation capability    : %d.%d\n", prop.major, prop.minor);
        printf("Clock rate               : %d\n", prop.clockRate);
        printf("Device copy overlap      : %s\n",
            prop.deviceOverlap ? "Enabled" : "Disabled");
        printf("Kernel execition timeout : %s\n",
            prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");
        putchar('\n');

        printf("--- Memory Information ---\n");
        printf("Total global mem  : %ld\n", prop.totalGlobalMem);
        printf("Total const mem   : %ld\n", prop.totalConstMem);
        printf("Memory pitch      : %ld\n", prop.memPitch);
        printf("Texture alignment : %ld\n", prop.textureAlignment);
        putchar('\n');

        printf("--- MP Information ---\n");
        printf("Multi processor count : %d\n", prop.multiProcessorCount);
        printf("Shared mem per block  : %ld\n", prop.sharedMemPerBlock);
        printf("Regs per block        : %d\n", prop.regsPerBlock);
        printf("Warp size             : %d\n", prop.warpSize);
        printf("Max threads per block : %d\n", prop.maxThreadsPerBlock);
        printf("Max block dim         : [%d, %d, %d]\n",
            prop.maxThreadsDim[0],
            prop.maxThreadsDim[1],
            prop.maxThreadsDim[2]);
        printf("Max grid dim          : [%d, %d, %d]\n",
            prop.maxGridSize[0],
            prop.maxGridSize[1],
            prop.maxGridSize[2]);

        if (i + 1 < count)
            putchar('\n');
    }
}
