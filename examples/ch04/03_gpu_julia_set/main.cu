#include <cstdio>
#include <cuda_runtime.h>
#include "utils/cuda_check.cuh"
#include "utils/image.hpp"

struct DevComplex
{
    float r, i;

    __device__ DevComplex(float r, float i) : r(r), i(i) {}

    __device__ float magnitude(void)
    {
        return r*r + i*i;
    }

    __device__ DevComplex operator+(const DevComplex& other)
    {
        return DevComplex(r + other.r, i + other.i);
    }

    __device__ DevComplex operator*(const DevComplex& other)
    {
        return DevComplex(r*other.r - i*other.i, i*other.r + r*other.i);
    }
};

__device__ int julia(int y, int x, int height, int width, float scale = 1.5)
{
    float jy = scale * (height/2.0 - y) / (height/2.0);
    float jx = scale * (x - width/2.0) / (width/2.0);

    DevComplex z(jx, jy);
    DevComplex c(-0.8, 0.156);

    for (int i = 0; i < 200; ++i) {
        z = z * z + c;
        if (z.magnitude() > 1000)
            return 0;
    }
    return 1;
}

__global__ void gpu_kernel(unsigned char *ptr, int channel)
{
    int y = blockIdx.y;
    int x = blockIdx.x;
    int height = gridDim.y;
    int width = gridDim.x;

    int julia_value = julia(y, x, height, width, 1.5);
    
    int offset = (y * gridDim.x + x) * channel;
    ptr[offset] = julia_value * 255;
}

int main()
{
    const int height  = 2000;
    const int width   = 2000;
    const int channel = 1;
    Image image(height, width, channel);

    unsigned char *dev_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dev_ptr, image.size()));

    dim3 blocks(width, height);
    gpu_kernel<<<blocks, 1>>>(dev_ptr, channel);

    CUDA_CHECK(cudaMemcpy(image.get_ptr(), dev_ptr, image.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_ptr));
    
    image.save("ch03_03_gpu_julia_set.png");
}
