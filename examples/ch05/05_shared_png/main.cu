#include <cuda_runtime.h>
#include "utils/cuda_check.cuh"
#include "utils/image.hpp"

#define PI 3.14159265
#define THREADS 32

__global__ void render(unsigned char *ptr, const int height, const int width, const int channel)
{
    __shared__ unsigned char cache[THREADS][THREADS];
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (y < height && x < width) {
        const float period = 128.0f;
        cache[threadIdx.y][threadIdx.x] = 255 * (sinf(x*2.0f*PI/ period) + 1.0f) *
        (sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;
    } else {
        cache[threadIdx.y][threadIdx.x] = 0.0;
    }

    // __syncthreads();

    if (y < height && x < width) {
        int offset = (y * width + x) * channel;
        ptr[offset] = cache[THREADS - threadIdx.y][THREADS - threadIdx.x];
    }
}

int main()
{
    const int height  = 2000;
    const int width   = 2000;
    const int channel = 1;
    Image image(height, width, channel);

    unsigned char *dev_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dev_ptr, image.size()));

    dim3 threads(THREADS, THREADS);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y);
    render<<<blocks, threads>>>(dev_ptr, height, width, channel);

    CUDA_CHECK(cudaMemcpy(image.get_ptr(), dev_ptr, image.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_ptr));

    image.save("ch05_05_shared_png.png");
}
