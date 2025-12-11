#include <cstdio>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "utils/cuda_check.cuh"
#include "utils/image.hpp"

__global__ void render_frame(unsigned char *ptr, int tick, int height, int width)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < height && x < width) {
        float fy = x - width/2.0;
        float fx = height/2.0 - y;
        float  d = sqrtf(fx*fx + fy*fy);
        unsigned char grey = 128.0f + 127.0f * cos(d/10.0f - tick/7.0f) / (d/10.0f + 1.0f);

        int offset = y * width + x;
        ptr[offset] = grey;
    }
}

int main()
{
    const int height  = 1000;
    const int width   = 1000;
    const int channel = 1;
    Image image(height, width, channel);

    unsigned char *dev_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dev_ptr, image.size()));
    
    const int frames = 44;
    dim3 threads(32, 32);
    dim3 blocks(
        (width  + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y);

    const char *output_dir = "ch05_03_ripple";
    struct stat st = {0};
    if (stat(output_dir, &st) == -1) {
        mkdir(output_dir, 0755);
    }

    for (int tick = 0; tick < frames; ++tick) {
        render_frame<<<blocks, threads>>>(dev_ptr, tick, height, width);
        CUDA_CHECK(cudaMemcpy(image.get_ptr(), dev_ptr, image.size(), cudaMemcpyDeviceToHost));

        char filename[256] = {0,};
        sprintf(filename, "%s/frame%04d.png", output_dir, tick);
        image.save(filename);

        printf("Rendered frame %04d, saved to %s\n", tick, filename);
    }

    // You can generate a GIF from the PNG frames in the output directory using the following command:
    // ffmpeg -framerate 30 -i frame%04d.png -vf "scale=800:-1:flags=lanczos" -loop 0 output.gif

    CUDA_CHECK(cudaFree(dev_ptr));
}
