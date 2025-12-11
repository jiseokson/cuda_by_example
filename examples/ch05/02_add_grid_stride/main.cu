#include <cstdio>
#include <cuda_runtime.h>
#include "utils/cuda_check.cuh"

#define N 500000

__global__ void add(float *c, const float *a, const float *b)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += stride;
    }
}

int main()
{
    float host_a[N], host_b[N];
    for (int i = 0; i < N; ++i) {
        host_a[i] = i + 1;
        host_b[i] = 2 * i;
    }
    
    float *dev_c;
    float *dev_a, *dev_b;
    CUDA_CHECK(cudaMalloc((void**)&dev_c, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc((void**)&dev_a, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc((void**)&dev_b, sizeof(float)*N));

    CUDA_CHECK(cudaMemcpy(dev_a, host_a, sizeof(float)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, host_b, sizeof(float)*N, cudaMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = 128;
    add<<<blocks, threads>>>(dev_c, dev_a, dev_b);

    float host_c[N];
    CUDA_CHECK(cudaMemcpy(host_c, dev_c, sizeof(float)*N, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dev_c));
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));

    const int previews = 15;
    const int wi = 7;
    const int wd = 12;
    const int prec = 2;
    printf("%*s %*s %*s %*s\n",
        wi, "Index",
        wd, "a",
        wd, "b",
        wd, "c");
    for (int i = 0; i < previews; ++i) {
        printf("%*d %*.*f %*.*f %*.*f\n",
            wi, i,
            wd, prec, host_a[i],
            wd, prec, host_b[i],
            wd, prec, host_c[i]);
    }

    printf("%*s\n", wi, "~");
    printf("%*d %*.*f %*.*f %*.*f\n",
        wi, N-1,
        wd, prec, host_a[N-1],
        wd, prec, host_b[N-1],
        wd, prec, host_c[N-1]);
}
