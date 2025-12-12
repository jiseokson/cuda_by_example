#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "utils/cuda_check.cuh"

#define N 1024*2

#define BLOCKS  128
#define THREADS 128

__global__ void dot_prod(float *c, const float *a, const float *b)
{
    __shared__ float cache[THREADS];

    float temp = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += stride;
    }

    cache[threadIdx.x] = temp;
    __syncthreads();

    int i = THREADS / 2;
    while (i > 0) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
        c[blockIdx.x] = cache[0];
}

inline void print_row(
    const int i,
    const float *a,
    const float *b,
    const float *c,
    const int wi,
    const int wd,
    const int prec)
{
    printf("%*d %*.*f %*.*f %*.*f",
        wi, i,
        wd, prec, a[i],
        wd, prec, b[i],
        wd, prec, a[i] * b[i]);
    if (i % THREADS == 0 && (i / THREADS) < BLOCKS)
        printf(" %*.*f\n", wd, prec, c[i / THREADS]);
    else
        putchar('\n');
}

inline bool equal_sig_digits(float x, float y, int sig_digits)
{
    if (x == y)
        return true;

    double scale = std::max(std::fabs(x), std::fabs(y));
    if (scale == 0.0)
        return true;

    double tol = std::pow(10.0, -sig_digits);
    return std::fabs(x - y) / scale < tol;
}

int main()
{
    float host_a[N], host_b[N];
    for (int i = 0; i < N; ++i) {
        host_a[i] = i;
        host_b[i] = 2*i;
    }
    
    float *dev_c;
    float *dev_a, *dev_b;
    CUDA_CHECK(cudaMalloc((void**)&dev_c, sizeof(float)*BLOCKS));
    CUDA_CHECK(cudaMalloc((void**)&dev_a, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc((void**)&dev_b, sizeof(float)*N));

    CUDA_CHECK(cudaMemcpy(dev_a, host_a, sizeof(float)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, host_b, sizeof(float)*N, cudaMemcpyHostToDevice));

    dot_prod<<<BLOCKS, THREADS>>>(dev_c, dev_a, dev_b);

    float host_c[BLOCKS] = {0.};
    CUDA_CHECK(cudaMemcpy(host_c, dev_c, sizeof(float)*BLOCKS, cudaMemcpyDeviceToHost));

    float result = 0.0f;
    for (int i = 0; i < BLOCKS; ++i) {
        result += host_c[i];
    }

    CUDA_CHECK(cudaFree(dev_c));
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));

    const int previews = std::min(512, N);
    const int thread_previews = std::min(10, THREADS);
    const int wi = 6;
    const int wd = 15;
    const int prec = 2;

    printf("\033[1m[Configurations]\033[0m\n");
    printf("Number of elements : %d\n", N);
    printf("Number of blocks   : %d\n", BLOCKS);
    printf("Threads per block  : %d\n\n", THREADS);

    printf("Preview samples   : %4d\n", previews);
    printf("Samples per block : %4d\n\n", thread_previews);

    printf("\033[1m[Results]\033[0m\n");
    printf("%*s %*s %*s %*s %*s\n",
        wi, "Index",
        wd, "a",
        wd, "b",
        wd, "a*b",
        wd, "c");

    for (int i = 0; i < previews; ++i) {
        int j = 0;
        for (; j < thread_previews && i < previews; ++j, ++i) {
            print_row(i, host_a, host_b, host_c, wi, wd, prec);
        }

        // Skip to the position of the last thread in the block
        // For this expression to work correctly, the initial i must be > 0
        // That is, the previous for-loop must have executed at least once
        i = (i + THREADS - 1) / THREADS * THREADS - 1;

        // If the last thread in the block has not been printed yet,
        // print it here
        if (j < THREADS && i < previews) {
            printf("%*s\n", wi, "~");
            print_row(i, host_a, host_b, host_c, wi, wd, prec);
        }

        putchar('\n');
    }

    // If the last element among the entire dataset (N) has not been printed yet,
    // print it here
    if (previews < N) {
        printf("%*s\n\n", wi, "~~");
        print_row(N-1, host_a, host_b, host_c, wi, wd, prec);
        putchar('\n');
    }

    printf("\033[1m[Tests]\033[0m\n");
    double closed_form = (N-1.0)*N*(2.0*N-1.0) / 3.0;
    printf("Dot product of a and b (GPU)                 : %.2f\n", result);
    printf("Dot product of a and b (closed-form)         : %.2f\n", closed_form);

    const int sig_digits = 7;
    if (equal_sig_digits(result, closed_form, sig_digits))
        printf("Result verification (%d significant digits)   : ✅\n", sig_digits);
    else
        printf("Result verification (%d significant digits)   : ❌\n", sig_digits);
}
