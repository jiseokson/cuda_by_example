#include <cstdio>
#include <cuda_runtime.h>
#include "utils/image.hpp"

struct HostComplex
{
    float r, i;

    HostComplex(float r, float i) : r(r), i(i) {}

    float magnitude(void)
    {
        return r*r + i*i;
    }

    HostComplex operator+(const HostComplex& other)
    {
        return HostComplex(r + other.r, i + other.i);
    }

    HostComplex operator*(const HostComplex& other)
    {
        return HostComplex(r*other.r - i*other.i, i*other.r + r*other.i);
    }
};

int julia(int y, int x, int height, int width, float scale = 1.5)
{
    float jy = scale * (height/2.0 - y) / (height/2.0);
    float jx = scale * (x - width/2.0) / (width/2.0);

    HostComplex z(jx, jy);
    HostComplex c(-0.8, 0.156);

    for (int i = 0; i < 200; ++i) {
        z = z * z + c;
        if (z.magnitude() > 1000)
            return 0;
    }
    return 1;
}

void cpu_kernel(unsigned char *ptr, int height, int width, int channel)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int offset = (y * width + x) * channel;
            ptr[offset] = julia(y, x, height, width, 1.5) * 255;
        }
    }
}

int main()
{
    const int height  = 2000;
    const int width   = 2000;
    const int channel = 1;

    Image image(height, width, channel);
    unsigned char *ptr = image.get_ptr();

    cpu_kernel(ptr, height, width, channel);

    image.save("ch03_02_cpu_julia_set.png");
}
