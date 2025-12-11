#include "stb_image_write.h"

class Image
{
private:
    int height;
    int width;
    int channel;

    unsigned char *data;

public:
    Image(int height, int width, int channel)
        : height(height), width(width), channel(channel)
    {
        data = new unsigned char[height * width * channel];
    }

    ~Image()
    {
        delete[] data;
    }

    unsigned char *get_ptr(void)
    {
        return data;
    }

    int size(void)
    {
        return height * width * channel;
    }

    int save(const char *filename)
    {
        return stbi_write_png(
            filename, width, height, channel, data, width * channel);
    }
};
