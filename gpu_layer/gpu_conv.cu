#include "gpu_conv.h"


void Conv::forward_gpu(const Matrix& bottom){

}

__global__ void ConvKernel_v1(int * in, int width, int height, 
    float * kernel, int kernel_width, int * out)
{

    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int padding = kernel_width / 2;
    int i = r * width + c;

    if (r < width && c < height){
        for (int y = -padding; y <= padding; y++) {
            for (int x = -padding; x <= padding; x++) {
                int currentX = x + c;
                int currentY = y + r;

                currentX = (currentX < 0) ? 0 : currentX;
                currentX = (currentX > (width - 1)) ? (width - 1) : currentX;
                currentY = (currentY < 0) ? 0 : currentY;
                currentY = (currentY > (height - 1)) ? (height - 1) : currentY;
                        
                int filterIdx = (y + padding) * kernel_width + x + padding;
                int pixelIdx = currentY * width + currentX;

                out[i] += kernel[filterIdx] * in[pixelIdx];
            }
        }
    }
}