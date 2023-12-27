#include "conv_gpu.h"
#include "conv_gpu.cuh"
#include <stdio.h>
#include <stdint.h>

// Declare and define your CUDA kernels here
__global__ void forward_kernel(/* parameters */) {
    // kernel code
}

__global__ void im2col_kernel(/* parameters */) {
    // kernel code
}

// Define your class functions here
void Conv::forward(const Matrix& bottom) {
    // Allocate GPU memory
    // Copy data to GPU
    // Launch forward_kernel
    // Copy results back to CPU
    // Free GPU memory
}

void Conv::im2col(const Vector& image, Matrix& data_col) {
    // Allocate GPU memory
    // Copy data to GPU
    // Launch im2col_kernel
    // Copy results back to CPU
    // Free GPU memory
}

__global__ void GPUConvolutionKernel(int* inPixels, int* outPixels, int height_in, int width_in,
	int height_kernel, int width_kernel, float* weight_kernel)
{
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;
	int padding = width_kernel / 2;
	int i = r * width_in + c;

	if (r < width_in && c < height_in){
		for (int y = -padding; y <= padding; y++) {
            for (int x = -padding; x <= padding; x++) {
				int currentX = x + c;
				int currentY = y + r;

				currentX = (currentX < 0) ? 0 : currentX;
				currentX = (currentX > (width_in - 1)) ? (width_in - 1) : currentX;
				currentY = (currentY < 0) ? 0 : currentY;
				currentY = (currentY > (height_in - 1)) ? (height_in - 1) : currentY;
						
				int filterIdx = (y + padding) * width_kernel + x + padding;
				int pixelIdx = currentY * width_in + currentX;

				outPixels[i] += weight_kernel[filterIdx] * inPixels[pixelIdx];
        	}
        }
	}
}