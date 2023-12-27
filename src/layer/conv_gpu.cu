#include "conv_gpu.h"
#include "conv_gpu.cuh"

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

__global__ void GPUConvolutionKernel(uchar3 * inPixels, int width, int height, 
		float * filter, int filterWidth, 
		uchar3 * outPixels)
{
	// TODO
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;
	int padding = filterWidth / 2;
	int i = r * width + c;

	if (r < width && c < height){
		float red = 0.0f;
		float green = 0.0f;
		float blue = 0.0f;
		for (int y = -padding; y <= padding; y++) {
            for (int x = -padding; x <= padding; x++) {
				int currentX = x + c;
				int currentY = y + r;

				currentX = (currentX < 0) ? 0 : currentX;
				currentX = (currentX > (width - 1)) ? (width - 1) : currentX;
				currentY = (currentY < 0) ? 0 : currentY;
				currentY = (currentY > (height - 1)) ? (height - 1) : currentY;
						
				int filterIdx = (y + padding) * filterWidth + x + padding;
				int pixelIdx = currentY * width + currentX;

				red += filter[filterIdx] * inPixels[pixelIdx].x;
				green += filter[filterIdx] * inPixels[pixelIdx].y;
				blue += filter[filterIdx] * inPixels[pixelIdx].z;
        	}
        }
		outPixels[i].x = red;
		outPixels[i].y = green;
		outPixels[i].z = blue;
	}
}