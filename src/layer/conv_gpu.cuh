#ifndef CONV_GPU_CUH
#define CONV_GPU_CUH

// Declare your CUDA kernels here
__global__ void forward_kernel(/* parameters */);
__global__ void im2col_kernel(/* parameters */);

// Declare any device functions here
__device__ float some_device_function(/* parameters */);

#endif