#include <iostream>
#include <vector>

// CUDA kernel for adding elements of two arrays
__global__ void add(int n, int *x, int *y) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}