#include "../../src/layer/conv.h"

void Conv::forward_gpu(const Matrix& bottom){
    int n_sample = bottom.cols();
    top.resize(height_out * width_out * channel_out, n_sample);
    data_cols.resize(n_sample);
    for (int i = 0; i < n_sample; i ++) {
        // im2col
        Matrix data_col;
        im2col(bottom.col(i), data_col);
        data_cols[i] = data_col;
        // conv by product

        //Matrix result = data_col * weight;  // result: (hw_out, channel_out)
        float* data_col_array = data_col.data();
        float* weight_array = weight.data();
        float* result_array = new float[data_col.rows() * weight.cols()];

        // Allocate GPU memory
        float* d_data_col;
        float* d_weight;
        float* d_result;
        cudaMalloc(&d_data_col, data_col.size() * sizeof(float));
        cudaMalloc(&d_weight, weight.size() * sizeof(float));
        cudaMalloc(&d_result, data_col.rows() * weight.cols() * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_data_col, data_col_array, data_col.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, weight_array, weight.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Call GPU kernel
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((weight.cols() + threadsPerBlock.x - 1) / threadsPerBlock.x, (data_col.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matMulKernel<<<numBlocks, threadsPerBlock>>>(d_data_col, d_weight, d_result, data_col.rows(), data_col.cols(), weight.cols());
        cudaDeviceSynchronize();

        // Copy result back to CPU
        cudaMemcpy(result_array, d_result, data_col.rows() * weight.cols() * sizeof(float), cudaMemcpyDeviceToHost);

        // Convert result array back to Eigen matrix
        Matrix result = Eigen::Map<Matrix>(result_array, data_col.rows(), weight.cols());

        result.rowwise() += bias.transpose();
        top.col(i) = Eigen::Map<Vector>(result.data(), result.size());

        // Free GPU memory
        cudaFree(d_data_col);
        cudaFree(d_weight);
        cudaFree(d_result);

        // Delete result array
        delete[] result_array;
    }
}

__global__ void ConvKernel_v1(int * in, int width, int height, 
    float * kernel, int kernel_width, int * out) {
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

__global__ void matMulKernel(float* A, float* B, float* C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (row < M && col < P) {
      float sum = 0;
      for (int i = 0; i < N; i++) {
        sum += A[row * N + i] * B[i * P + col];
      }
      C[row * P + col] = sum;
    }
  }