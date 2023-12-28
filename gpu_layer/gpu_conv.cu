#include "../../src/layer/conv.h"

// Define the size of the shared memory. This should be chosen based on the architecture of your GPU.
#define TILE_WIDTH 16

__global__ void matMulKernel(float* A, float* B, float* C, int M, int N, int P) {
  // Allocate shared memory
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0;

  // Loop over the tiles of the input matrices
  for (int m = 0; m < (N - 1)/TILE_WIDTH + 1; ++m) {

    // Load one tile of A and B into shared memory
    if (row < M && m*TILE_WIDTH + threadIdx.x < N)
      As[threadIdx.y][threadIdx.x] = A[row*N + m*TILE_WIDTH + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    if (m*TILE_WIDTH + threadIdx.y < N && col < P)
      Bs[threadIdx.y][threadIdx.x] = B[(m*TILE_WIDTH + threadIdx.y)*P + col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    // Synchronize to make sure the tile is loaded
    __syncthreads();

    // Multiply the elements of the tile and accumulate the results
    for (int k = 0; k < TILE_WIDTH; ++k)
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

    // Synchronize to make sure the computation is done before loading the next tile
    __syncthreads();
  }

  // Write the computed value to the output matrix
  if (row < M && col < P)
    C[row * P + col] = sum;
}

__global__ void im2colKernel(const float* image, float* data_col, int height_in, int width_in, int height_out, int width_out, int height_kernel, int width_kernel, int stride, int pad_h, int pad_w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < height_out * width_out) {
    int step_h = i / width_out;
    int step_w = i % width_out;
    int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window

    for (int j = 0; j < height_kernel * width_kernel; j ++) {
      int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
      int cur_row = start_idx / width_in + j / width_kernel - pad_h;

      if (cur_col < 0 || cur_col >= width_in || cur_row < 0 || cur_row >= height_in) {
        data_col[i * height_kernel * width_kernel + j] = 0;
      }
      else {
        int pick_idx = cur_row * width_in + cur_col;
        data_col[i * height_kernel * width_kernel + j] = image[pick_idx];  // pick which pixel
      }
    }
  }
}

void Conv::im2col(const Vector& image, Matrix& data_col) {
  // Convert Eigen Vector to array
  const float* image_array = image.data();

  // Calculate dimensions
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;

  // Allocate GPU memory
  float* d_image;
  float* d_data_col;
  cudaMalloc(&d_image, hw_in * sizeof(float));
  cudaMalloc(&d_data_col, hw_out * hw_kernel * sizeof(float));

  // Copy data to GPU
  cudaMemcpy(d_image, image_array, hw_in * sizeof(float), cudaMemcpyHostToDevice);

  // Call GPU kernel
  int numThreads = 256;
  int numBlocks = (hw_out + numThreads - 1) / numThreads;
  im2colKernel<<<numBlocks, numThreads>>>(d_image, d_data_col, height_in, width_in, height_out, width_out, height_kernel, width_kernel, stride, pad_h, pad_w);
  cudaDeviceSynchronize();

  // Copy result back to CPU
  float* data_col_array = new float[hw_out * hw_kernel];
  cudaMemcpy(data_col_array, d_data_col, hw_out * hw_kernel * sizeof(float), cudaMemcpyDeviceToHost);

  // Convert result array back to Eigen matrix
  data_col = Eigen::Map<Matrix>(data_col_array, hw_out, hw_kernel);

  // Free GPU memory
  cudaFree(d_image);
  cudaFree(d_data_col);

  // Delete result array
  delete[] data_col_array;
}

void Conv::forward(const Matrix& bottom){
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