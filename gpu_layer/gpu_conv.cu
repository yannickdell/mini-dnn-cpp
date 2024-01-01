#include "../../src/layer/conv.h"

#define TILE_WIDTH 32

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}
// v1_use global memory
__global__ void matMulKernel_v1(float* A, float* B, float* C, int M, int N, int P) {
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

// v2_use shared memory
__global__ void matMulKernel_v2(float* A, float* B, float* C, int M, int N, int P) {
  // Allocate shared memory
  __shared__ float As[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0;

  // Loop over the tiles of the input matrices
  for (int m = 0; m < ((N - 1)/TILE_WIDTH + 1); ++m) {

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

// __global__ void im2colKernel(const float* image, float* data_col, int height_in, int width_in, int height_out, int width_out, int height_kernel, int width_kernel, int stride, int pad_h, int pad_w) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;

//   if (i < height_out * width_out) {
//     int step_h = i / width_out;
//     int step_w = i % width_out;
//     int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window

//     for (int j = 0; j < height_kernel * width_kernel; j ++) {
//       int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
//       int cur_row = start_idx / width_in + j / width_kernel - pad_h;

//       if (cur_col < 0 || cur_col >= width_in || cur_row < 0 || cur_row >= height_in) {
//         data_col[i * height_kernel * width_kernel + j] = 0;
//       }
//       else {
//         int pick_idx = cur_row * width_in + cur_col;
//         data_col[i * height_kernel * width_kernel + j] = image[pick_idx];  // pick which pixel
//       }
//     }
//   }
// }

// void Conv::im2col(const Vector& image, Matrix& data_col) {
//   // Convert Eigen Vector to array
//   const float* image_array = image.data();

//   // Calculate dimensions
//   int hw_in = height_in * width_in;
//   int hw_kernel = height_kernel * width_kernel;
//   int hw_out = height_out * width_out;

//   // Allocate GPU memory
//   float* d_image;
//   float* d_data_col;
//   cudaMalloc(&d_image, hw_in * sizeof(float));
//   cudaMalloc(&d_data_col, hw_out * hw_kernel * sizeof(float));

//   // Copy data to GPU
//   cudaMemcpy(d_image, image_array, hw_in * sizeof(float), cudaMemcpyHostToDevice);

//   // Call GPU kernel
//   int numThreads = 256;
//   int numBlocks = (hw_out + numThreads - 1) / numThreads;

//   im2colKernel<<<numBlocks, numThreads,smem_size>>>(d_image, d_data_col, height_in, width_in, height_out, width_out, height_kernel, width_kernel, stride, pad_h, pad_w);
//   cudaDeviceSynchronize();

//   // Copy result back to CPU
//   float* data_col_array = new float[hw_out * hw_kernel];
//   cudaMemcpy(data_col_array, d_data_col, hw_out * hw_kernel * sizeof(float), cudaMemcpyDeviceToHost);

//   // Convert result array back to Eigen matrix
//   data_col = Eigen::Map<Matrix>(data_col_array, hw_out, hw_kernel);

//   // Free GPU memory
//   cudaFree(d_image);
//   cudaFree(d_data_col);

//   // Delete result array
//   delete[] data_col_array;
// }

// v3_ use stream
void Conv::forward(const Matrix& bottom){
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);
  
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  for (int i = 0; i < n_sample; i ++) {
    // im2col
    Matrix data_col;
    im2col(bottom.col(i), data_col);
    data_cols[i] = data_col;
    size_t nByte_data_col = data_col.size() * sizeof(float);
    size_t nByte_weight = weight.size() * sizeof(float);
    size_t nByte_result = data_col.rows() * weight.cols() * sizeof(float);

    float* data_col_array = data_col.data();
    float* weight_array = weight.data();
    float* result_array = (float*)malloc(nByte_result);

    // Allocate GPU memory
    float* d_data_col;
    float* d_weight;
    float* d_result;
    
    CHECK(cudaMalloc(&d_data_col, nByte_data_col));
    CHECK(cudaMalloc(&d_weight, nByte_weight));
    CHECK(cudaMalloc(&d_result, nByte_result));

    // Copy data to GPU
    // CHECK(cudaMemcpyAsync(d_data_col, data_col_array, nByte_data_col, cudaMemcpyHostToDevice, stream1));
    // CHECK(cudaMemcpyAsync(d_weight, weight_array, nByte_weight, cudaMemcpyHostToDevice, stream2));

    // Call GPU kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((weight.cols() + threadsPerBlock.x - 1) / threadsPerBlock.x, (data_col.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y);
    int smem_size = (threadsPerBlock.x + TILE_WIDTH - 1)*(threadsPerBlock.y + TILE_WIDTH - 1) * sizeof(float);
    // matMulKernel_v2<<<numBlocks, threadsPerBlock, smem_size>>>(d_data_col, d_weight, d_result, data_col.rows(), data_col.cols(), weight.cols());
    // cudaDeviceSynchronize();

    // Process first part
    int len_data_first = data_col.size() / 2;
    int len_weight_first = weight.size() / 2;
    int len_result_first = (data_col.rows() * weight.cols()) / 2;
    CHECK(cudaMemcpyAsync(&d_data_col[0], &data_col_array[0], len_data_first * sizeof(float), cudaMemcpyHostToDevice, stream1));
    CHECK(cudaMemcpyAsync(&d_weight[0], &weight_array[0], len_weight_first * sizeof(float), cudaMemcpyHostToDevice, stream1));
    matMulKernel_v2<<<numBlocks, threadsPerBlock, smem_size, stream1>>>(&d_data_col[0], &d_weight[0], &d_result[0], 
        data_col.rows() / 2, data_col.cols() / 2, weight.cols() / 2);
    CHECK(cudaMemcpyAsync(&result_array[0], &d_result[0], len_result_first * sizeof(float), cudaMemcpyDeviceToHost, stream1));

    // Process second part
    int start_data_second = data_col.size() / 2;
    int len_data_second = data_col.size() - start_data_second;
    int start_weight_second = weight.size() / 2;
    int len_weight_second = weight.size() - start_weight_second;
    int start_result_second = (data_col.rows() * weight.cols()) / 2;
    int len_result_second = (data_col.rows() * weight.cols()) - start_result_second;
    CHECK(cudaMemcpyAsync(&d_data_col[start_data_second], &data_col_array[start_data_second], len_data_second * sizeof(float), cudaMemcpyHostToDevice, stream2));
    CHECK(cudaMemcpyAsync(&d_weight[start_weight_second], &weight_array[start_weight_second], len_weight_second * sizeof(float), cudaMemcpyHostToDevice, stream2));
    matMulKernel_v2<<<numBlocks, threadsPerBlock, smem_size, stream2>>>(&d_data_col[start_data_second], 
        &d_weight[start_weight_second], &d_result[start_result_second], 
        data_col.rows() - data_col.rows() / 2, data_col.cols() - data_col.cols() / 2, weight.cols() - weight.cols() / 2);
    CHECK(cudaMemcpyAsync(&result_array[start_result_second], &d_result[start_result_second], len_result_second * sizeof(float), cudaMemcpyDeviceToHost, stream2));
    
    cudaDeviceSynchronize();

    // Copy result back to CPU
    // CHECK(cudaMemcpy(result_array, d_result, result_size, cudaMemcpyDeviceToHost));

    // Convert result array back to Eigen matrix
    Matrix result = Eigen::Map<Matrix>(result_array, data_col.rows(), weight.cols());

    result.rowwise() += bias.transpose();
    top.col(i) = Eigen::Map<Vector>(result.data(), result.size());

    // Free GPU memory
    CHECK(cudaFree(d_data_col));
    CHECK(cudaFree(d_weight));
    CHECK(cudaFree(d_result));

    // Delete result arrays
    delete[] result_array;
  }
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
}

// use_stream v2
// void Conv::forward(const Matrix& bottom){
//   int n_sample = bottom.cols();
//   top.resize(height_out * width_out * channel_out, n_sample);
//   data_cols.resize(n_sample);
  
//   int nStreams = 8;
//   cudaStream_t stream[nStreams];
//   for (int i = 0; i < nStreams; i++) {
//       cudaStreamCreate(&stream[i]);
//   }

//   for (int i = 0; i < n_sample; i ++) {
//     // im2col
//     Matrix data_col;
//     im2col(bottom.col(i), data_col);
//     data_cols[i] = data_col;

//     float* data_col_array = data_col.data();
//     float* weight_array = weight.data();
//     float* result_array = new float[data_col.rows() * weight.cols()];

//     // Allocate GPU memory
//     float* d_data_col;
//     float* d_weight;
//     float* d_result;
//     size_t nByte_data_col = data_col.size() * sizeof(float);
//     size_t nByte_weight = weight.size() * sizeof(float);
//     size_t nByte_result = data_col.rows() * weight.cols() * sizeof(float);
//     CHECK(cudaMalloc(&d_data_col, nByte_data_col));
//     CHECK(cudaMalloc(&d_weight, nByte_weight));
//     CHECK(cudaMalloc(&d_result, nByte_result));

//     // Call GPU kernel
//     dim3 threadsPerBlock(32, 32);
//     dim3 numBlocks((weight.cols() + threadsPerBlock.x - 1) / threadsPerBlock.x, (data_col.rows() + threadsPerBlock.y - 1) / threadsPerBlock.y);
//     int smem_size = (threadsPerBlock.x + TILE_WIDTH - 1)*(threadsPerBlock.y + TILE_WIDTH - 1) * sizeof(float);

//     for (int j = 0; j < nStreams; j++){
//       // Start from different points in the data_col_array and weight_array for each stream
//       int start_data = j * data_col.size()/nStreams;
//       int start_weight = j * weight.size()/nStreams;
//       int start_result = j * (data_col.rows() * weight.cols())/nStreams;

//       // Calculate the length of data for each stream
//       int len_data = data_col.size()/nStreams;
//       int len_weight = weight.size()/nStreams;
//       int len_result = (data_col.rows() * weight.cols())/nStreams;

//       // If it's the last stream, adjust the length to include the remaining data
//       if (j == nStreams - 1) {
//           len_data = data_col.size() - start_data;
//           len_weight = weight.size() - start_weight;
//           len_result = (data_col.rows() * weight.cols()) - start_result;
//       }

//       // Copy data to GPU
//       CHECK(cudaMemcpyAsync(&d_data_col[start_data],&data_col_array[start_data],len_data * sizeof(float),cudaMemcpyHostToDevice, stream[j]));
//       CHECK(cudaMemcpyAsync(&d_weight[start_weight],&weight_array[start_weight],len_weight * sizeof(float),cudaMemcpyHostToDevice, stream[j]));

//       // Call GPU kernel
//       matMulKernel_v2<<<numBlocks,threadsPerBlock,smem_size,stream[j]>>>(&d_data_col[start_data], &d_weight[start_weight], &d_result[start_result], 
//       data_col.rows()/nStreams, data_col.cols()/nStreams, weight.cols()/nStreams);

//       // Copy result back to CPU
//       CHECK(cudaMemcpyAsync(&result_array[start_result],&d_result[start_result],len_result * sizeof(float),cudaMemcpyDeviceToHost, stream[j]));    
//     }

//     // Convert result array back to Eigen matrix
//     Matrix result = Eigen::Map<Matrix>(result_array, data_col.rows(), weight.cols());

//     result.rowwise() += bias.transpose();
//     top.col(i) = Eigen::Map<Vector>(result.data(), result.size());

//     // Free GPU memory
//     CHECK(cudaFree(d_data_col));
//     CHECK(cudaFree(d_weight));
//     CHECK(cudaFree(d_result));

//     // Delete result array
//     delete[] result_array;
//   }

//   for (int i = 0; i < nStreams; i++) {
//       CHECK(cudaStreamSynchronize(stream[i]));
//       cudaStreamDestroy(stream[i]);
//   }
// }