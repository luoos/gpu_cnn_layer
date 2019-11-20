
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 32
#define BLOCK_SIZE 512

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float const_k[12000];

__global__ void unroll_x(float *x, float *x_unroll, int batch_idx, int H, int W, int K, int C) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int col_size = H_out * W_out;
    int input_map_idx = idx / col_size;
    int output_idx = idx % col_size; // start point of i-th K*K
    int input_row = output_idx / W_out;
    int input_col = output_idx % W_out;
    int x_row_base = input_map_idx*K*K*col_size;

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    int i = 0;
    if (idx < C * col_size) {
        for (int r = 0; r < K; r++) {
            for (int c = 0; c < K; c++) {
                x_unroll[x_row_base+i*col_size+output_idx] = x4d(batch_idx, input_map_idx, input_row+r, input_col+c);
                i++;
            }
    }
    }
#undef x4d
}

__global__ void forward_kernel( float *y, float *x, int batch_id,
                                const int B, const int M, const int C,
                                const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int row = blockIdx.y*blockDim.y + threadIdx.y;
    const int col = blockIdx.x*blockDim.x + threadIdx.x;
    const int output_size = H_out * W_out;
    const int inner_size = C * K * K;

    if ((row < M) && (col < output_size)) {
        float acc = 0;
        for (int k = 0; k < inner_size; k++) {
            acc += const_k[row*inner_size + k] * x[k*output_size + col];
        }
        y[batch_id * (M * output_size) + row * output_size + col] = acc;
    }
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // Set the kernel dimensions
    cudaMemcpyToSymbol(const_k, w.dptr_, sizeof(float) * M * C * K * K);
    float *device_X_unroll;
    const int X_unroll_size = C*K*K*H_out*W_out;
    cudaMalloc((void **)&device_X_unroll, sizeof(float)*X_unroll_size);

    // dimension for unroll kernel
    dim3 dimBlock_unroll(BLOCK_SIZE, 1, 1);
    // each thread loads K*K elements
    dim3 dimGrid_unroll(ceil(X_unroll_size/(1.0*BLOCK_SIZE*K*K)), 1, 1);

    // dimension for matrix multiplication kernel
    dim3 dimBlock_multi(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid_multi(ceil(H_out*W_out/(1.0*TILE_WIDTH)), ceil(M/(1.0*TILE_WIDTH)), 1);

    for (int bi = 0; bi < B; bi++) {
        unroll_x<<<dimGrid_unroll, dimBlock_unroll>>>(x.dptr_, device_X_unroll, bi, H, W, K, C);
        cudaDeviceSynchronize();
        forward_kernel<<<dimGrid_multi, dimBlock_multi>>>(y.dptr_, device_X_unroll, bi, B, M, C, H, W, K);
        cudaDeviceSynchronize();
    }


    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif