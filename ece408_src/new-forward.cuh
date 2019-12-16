#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 32
#define BLOCK_SIZE 512
#define B_SIZE 16

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void unroll_multiply( float *y, float *x, float *x_unroll, float *w, int batch_id,
                                const int B, const int M, const int C,
                                const int H, const int W, const int K)
{
    __shared__ float sharedW[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedX[TILE_WIDTH][TILE_WIDTH];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) w[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

    int bx = blockIdx.x;    int by = blockIdx.y;    int bz = blockIdx.z;
    int tx = threadIdx.x;   int ty = threadIdx.y;
    const int row = by*TILE_WIDTH + ty;
    const int col = bx*TILE_WIDTH + tx;
    const int X_col_size = H_out * W_out;
    const int inner_size = C * K * K;
    float acc = 0;

    //new implement for unroll fusion
    // int index_x = bx * TILE_WIDTH + tx;
    // int index_y = by * TILE_WIDTH + ty;
    // int h_base = index_x / W_out;
    // int w_base = index_x % W_out;
    // int h_unroll = h_base + index_y % (K * K) / K; 
    // int w_unroll = w_base + index_y % K;
    // int c = index_y / (K * K);
    int blvl = ceil(inner_size/(1.0*TILE_WIDTH));

    for(int base = 0; base < blvl; base ++){
        int row_now = base * TILE_WIDTH + ty; 
        int col_now = base * TILE_WIDTH + tx; 
        int k_m = row; int k_c = col_now / (K * K); int k_h = col_now % (K * K) / K; int k_w = col_now % K; 

        if (col_now < inner_size && row < M) {
            sharedW[ty][tx] = k4d(k_m, k_c, k_h, k_w);
        } else {
            sharedW[ty][tx] = 0;
        }

        int x_b = batch_id * B_SIZE + bz; int x_c = row_now / (K * K); int x_p = row_now % (K * K) / K; int x_q = row_now % K;    
        int x_h = col / W_out; int x_w = col % W_out;   
        
        if (row_now < inner_size && col < X_col_size && x_b < B) {
            sharedX[ty][tx] = x4d(x_b, x_c, x_h + x_p, x_w + x_q);
        } else {
            sharedX[ty][tx] = 0;
        }
        __syncthreads();

        for(int i = 0; i < TILE_WIDTH; i++){
            acc += sharedW[ty][i] * sharedX[i][tx];
        }
        __syncthreads();
    }

    int y_b = batch_id * B_SIZE + bz; int y_m = row; int y_h = col / W_out; int y_w = col % W_out;   
    if (row < M && col < X_col_size && y_b < B) {
        y4d(y_b, y_m, y_h, y_w) = acc;
    }
    #undef x4d
    #undef k4d
    #undef y4d
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
    float *device_X_unroll;
    const int X_unroll_size = C*K*K*H_out*W_out;
    cudaMalloc((void **)&device_X_unroll, sizeof(float)*X_unroll_size);

    // dimension for matrix multiplication kernel
    int bSize = ceil(B / (B_SIZE * 1.0));
    dim3 dimBlock_multi(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid_multi(ceil(H_out*W_out/(1.0*TILE_WIDTH)), ceil(C*K*K/(1.0*TILE_WIDTH)), B_SIZE);

    for (int bi = 0; bi < bSize; bi++) {
        unroll_multiply<<<dimGrid_multi, dimBlock_multi>>>(y.dptr_, x.dptr_, device_X_unroll, w.dptr_, bi, B, M, C, H, W, K);
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