#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 32

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void unroll_multiply( float *y, float *x, float *w,
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
    const int Y_width = H_out * W_out;
    const int inner_size = C * K * K;
    float acc = 0;

    for(int base = 0; base < ceil(inner_size/(1.0*TILE_WIDTH)); base++){
        int row_now = base * TILE_WIDTH + ty;
        int col_now = base * TILE_WIDTH + tx;
        int k_m = row;
        int k_c = col_now / (K * K);
        int k_h = col_now % (K * K) / K;
        int k_w = col_now % K;

        if (col_now < inner_size && row < M) {
            sharedW[ty][tx] = k4d(k_m, k_c, k_h, k_w);
        } else {
            sharedW[ty][tx] = 0;
        }

        int x_b = bz;
        int x_c = row_now / (K * K);
        int x_p = row_now % (K * K) / K;
        int x_q = row_now % K;
        int x_h = col / W_out;
        int x_w = col % W_out;

        if (row_now < inner_size && col < Y_width) {
            sharedX[ty][tx] = x4d(x_b, x_c, x_h + x_p, x_w + x_q);
        } else {
            sharedX[ty][tx] = 0;
        }

        __syncthreads();
        acc += (
            (sharedW[ty][ 0] * sharedX[ 0][tx]) +
            (sharedW[ty][ 1] * sharedX[ 1][tx]) +
            (sharedW[ty][ 2] * sharedX[ 2][tx]) +
            (sharedW[ty][ 3] * sharedX[ 3][tx]) +
            (sharedW[ty][ 4] * sharedX[ 4][tx]) +
            (sharedW[ty][ 5] * sharedX[ 5][tx]) +
            (sharedW[ty][ 6] * sharedX[ 6][tx]) +
            (sharedW[ty][ 7] * sharedX[ 7][tx]) +
            (sharedW[ty][ 8] * sharedX[ 8][tx]) +
            (sharedW[ty][ 9] * sharedX[ 9][tx]) +
            (sharedW[ty][10] * sharedX[10][tx]) +
            (sharedW[ty][11] * sharedX[11][tx]) +
            (sharedW[ty][12] * sharedX[12][tx]) +
            (sharedW[ty][13] * sharedX[13][tx]) +
            (sharedW[ty][14] * sharedX[14][tx]) +
            (sharedW[ty][15] * sharedX[15][tx]) +
            (sharedW[ty][16] * sharedX[16][tx]) +
            (sharedW[ty][17] * sharedX[17][tx]) +
            (sharedW[ty][18] * sharedX[18][tx]) +
            (sharedW[ty][19] * sharedX[19][tx]) +
            (sharedW[ty][20] * sharedX[20][tx]) +
            (sharedW[ty][21] * sharedX[21][tx]) +
            (sharedW[ty][22] * sharedX[22][tx]) +
            (sharedW[ty][23] * sharedX[23][tx]) +
            (sharedW[ty][24] * sharedX[24][tx]) +
            (sharedW[ty][25] * sharedX[25][tx]) +
            (sharedW[ty][26] * sharedX[26][tx]) +
            (sharedW[ty][27] * sharedX[27][tx]) +
            (sharedW[ty][28] * sharedX[28][tx]) +
            (sharedW[ty][29] * sharedX[29][tx]) +
            (sharedW[ty][30] * sharedX[30][tx]) +
            (sharedW[ty][31] * sharedX[31][tx]));
        __syncthreads();
    }

    int y_b = bz; int y_m = row; int y_h = col / W_out; int y_w = col % W_out;
    if (row < M && col < Y_width) {
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
    const int Y_width = H_out * W_out;
    // cudaMemcpyToSymbol(const_k, w.dptr_, sizeof(float) * M * C * K * K);

    dim3 dimBlock_multi(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid_multi(ceil(Y_width/(1.0*TILE_WIDTH)), ceil(M/(1.0*TILE_WIDTH)), B);

    unroll_multiply<<<dimGrid_multi, dimBlock_multi>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

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