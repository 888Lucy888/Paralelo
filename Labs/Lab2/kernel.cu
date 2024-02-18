#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define GPUErrorAssertion(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void vectorAdd(int* a, int* b, int* c, int N) {

    // Thread ID
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tidz = threadIdx.z;

    // Block ID
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bidz = blockIdx.z;

    // Block Dimensions
    int block_dimx = blockDim.x;
    int block_dimy = blockDim.y;
    int block_dimz = blockDim.z;

    // Grid Dim
    int gdimx = gridDim.x;
    int gdimy = gridDim.y;
    int gdimz = gridDim.z;

    // Row Offset
    int row_offset_x = gdimx * blockDim.x * bidx;
    int row_offset_y = gdimy * blockDim.y * bidy;
    int row_offset_z = gdimz * blockDim.z * bidz;

    // Block Offset 
    int offset_x = bidx * blockDim.x;
    int offset_y = bidy * blockDim.y;
    int offset_z = bidz * blockDim.z;

    // Grid ID
    int gidx = tidx + offset_x + row_offset_x;
    int gidy = tidy + offset_y + row_offset_y;
    int gidz = tidz + offset_z + row_offset_z;

    // Total threads per block
    int block_size = block_dimx * block_dimy * block_dimz;

    // Calculate global index
    int globalid = tidx + tidy * block_dimx + tidz * (block_dimx * block_dimy) +
        (bidx * gridDim.y * gridDim.z * block_size) +
        (bidy * gridDim.z * block_size) +
        (bidz * block_size);
                                                              
    if (globalid < N) {
        c[globalid] = a[globalid] + b[globalid];
    }
}

int main() {
    const int N = 10000;
    const int dataSize = N * sizeof(int);

    int* a, * b, * c;
    int* d_a, * d_b, * d_c;

    a = (int*)malloc(dataSize);
    b = (int*)malloc(dataSize);
    c = (int*)malloc(dataSize);

    GPUErrorAssertion(cudaMalloc((void**)&d_a, dataSize));
    GPUErrorAssertion(cudaMalloc((void**)&d_b, dataSize));
    GPUErrorAssertion(cudaMalloc((void**)&d_c, dataSize));

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    GPUErrorAssertion(cudaMemcpy(d_a, a, dataSize, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(d_b, b, dataSize, cudaMemcpyHostToDevice));

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    vectorAdd <<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    GPUErrorAssertion(cudaDeviceSynchronize());

    GPUErrorAssertion(cudaMemcpy(c, d_c, dataSize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 20; i++) {
        printf("c[%d] = %d\n", i, c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}
