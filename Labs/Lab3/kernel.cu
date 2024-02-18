#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);}

// ERROR Management
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: $s $s $d\n\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void matrixAdd(int* a, int* b, int* c, int N) {
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

__global__ void matrixMultiply(int* a, int* b, int* c, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += a[row * n + i] * b[i * p + col];
        }
        c[row * p + col] = sum;
    }
}

void matrixAddition(int* a, int* b, int* c, int N) {
    int* d_a, * d_b, * d_c;
    size_t size = N * sizeof(int);

    GPUErrorAssertion(cudaMalloc((void**)&d_a, size));
    GPUErrorAssertion(cudaMalloc((void**)&d_b, size));
    GPUErrorAssertion(cudaMalloc((void**)&d_c, size));

    GPUErrorAssertion(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    int threads4Block = 256;
    int blocksPerGrid = (N + threads4Block - 1) / threads4Block;

    matrixAdd <<<blocksPerGrid, threads4Block>>>(d_a, d_b, d_c, N);

    GPUErrorAssertion(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void matrixMultiplication(int* a, int* b, int* c, int m, int n, int p) {
    int* d_a, * d_b, * d_c;
    size_t size_a = m * n * sizeof(int);
    size_t size_b = n * p * sizeof(int);
    size_t size_c = m * p * sizeof(int);

    GPUErrorAssertion(cudaMalloc((void**)&d_a, size_a));
    GPUErrorAssertion(cudaMalloc((void**)&d_b, size_b));
    GPUErrorAssertion(cudaMalloc((void**)&d_c, size_c));

    GPUErrorAssertion(cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiply <<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, m, n, p);

    GPUErrorAssertion(cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    const int M = 300; // Rows of A
    const int N = 400; // Cols of A and rows of B
    const int P = 500; // Cols of B
    // Necesarry for multiplication same col A and rows B

    int* A = (int*)malloc(M * N * sizeof(int));
    int* B = (int*)malloc(N * P * sizeof(int));
    int* C_add = (int*)malloc(M * N * sizeof(int));
    int* C_mul = (int*)malloc(M * P * sizeof(int));

    // Initialize matrices
    for (int i = 0; i < M * N; ++i) {
        A[i] = i;
    }
    for (int i = 0; i < N * P; ++i) {
        B[i] = i;
    }

    // Time for matrix addition
    struct timeval start, end;
    gettimeofday(&start, NULL);
    matrixAddition(A, B, C_add, M * N);
    gettimeofday(&end, NULL);
    double add_time = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

    // Time for matrix multiplication
    gettimeofday(&start, NULL);
    matrixMultiplication(A, B, C_mul, M, N, P);
    gettimeofday(&end, NULL);
    double mul_time = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

    printf("Time for matrix addition: %.6f seconds\n", add_time);
    printf("Time for matrix multiplication: %.6f seconds\n", mul_time);

    // Free memory
    free(A);
    free(B);
    free(C_add);
    free(C_mul);

    return 0;
}