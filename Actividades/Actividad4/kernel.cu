
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: $s $s $d\n\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void mat_mul(int* c, const int* a, const int* b, const int RA, const int CARB, const int CB)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < RA && col < CB) {
        int sum = 0;
        for (int i = 0; i < CARB; i++) {
            sum += a[row * CARB + i] * b[i * CARB + col];
        }
        c[row * CARB + col] = sum;
    }
}

int main()
{
    // Initialize variables (Matrix)
    int* MA_cpu;
    int* MB_cpu;
    int* MAXMB_cpu;

    int* MA_gpu;
    int* MB_gpu;
    int* MAXMB_gpu;

    const int RA = 10; // Rows of A
    const int CARB = 50; // Cols of A and rows of B
    const int CB = 20; // Cols of B

    const int data_size_MA = RA * CARB * sizeof(int);
    const int data_size_MB = CARB * CB * sizeof(int);
    const int data_size_MAXMB = RA * CB * sizeof(int);
  
    // Initialize Grid, Blocks
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(10, 10, 1);

    // Allocate memory space in CPU
    MA_cpu = (int*)malloc(data_size_MA);
    MB_cpu = (int*)malloc(data_size_MB);
    MAXMB_cpu = (int*)malloc(data_size_MAXMB);

    // Allocate memory space in GPU
    GPUErrorAssertion(cudaMalloc((int**)&MA_gpu, data_size_MA));
    GPUErrorAssertion(cudaMalloc((int**)&MB_gpu, data_size_MB));
    GPUErrorAssertion(cudaMalloc((int**)&MAXMB_gpu, data_size_MAXMB));

    // Initialize Matrix A and Matrix B
    for (int i = 0; i < RA * CARB; ++i) {
        MA_cpu[i] = rand() % 10;
    }
    for (int i = 0; i < CARB * CB; ++i) {
        MB_cpu[i] = rand() % 10;
    }

    // Copy data from CPU to GPU
    GPUErrorAssertion(cudaMemcpy(MA_gpu, MA_cpu, data_size_MA, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(MB_gpu, MB_cpu, data_size_MB, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(MAXMB_gpu, MAXMB_cpu, data_size_MAXMB, cudaMemcpyHostToDevice));

    // Call kernel
    mat_mul << <gridSize, blockSize >> > (MAXMB_gpu, MA_gpu, MB_gpu, RA, CARB, CB);

    // Copy data from GPU to CPU
    GPUErrorAssertion(cudaMemcpy(MAXMB_cpu, MAXMB_gpu, data_size_MAXMB, cudaMemcpyDeviceToHost));

    // Print MAXMB Results
    for (int i = 0; i < RA * CB; ++i) {
        printf("MAXMB[%d] = %d\n", i, MAXMB_gpu[i]);
    }

    // Free memory space
    cudaFree(MA_gpu);
    cudaFree(MB_gpu);
    cudaFree(MAXMB_gpu);
    
    return 0;
}
