
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int index = idz * blockDim.x * blockDim.y * gridDim.x * gridDim.y + idy * blockDim.x * gridDim.x + idx;

    if (index < N) {
        c[index] = a[index] + b[index];
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

    dim3 blockSize(8, 8, 8);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y, (N + blockSize.z - 1) / blockSize.z);

    vectorAdd << <gridSize, blockSize >> > (d_a, d_b, d_c, N);
    GPUErrorAssertion(cudaDeviceSynchronize());

    GPUErrorAssertion(cudaMemcpy(c, d_c, dataSize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; i++) {
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
