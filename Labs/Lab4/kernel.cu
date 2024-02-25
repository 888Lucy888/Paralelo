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

__global__ void convolution2D(int *matr, int *res, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width && row < height) {
        int g_index = row * width + col;
        float sum = 0.0f;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int curRow = row + i;
                int curCol = col + j;
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    sum += matr[curRow * width + curCol];
                }
            }
        }
        res[g_index] = sum;
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const int dataSize = width * height * sizeof(int);

    int *MAT, *RES;
    int *MAT_gpu, *RES_gpu;

    MAT = (int*)malloc(dataSize);
    RES = (int*)malloc(dataSize);

    GPUErrorAssertion(cudaMalloc((void**)&MAT_gpu, dataSize));
    GPUErrorAssertion(cudaMalloc((void**)&RES_gpu, dataSize));

    for (int i = 0; i < width * height; ++i) {
        MAT[i] = rand() % 9;
    }

    printf("VALUES BEFORE: \n");
    for (int i = 0; i < 15; ++i) {
        printf("MAT[%d] = %d\n", i, MAT[i]);

    }

    GPUErrorAssertion(cudaMemcpy(MAT_gpu, MAT, dataSize, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    convolution2D<<<gridSize, blockSize>>>(MAT_gpu, RES_gpu, width, height);
    GPUErrorAssertion(cudaDeviceSynchronize());

    GPUErrorAssertion(cudaMemcpy(RES, RES_gpu, dataSize, cudaMemcpyDeviceToHost));

    printf("VALUES AFTER: \n");
    for (int i = 0; i < 15; ++i) {
        printf("RES[%d] = %d\n", i, RES[i]);
    }

    cudaFree(MAT_gpu);
    cudaFree(RES_gpu);
    free(MAT);
    free(RES);

    return 0;
}
