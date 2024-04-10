
#include "cuda_runtime.h"
#include "device_launch_parameters.h"﻿
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

/*
    Outputs table comparing time for CPU and GPU for:
    - Quick Sort
    - Merge Sort
    - Bubble Sort
    - Bitonic Sort
*/

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: $s $s $d\n\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

void swap(int arr[], int pos1, int pos2) {
    int temp;
    temp = arr[pos1];
    arr[pos1] = arr[pos2];
    arr[pos2] = temp;
}

int partition(int arr[], int low, int high, int pivot) {
    int i = low;
    int j = low;
    while (i <= high) {
        if (arr[i] > pivot) {
            i++;
        }
        else {
            swap(arr, i, j);
            i++;
            j++;
        }
    }
    return j - 1;
}

__device__ void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int pos = partition(arr, low, high, pivot);

        quickSort(arr, low, pos - 1);
        quickSort(arr, pos + 1, high);
    }
}

__global__ void quickSort(int* data, const int n, int* sorted)
{
    quickSort << <1, 1>> > (data, 0, n - 1);
}

int main()
{
    // Initialize variables
    int* data_cpu;
    int* sorted_cpu;

    int* data_gpu;
    int* sorted_gpu;

    const int n = 100;

    // Initialize Grid, Blocks
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(10, 10, 1);

    // Allocate memory space in CPU
    data_cpu = (int*)malloc(n * sizeof(int));
    sorted_cpu = (int*)malloc(n * sizeof(int));

    // Allocate memory space in GPU
    GPUErrorAssertion(cudaMalloc((int**)&data_gpu, n * sizeof(int)));
    GPUErrorAssertion(cudaMalloc((int**)&sorted_gpu, n * sizeof(int)));

    // Initialize Array
    for (int i = 0; i < n; i++) {
        data_cpu[i] = rand() % 10;
    }

    // Copy data from CPU to GPU
    GPUErrorAssertion(cudaMemcpy(data_gpu, data_cpu, n * sizeof(int), cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(sorted_gpu, sorted_cpu, n * sizeof(int), cudaMemcpyHostToDevice));

    // CPU
    quickSort(data_cpu, 0, n - 1);

    // Start Clock
    auto start = std::chrono::steady_clock::now();

    // CPU
    quickSort(data_cpu, 0, n - 1);

    // Call kernel
    //quickSort << <gridSize, blockSize >> > (data_gpu, n, sorted_gpu);

    cudaDeviceSynchronize();

    // End Clock
    auto end = std::chrono::steady_clock::now();

    // Copy data from GPU to CPU
    GPUErrorAssertion(cudaMemcpy(data_cpu, data_gpu, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Print MAXMB Results
    for (int i = 0; i < n; i++) {
        cout << data_cpu[i] << "\t";
    }

    // Free memory space
    cudaFree(data_gpu);
    cudaFree(sorted_gpu);

    return 0;
}

