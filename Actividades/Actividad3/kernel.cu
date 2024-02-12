#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>

using namespace std;

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__);}

// MANEJO DE ERRORES EN TIEMPO DE EJECUCIÓN (encontrar error, detener programa) / (si cada funcion de cuda se ejecuta correctamente)
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: $s $s $d\n\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void queryDevice() {
    int d_Count = 0;

    cudaGetDeviceCount(&d_Count);

    if (d_Count == 0) {
        printf("No CUDA device found:\n\r");
    }

    cudaDeviceProp(prop);

    for (int devNo = 0; devNo < d_Count; devNo++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, devNo);
        printf("Device Number: %d\n", devNo);
        printf(" Device Name: %s\n", prop.name);
        printf(" Number of Multiprocessors:         %d\n", prop.multiProcessorCount);
        printf(" Compute Capability (version):      %d,%d\n", prop.major, prop.minor);
        printf(" Memory Clock Rate (KHz):           %d\n", prop.memoryClockRate);
        printf(" Memory Bus Rate (bits):            %d\n", prop.memoryBusWidth);
        printf(" Peak Memory Bandwith (GB/s):       %8.2f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf(" Total Amount of Global Memory:     %dKB\n", prop.totalGlobalMem / 1024);
        printf(" Total Amount of Const Memory:      %dKB\n", prop.totalConstMem / 1024);
        printf(" Total of Shared Memory per block:  %dKB\n", prop.sharedMemPerBlock / 1024);
        printf(" Total of Shared Memory per MP:     %dKB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf(" Warp Size:                         %d\n", prop.warpSize);
        printf(" Max. threads per block:            %d\n", prop.maxThreadsPerBlock);
        printf(" Max. threads per MP:               %d\n", prop.maxThreadsPerMultiProcessor);
        printf(" Maximum number of warps per MP:    %d\n", prop.maxThreadsPerMultiProcessor / 32);
        printf(" Maximum Grid size:                (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf(" Maximum Block dimension:          (%d,%d,%d)\n\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    }
}

__global__ void print_all_idx()
{
    // Thread ID
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tidz = threadIdx.z;

    // Block ID
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bidz = blockIdx.z;

    // Block ID
    int gdimx = gridDim.x;
    int gdimy = gridDim.y;
    int gdimz = gridDim.z;

    // Printing in kernel bad (only to see here)
    printf("[DEVICE] threadIdx.x: %d, blockIdx.x %d, gridDim.x: %d \n", tidx, bidx, gdimx);
    printf("[DEVICE] threadIdx.y: %d, blockIdx.y %d, gridDim.y: %d \n", tidy, bidy, gdimy);
    printf("[DEVICE] threadIdx.z: %d, blockIdx.z %d, gridDim.z: %d \n", tidz, bidz, gdimz);
}

int main()
{
    queryDevice();

    // 1 --------------------- Inicializar Datos -------------------------------------------------------
    dim3 blockSize(4, 4, 4);
    dim3 gridSize(2, 2, 2);

    int* c_cpu;
    int* a_host;
    int* b_cpu;

    int* c_gpu;
    int* a_device;
    int* b_gpu;

    const int data_count = 100000;
    const int data_size = data_count * sizeof(int);

    // Reservar en memoria en CPU
    c_cpu = (int*)malloc(data_size);
    a_host = (int*)malloc(data_size);
    b_cpu = (int*)malloc(data_size);

    // Reservar en GPU (memoria de video) Memory Allocation
    GPUErrorAssertion(cudaMalloc((void**)&c_gpu, data_size));
    GPUErrorAssertion(cudaMalloc((void**)&a_device, data_size));
    GPUErrorAssertion(cudaMalloc((void**)&b_gpu, data_size));


    // 2 --------------------- Transferir a Memoria GPU (inicio, fin, size, tipo) ----------------------
    GPUErrorAssertion(cudaMemcpy(c_gpu, c_cpu, data_size, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(a_device, a_host, data_size, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(b_gpu, b_cpu, data_size, cudaMemcpyHostToDevice));


    // 3 --------------------- Lanzar kernel -----------------------------------------------------------
    print_all_idx << <gridSize, blockSize >> > ();


    // 4 --------------------- Transferir de Memoria GPU a CPU (inicio, fin, size, tipo) ---------------
    GPUErrorAssertion(cudaMemcpy(c_cpu, c_gpu, data_size, cudaMemcpyDeviceToHost));
    GPUErrorAssertion(cudaMemcpy(a_host, a_device, data_size, cudaMemcpyDeviceToHost));
    GPUErrorAssertion(cudaMemcpy(b_cpu, b_gpu, data_size, cudaMemcpyDeviceToHost));


    // 5 --------------------- Limpieza, Reset, Liberar Memoria ----------------------------------------
    cudaDeviceReset();
    cudaFree(c_gpu);
    cudaFree(a_device);
    cudaFree(b_gpu);

    return 0;
}