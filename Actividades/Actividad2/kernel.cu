
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

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
    // 1 --------------------- Inicializar Datos -------------------------------------------------------
    dim3 blockSize(4,4,4);
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
    cudaMalloc((void**)&c_gpu, data_size);
    cudaMalloc((void**)&a_device, data_size);
    cudaMalloc((void**)&b_gpu, data_size);


    // 2 --------------------- Transferir a Memoria GPU (inicio, fin, size, tipo) ----------------------
    cudaMemcpy(c_gpu, c_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(a_device, a_host, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, data_size, cudaMemcpyHostToDevice);


    // 3 --------------------- Lanzar kernel -----------------------------------------------------------
    print_all_idx << <gridSize , blockSize >> > ();


    // 4 --------------------- Transferir de Memoria GPU a CPU (inicio, fin, size, tipo) ---------------
    cudaMemcpy(c_cpu, c_gpu, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(a_host, a_device, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b_gpu, data_size, cudaMemcpyDeviceToHost);


    // 5 --------------------- Limpieza, Reset, Liberar Memoria ----------------------------------------
    cudaDeviceReset();
    cudaFree(c_gpu);
    cudaFree(a_device);
    cudaFree(b_gpu);

    return 0;
}