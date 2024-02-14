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

    // Printing in kernel bad (only to see here)
    printf("[DEVICE] threadIdx.x: %d, blockIdx.x %d, gridDim.x: %d, gidx: %d \n", tidx, bidx, gdimx, gidx);
    printf("[DEVICE] threadIdx.y: %d, blockIdx.y %d, gridDim.y: %d, gidy: %d \n", tidy, bidy, gdimy, gidy);
    printf("[DEVICE] threadIdx.z: %d, blockIdx.z %d, gridDim.z: %d, gidz: %d \n", tidz, bidz, gdimz, gidz);
    printf("[DEVICE] Global ID: %d \n", globalid);
}

int main()
{
    // 1 --------------------- Inicializar Datos -------------------------------------------------------
    dim3 blockSize(4, 2, 1);
    dim3 gridSize(2, 2, 1);

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
    print_all_idx << <gridSize, blockSize >> > ();


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
