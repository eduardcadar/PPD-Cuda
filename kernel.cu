#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "jpge.h"
#include "jpgd.h"

#include <stdio.h>
#include <string>
#include <chrono>
using namespace std;

typedef unsigned char uint8;

cudaError_t blurWithCuda(const uint8* image_data, const int height, const int width, uint8* new_image_data, const int noThreads);

__global__ void boxBlurKernel(const uint8* image_data, const int height, const int width, uint8* new_image_data, const int no_threads)
{
    const int pixelsX[9] = { -1, +0, +1, -1, +0, +1, -1, +0, +1 };
    const int pixelsY[9] = { -1, -1, -1, +0, +0, +0, +1, +1, +1 };
    int threadX = threadIdx.x, blockX = blockIdx.x;

    int start = 0;
    int size = height * width;
    int end = size / no_threads;
    int r = size % no_threads;
    int crtStart, crtEnd;
    for (int i = 0; i <= threadX; i++) {
        if (r > 0) end++, r--;
        crtStart = start, crtEnd = end;
        start = end;
        end += size / no_threads;
    }

    for (int i = crtStart; i < crtEnd; i++) {
        int y = i / width;
        int x = i % width;

        // pixelii de pe margine isi pastreaza valoarea
        if (x < 1 || x + 1 == width || y < 1 || y + 1 == height) {
            new_image_data[(y * width * 3) + x * 3 + blockX] = image_data[(y * width * 3) + x * 3 + blockX];
            continue;
        }

        int sum = 0;
        for (int j = 0; j < 9; j++) {
            int pixelX = x + pixelsX[j];
            int pixelY = y + pixelsY[j];
            // blockX determines if the thread calculates the red, blue or green value (0, 1, 2)
            sum += image_data[(pixelY * width * 3) + pixelX * 3 + blockX];
        }
        new_image_data[(y * width * 3) + x * 3 + blockX] = sum / 9;
    }
}

void boxBlurCPU(const uint8* image_data, const int height, const int width, uint8* new_image_data)
{
    const int pixelsX[9] = { -1, +0, +1, -1, +0, +1, -1, +0, +1 };
    const int pixelsY[9] = { -1, -1, -1, +0, +0, +0, +1, +1, +1 };
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // pixelii de pe margine isi pastreaza valoarea
            if (x < 1 || x + 1 == width || y < 1 || y + 1 == height) {
                new_image_data[(y * width * 3) + x * 3 + 0] = image_data[(y * width * 3) + x * 3 + 0];
                new_image_data[(y * width * 3) + x * 3 + 1] = image_data[(y * width * 3) + x * 3 + 1];
                new_image_data[(y * width * 3) + x * 3 + 2] = image_data[(y * width * 3) + x * 3 + 2];
                continue;
            }
            int sumR = 0, sumG = 0, sumB = 0;
            for (int i = 0; i < 9; i++) {
                int pixelX = x + pixelsX[i];
                int pixelY = y + pixelsY[i];
                sumR += image_data[(pixelY * width * 3) + pixelX * 3 + 0];
                sumG += image_data[(pixelY * width * 3) + pixelX * 3 + 1];
                sumB += image_data[(pixelY * width * 3) + pixelX * 3 + 2];
            }
            new_image_data[(y * width * 3) + x * 3 + 0] = sumR / 9;
            new_image_data[(y * width * 3) + x * 3 + 1] = sumG / 9;
            new_image_data[(y * width * 3) + x * 3 + 2] = sumB / 9;
        }
    }
}

int main()
{
    const int no_threads = 8;
    const string image_name = "sample";
    const string input_file = image_name + ".jpeg";
    const string output_file = "new_" + image_name + ".jpeg";
    const char* file_name = input_file.c_str();
    const char* new_file_name = output_file.c_str();

    const int req_comps = 3;
    int width = 0, height = 0, actual_comps = 0;
    uint8* pImage_data = jpgd::decompress_jpeg_image_from_file(file_name, &width, &height, &actual_comps, req_comps);
    if (!pImage_data) {
        printf("Nu am putut incarca fisierul JPEG \"%s\"\n", file_name);
        return -1;
    }
    if (actual_comps != req_comps) {
        printf("Imaginea nu este RGB, nu are 3 componente, ci doar %d\n", actual_comps);
        free(pImage_data);
        return -1;
    }
    uint8* new_image_data_CPU = (uint8*)malloc(req_comps * width * height * sizeof uint8);
    uint8* new_image_data_GPU = (uint8*)malloc(req_comps * width * height * sizeof uint8);

    printf("height: %d, width: %d\n", height, width);

    auto startCPU = std::chrono::system_clock::now();
    boxBlurCPU(pImage_data, height, width, new_image_data_CPU);
    auto endCPU = std::chrono::system_clock::now();
    auto timeCPU = endCPU - startCPU;
    printf("time CPU = %f ms\n", chrono::duration <double, milli>(timeCPU).count());

    auto startGPU = std::chrono::system_clock::now();
    cudaError_t cudaStatus = blurWithCuda(pImage_data, height, width, new_image_data_GPU, no_threads);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "blurWithCuda failed!");
        return 1;
    }
    auto endGPU = std::chrono::system_clock::now();
    auto timeGPU = endGPU - startGPU;
    printf("time GPU = %f ms\n", chrono::duration <double, milli>(timeGPU).count());

    for (int i = 0; i < req_comps * width * height; i++) {
        if (new_image_data_CPU[i] != new_image_data_GPU[i])
            printf("pixel %d different: cpu %d gpu %d\n", i, new_image_data_CPU[i], new_image_data_GPU[i]);
    }

    jpge::params params;
    params.m_quality = 90;
    params.m_two_pass_flag = true;
    jpge::compress_image_to_jpeg_file(new_file_name, width, height, req_comps, new_image_data_CPU, params);

    free(pImage_data);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to blur images in parallel.
cudaError_t blurWithCuda(const uint8* image_data, const int height, const int width, uint8* new_image_data, const int no_threads)
{
    uint8* dev_image_data = 0;
    uint8* dev_new_image_data = 0;
    cudaError_t cudaStatus;
    int size = 3 * width * height;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)& dev_image_data, size * sizeof(uint8));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_new_image_data, size * sizeof(uint8));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_image_data, image_data, size * sizeof(uint8), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    boxBlurKernel<<< 3, no_threads >>>(dev_image_data, height, width, dev_new_image_data, no_threads);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "boxBlurKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching boxBlurKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(new_image_data, dev_new_image_data, size * sizeof(uint8), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_image_data);
    cudaFree(dev_new_image_data);
    
    return cudaStatus;
}
