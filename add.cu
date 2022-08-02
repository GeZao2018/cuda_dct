// add.cpp
#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

__global__ void cuda_add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    
    int N = 1<<24; // 30M elements

    float *x = new float[N];
    float *y = new float[N];
    float *x2, *y2;
    cudaMallocManaged(&x2, sizeof(float)*N);
    cudaMallocManaged(&y2, sizeof(float)*N);

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
        x2[i] = 1.0f;
        y2[i] = 2.0f;
    }

    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1,NULL);
    // Run kernel on 30M elements on the CPU
    add(N, x, y);
    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) * 1000 + (double)(t2.tv_usec - t1.tv_usec)/1000.0;
    std::cout << "add cpu time: " << timeuse << "ms" << std::endl;

    gettimeofday(&t1, NULL);
    cuda_add<<<2, 512>>>(N, x2, y2);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) * 1000 + (double)(t2.tv_usec - t1.tv_usec)/1000.0;
    std::cout << "add gpu time: " << timeuse << "ms" << std::endl;

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f, maxError2 = 0.0f;
    for (int i = 0; i < N; i++) 
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
        maxError2 = fmax(maxError2, fabs(y2[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << ", " << maxError2 << std::endl;

    // Free memory
    delete [] x;
    delete [] y;
    cudaFree(x2);
    cudaFree(y2);

    return 0;
}