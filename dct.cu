#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>

#define PI acos(-1)

using namespace cv;


__global__ void cuda_dct(float *image, float *alpha, float *temp, float *result) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            float tmp = 0.f;
            for (int k = 0; k < 256; k++) {
                tmp += alpha[i*256+k] * image[(ix+k)*1024+iy+j];
            }
            temp[i*256+j] = tmp;
        }
    }
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            float tmp = 0.f;
            for (int k = 0; k < 256; k++) {
                tmp += temp[i*256+k] * alpha[j*256+k];
            }
            result[i*256+j] = tmp;
        }
    }
}

__global__ void cuda_dct_2(float *image, float *alpha, float *result) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int r = 0; r < 256; r++) {
        for (int c = 0; c < 256; c++) {
            float tmp = 0.0f;
            for (int k = 0; k < 256; k++) {
                tmp += alpha[r * 256 + k] * image[(x + k) * 1024 + y + c];
            }
            for (int k = 0; k < 256; k++) {
                result[r * 256 + k] += tmp * alpha[k * 256 + c];
            }
        }
    }
}

// __syncthreads 只能块内线程同步
__global__ void cuda_matmul(int x, int y, float *image, float *alpha, float *temp, float *result) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    float tmp = 0.0f;
    for (int k = 0; k < 32; k++) {
        tmp += alpha[ix * 32 + k] * image[(x + k) * 1024 + y + iy];
    }
    temp[ix * 32 + iy] = tmp;
    __syncthreads();
    tmp = 0.0f;
    for (int k = 0; k < 32; k++) {
        tmp += temp[ix * 32 + k] * alpha[iy * 32 + k];
    }
    result[ix * 32 + iy] = tmp;
    __syncthreads();
}

__global__ void cuda_matmul_2(int x, int y, float *image, float *alpha, float *result) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    for (int i = 0; i < 256; i++) {
        float tmp = 0.0f;
        for (int j = 0; j < 256; j++) {
            tmp += alpha[ix * 256 + j] * image[(j+x) * 1024 + y+i];
        }
        result[ix * 256 + iy] += tmp * alpha[iy * 256 + i];
    }
}

void test_cuda_matmul_2(cv::Mat& img, int dct_size=256) {
    int m = img.rows, n = img.cols;
    float * image = new float[m * n];
    float * alpha = new float[dct_size * dct_size];
    float * temp = new float[dct_size * dct_size];
    float * result = new float[dct_size * dct_size];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            image[i * n + j] = img.at<float>(i, j);
        }
    }
    float a = sqrt(1. / dct_size), b = sqrt(2. / dct_size);
    for (int j = 0; j < dct_size; j++) alpha[j] = a;
    for (int i = 1; i < dct_size; i++) {
        for (int j = 0; j < dct_size; j++) {
            alpha[i*dct_size + j] = b * cos(PI*(j+0.5)*i/dct_size);
        }
    }

    float *d_image, *d_alpha, *d_temp, *d_result;
    cudaMalloc((void **)&d_image, m * n * sizeof(float));
    cudaMalloc((void **)&d_alpha, dct_size * dct_size * sizeof(float));
    cudaMalloc((void **)&d_temp, dct_size * dct_size * sizeof(float));
    cudaMalloc((void **)&d_result, dct_size * dct_size * sizeof(float));

    cudaMemcpy((void*)d_image, (void*)image, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_alpha, (void*)alpha, dct_size * dct_size * sizeof(float), cudaMemcpyHostToDevice);

    struct timeval t1, t2;
    double timeuse;
    printf("start processing in GPU\n");
    gettimeofday(&t1,NULL);

    dim3 block_size(32, 32);
    dim3 grid_size(8, 8);
    cuda_matmul_2<<<grid_size, block_size>>>(0, 0, d_image, d_alpha, d_result);
    cudaDeviceSynchronize();

    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) * 1000 + (double)(t2.tv_usec - t1.tv_usec)/1000.0;
    printf("GPU time: %lfms\n", timeuse);

    cudaMemcpy((void*)result, (void*)d_result, dct_size * dct_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("test_cuda_matmul_result\n");
    for (int i = 0; i < 256; i++) {
        printf("%lf, ", result[i]);
    }
    printf("\n");
}

void test_cuda_matmul(cv::Mat& img, int dct_size=32) {
    int m = img.rows, n = img.cols;
    float * image = new float[m * n];
    float * alpha = new float[dct_size * dct_size];
    float * temp = new float[dct_size * dct_size];
    float * result = new float[dct_size * dct_size];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            image[i * n + j] = img.at<float>(i, j);
        }
    }
    float a = sqrt(1. / dct_size), b = sqrt(2. / dct_size);
    for (int j = 0; j < dct_size; j++) alpha[j] = a;
    for (int i = 1; i < dct_size; i++) {
        for (int j = 0; j < dct_size; j++) {
            alpha[i*dct_size + j] = b * cos(PI*(j+0.5)*i/dct_size);
        }
    }

    float *d_image, *d_alpha, *d_temp, *d_result;
    cudaMalloc((void **)&d_image, m * n * sizeof(float));
    cudaMalloc((void **)&d_alpha, dct_size * dct_size * sizeof(float));
    cudaMalloc((void **)&d_temp, dct_size * dct_size * sizeof(float));
    cudaMalloc((void **)&d_result, dct_size * dct_size * sizeof(float));

    cudaMemcpy((void*)d_image, (void*)image, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_alpha, (void*)alpha, dct_size * dct_size * sizeof(float), cudaMemcpyHostToDevice);

    struct timeval t1, t2;
    double timeuse;
    printf("start processing in GPU\n");
    gettimeofday(&t1,NULL);

    dim3 block_size(32, 32);
    dim3 grid_size(1, 1);
    cuda_matmul<<<grid_size, block_size>>>(0, 0, d_image, d_alpha, d_temp, d_result);
    cudaDeviceSynchronize();

    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) * 1000 + (double)(t2.tv_usec - t1.tv_usec)/1000.0;
    printf("GPU time: %lfms\n", timeuse);

    cudaMemcpy((void*)result, (void*)d_result, dct_size * dct_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("test_cuda_matmul_result\n");
    for (int i = 0; i < 32; i++) {
        printf("%lf, ", result[i]);
    }
    printf("\n");
}

void test_dct_mul(float *image, float *alpha, float *temp, float *result) {
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < 256; k++) {
                tmp += alpha[i*256+k] * image[k*256+j];
            }
            temp[i*256+j] = tmp;
        }
    }
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < 256; k++) {
                tmp += temp[i*256+k] * alpha[j*256+k];
            }
            result[i*256+j] = tmp;
        }
    }
    printf("test_dct_mul\n");
    for (int i = 0; i < 256; i++) {
        printf("%lf, ", result[i]);
    }
    printf("\n");
}

void test_cuda_dct(cv::Mat& img, int dct_size=256) {
    int m = img.rows, n = img.cols;
    float * image = new float[m * n];
    float * alpha = new float[dct_size * dct_size];
    float * temp = new float[dct_size * dct_size];
    float * result = new float[dct_size * dct_size];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            image[i * n + j] = img.at<float>(i, j);
        }
    }
    float a = sqrt(1. / dct_size), b = sqrt(2. / dct_size);
    for (int j = 0; j < dct_size; j++) alpha[j] = a;
    for (int i = 1; i < dct_size; i++) {
        for (int j = 0; j < dct_size; j++) {
            alpha[i*dct_size + j] = b * cos(PI*(j+0.5)*i/dct_size);
        }
    }

    float *d_image, *d_alpha, *d_temp, *d_result;
    cudaMalloc((void **)&d_image, m * n * sizeof(float));
    cudaMalloc((void **)&d_alpha, dct_size * dct_size * sizeof(float));
    cudaMalloc((void **)&d_temp, dct_size * dct_size * sizeof(float));
    cudaMalloc((void **)&d_result, dct_size * dct_size * sizeof(float));

    cudaMemcpy((void*)d_image, (void*)image, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_alpha, (void*)alpha, dct_size * dct_size * sizeof(float), cudaMemcpyHostToDevice);

    struct timeval t1, t2;
    double timeuse;
    printf("start processing in GPU\n");
    gettimeofday(&t1,NULL);

    dim3 block_size(32, 32);
    dim3 grid_size(1, 1);
    cuda_dct<<<grid_size, block_size>>>(d_image, d_alpha, d_temp, d_result);
    cudaDeviceSynchronize();

    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) * 1000 + (double)(t2.tv_usec - t1.tv_usec)/1000.0;
    printf("GPU time: %lfms\n", timeuse);

    cudaMemcpy((void*)result, (void*)d_result, dct_size * dct_size * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("test_cuda_result\n");
    // for (int i = 0; i < 256; i++) {
    //     printf("%lf, ", result[i]);
    // }
    // printf("\n");

    // test_dct_mul(image, alpha, temp, result);
}

void test_cuda_dct_2(Mat& img, int dct_size=256) {
    int m = img.rows, n = img.cols;
    float * image = new float[m * n];
    float * alpha = new float[dct_size * dct_size];
    float * result = new float[dct_size * dct_size];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            image[i * n + j] = img.at<float>(i, j);
        }
    }
    float a = sqrt(1. / dct_size), b = sqrt(2. / dct_size);
    for (int j = 0; j < dct_size; j++) alpha[j] = a;
    for (int i = 1; i < dct_size; i++) {
        for (int j = 0; j < dct_size; j++) {
            alpha[i*dct_size + j] = b * cos(PI*(j+0.5)*i/dct_size);
        }
    }

    float *d_image, *d_alpha, *d_result;
    cudaMalloc((void **)&d_image, m * n * sizeof(float));
    cudaMalloc((void **)&d_alpha, dct_size * dct_size * sizeof(float));
    cudaMalloc((void **)&d_result, dct_size * dct_size * sizeof(float));

    cudaMemcpy((void*)d_image, (void*)image, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_alpha, (void*)alpha, dct_size * dct_size * sizeof(float), cudaMemcpyHostToDevice);

    struct timeval t1, t2;
    double timeuse;
    printf("start processing in GPU\n");
    gettimeofday(&t1,NULL);

    dim3 block_size(32, 32);
    dim3 grid_size(24, 24);
    cuda_dct_2<<<grid_size, block_size>>>(d_image, d_alpha, d_result);
    cudaDeviceSynchronize();

    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) * 1000 + (double)(t2.tv_usec - t1.tv_usec)/1000.0;
    printf("GPU time: %lfms\n", timeuse);

    cudaMemcpy((void*)result, (void*)d_result, dct_size * dct_size * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("test_cuda_result\n");
    // for (int i = 0; i < 256; i++) {
    //     printf("%lf, ", result[i]);
    // }
    // printf("\n");
}


void original_dct(cv::Mat& img, int dct_size=256) {

    struct timeval t1, t2;
    double timeuse;
    printf("start processing in CPU\n");
    gettimeofday(&t1,NULL);

    int m = img.rows, n = img.cols;
    Mat out = Mat::zeros(Size(dct_size, dct_size), CV_32F);
    for (int i = 0; i <= m-dct_size; i++) {
        for (int j = 0; j <= n-dct_size; j++) {
            dct(img(Rect(i, j, dct_size, dct_size)), out);
        }
    }

    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) * 1000 + (double)(t2.tv_usec - t1.tv_usec)/1000.0;
    printf("CPU time: %lfms\n", timeuse);   // CPU time: 457360.401000ms, 454779.757000ms, 428182.293000ms

    for (int j = 0; j < dct_size; j++) {
        printf("%lf, ", out.at<float>(0, j));
    }
    printf("\n");
}


int main() {
    cv::Mat img = cv::imread("image.jpg", 0);
    cv::Mat img_float;
    // resize(img, img, Size(256, 256));
    img.convertTo(img_float, CV_32FC1);
 
    // original_dct(img_float);
    // test_cuda_dct(img_float);
    test_cuda_dct_2(img_float);

    // test_cuda_matmul(img_float);
    // test_cuda_matmul_2(img_float);

    // int dev = 0;
    // cudaDeviceProp devProp;
    // cudaGetDeviceProperties(&devProp, dev);
    // std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    // std::cout << "SM的数量: " << devProp.multiProcessorCount << std::endl;
    // std::cout << "每个线程块的共享内存大小: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    // std::cout << "每个线程块的最大线程数: " << devProp.maxThreadsPerBlock << std::endl;
    // std::cout << "每个EM的最大线程数: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    // std::cout << "每个SM的最大线程束数: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    
    return 0;
}