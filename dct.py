import math
import cv2
import time
import numpy as np
from numba import cuda


@cuda.jit
def matmul(a, b, c):
    row, col = cuda.grid(2)
    if row < c.shape[0] and col < c.shape[1]:
        tmp = 0.
        for k in range(a.shape[1]):
            tmp += a[row, k] * b[k, col]
        c[row, col] = tmp


@cuda.jit
def dct_gpu(block, alpha, beta, mcos, result):
    x, y = cuda.grid(2)
    tmp = 0.
    for i in range(256):
        for j in range(256):
            tmp += block[i][j] * mcos[i][x] * mcos[j][y]
    result[x][y] = alpha[x] * beta[y] * tmp


def test_dct_gpu(img, dct_size=256):
    m, n = img.shape
    img = img.astype(np.float32)
    img_device = cuda.to_device(img)
    result = cuda.device_array((256, 256))

    mcos = np.zeros((dct_size, dct_size), dtype=np.float32)
    alpha = np.zeros(dct_size, dtype=np.float32)
    beta = np.zeros(dct_size, dtype=np.float32)
    alpha[1:] = beta[1:] = math.sqrt(2. / dct_size)
    alpha[0] = beta[0] = math.sqrt(1. / dct_size)
    for i in range(dct_size):
        for j in range(dct_size):
            mcos[i][j] = math.cos((2 * i + 1) * math.pi * j / (2 * dct_size))
    alpha_device = cuda.to_device(alpha)
    beta_device = cuda.to_device(beta)
    mcos_device = cuda.to_device(mcos)

    threads_per_block = (8, 8)
    blocks_per_grid = (m // 8, n // 8)
    dct_gpu[blocks_per_grid, threads_per_block](img_device, alpha_device, beta_device, mcos_device, result)
    cuda.synchronize()

    print("start processing in GPU")
    start_gpu = time.time()
    dct_gpu[blocks_per_grid, threads_per_block](img_device, alpha_device, beta_device, mcos_device, result)
    cuda.synchronize()
    end_gpu = time.time()
    print("GPU time: " + str(end_gpu - start_gpu))
    # print(result.copy_to_host())

def test_matmul():
    N = 256
    A = np.full((N, N), 1, np.float32)
    B = np.full((N, N), 4, np.float32)
    a_device = cuda.to_device(A)
    b_device = cuda.to_device(B)
    c_result = cuda.device_array((N, N))

    threads_per_block = (8, 8)
    blocks_per_grid_x = int(math.ceil(A.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(B.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    matmul[blocks_per_grid, threads_per_block](a_device, b_device, c_result)
    cuda.synchronize()
    print("start processing in GPU")
    start_gpu = time.time()
    for i in range(20000):
        matmul[blocks_per_grid, threads_per_block](a_device, b_device, c_result)
        cuda.synchronize()
    end_gpu = time.time()
    print("GPU time: " + str(end_gpu - start_gpu))
    # print(c_result.copy_to_host())


def original_dct(img, dct_size=256):

    a = np.zeros((dct_size, dct_size), dtype=np.float32)
    a[0, :] = np.sqrt(1 / dct_size)
    for i in range(1, dct_size):
        for j in range(dct_size):
            a[i][j] = np.sqrt(2/dct_size) * np.cos(np.pi*(j+.5)*i/dct_size)
    a_t = np.transpose(a)

    print("start processing in CPU")
    start_cpu = time.time()

    m, n = img.shape
    img = img.astype(np.float32)
    res = np.zeros((dct_size, dct_size), dtype=np.float32)
    for i in range(m-dct_size+1):
        for j in range(n-dct_size+1):
            # res = cv2.dct(img[i:i+dct_size, j:j+dct_size])
            res = np.dot(np.dot(a, img[i:i+dct_size, j:j+dct_size]), a_t)

    end_cpu = time.time()
    print("CPU time: " + str(end_cpu - start_cpu))  # 256: 175.69027137756348
    # print(res)


if __name__ == '__main__':
    img = np.random.randint(0, 256, size=(256, 256))
    test_dct_gpu(img)
    original_dct(img)
    # test_matmul()

    # img = np.random.randint(0, 256, size=(2048, 2048))
    # img = img.astype(np.float32)
    # ts = []
    # for dct_size in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    #     start = time.time()
    #     for i in range(1000):
    #         cv2.dct(img[:dct_size, :dct_size])
    #     end = time.time()
    #     print(str(dct_size), str(end-start))
    #     ts.append(end-start)
    # t = ts[0]
    # for i in range(len(ts)):
    #     ts[i] /= t
    # print(ts)