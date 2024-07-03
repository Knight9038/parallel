#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <CL/sycl.hpp>
#include <ctime>

using namespace std;
using namespace sycl;

const int N = 4096;
using ele_t = float;

void LU_gpu(ele_t mat[N][N], int n) {
    queue q{ cpu_selector{} };
    // queue q{ gpu_selector{} };

    ele_t(*new_mat)[N] = (ele_t(*)[N])malloc_shared<ele_t>(N * N, q);
    memcpy(new_mat, mat, sizeof(ele_t) * N * N);

    timespec start, end;
    double time_used = 0;
    clock_gettime(CLOCK_REALTIME, &start);

    for (int i = 0; i < n; i++) {
        q.parallel_for(range{ (unsigned long)(n - i - 1) }, [=](id<1> idx) {
            int j = idx[0] + i + 1;
            ele_t div = new_mat[j][i] / new_mat[i][i];
            for (int k = i; k < n; k++) {
                new_mat[j][k] -= new_mat[i][k] * div;
            }
        }).wait();
    }

    clock_gettime(CLOCK_REALTIME, &end);
    time_used += end.tv_sec - start.tv_sec;
    time_used += double(end.tv_nsec - start.tv_nsec) / 1000000000;

    std::cout << "并行算法用时: " << time_used << "秒" << std::endl;
    std::cout << "并行算法使用设备: " << q.get_device().get_info<info::device::name>() << std::endl;
}

int main() {
    static ele_t mat[N][N];
    // Initialize matrix with random values
    srand(time(nullptr));
    for (int i = 0; i < N; i++) {
        mat[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            mat[i][j] = rand() % RAND_MAX + 2;
            mat[j][i] = rand() % RAND_MAX + 2;
        }
    }

    LU_gpu(mat, N);

    return 0;
}