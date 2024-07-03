#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <immintrin.h> // 包含SSE指令头文件

using namespace std;

int N = 1024;

vector<vector<double>> m(N, vector<double>(N, 0.0)); // 初始化全局矩阵 m
vector<vector<double>> L(N, vector<double>(N, 0.0)); // 新增Lower矩阵
vector<vector<double>> U(N, vector<double>(N, 0.0)); // 新增Upper矩阵

void printMatrix(const vector<vector<double>>& mat) {
    for (const auto& row : mat) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

void m_reset() {
    srand(time(nullptr));

    for (int i = 0; i < N; i++) {
        m[i][i] = 1.0;

        for (int j = i + 1; j < N; j++) {
            m[i][j] = rand() % RAND_MAX + 2;
            m[j][i] = rand() % RAND_MAX + 2;
        }
    }
}

void croutLuDecomposition(vector<vector<double>>& mat, vector<vector<double>>& lower, vector<vector<double>>& upper) {
    int n = mat.size();

    for (int i = 0; i < n; i++) {
        lower[i][i] = 1.0;

        // 优化上三角矩阵
        for (int j = i; j < n; j++) {
            __m256d sum = _mm256_setzero_pd();
            for (int k = 0; k < i; k += 4) {
                __m256d lower_vec = _mm256_loadu_pd(&lower[i][k]);
                __m256d upper_vec = _mm256_loadu_pd(&upper[k][j]);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(lower_vec, upper_vec));
            }
            double sum_array[4];
            _mm256_storeu_pd(sum_array, sum);
            double total_sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
            for (int k = (i / 4) * 4; k < i; k++) {
                total_sum += lower[i][k] * upper[k][j];
            }
            upper[i][j] = mat[i][j] - total_sum;
        }

        // 优化下三角矩阵
        for (int j = i + 1; j < n; j++) {
            __m256d sum = _mm256_setzero_pd();
            for (int k = 0; k < i; k += 4) {
                __m256d lower_vec = _mm256_loadu_pd(&lower[j][k]);
                __m256d upper_vec = _mm256_loadu_pd(&upper[k][i]);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(lower_vec, upper_vec));
            }
            double sum_array[4];
            _mm256_storeu_pd(sum_array, sum);
            double total_sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
            for (int k = (i / 4) * 4; k < i; k++) {
                total_sum += lower[j][k] * upper[k][i];
            }
            lower[j][i] = (mat[j][i] - total_sum) / upper[i][i];
        }
    }
}

int main() {
    m_reset();
    int iterations = 1;

    using namespace std::chrono;
    double count = 0;

    for (int i = 0; i < iterations; ++i)
    {
        m_reset();
        auto start = steady_clock::now();
        croutLuDecomposition(m, L, U);
        auto end = steady_clock::now();
        duration<double, milli> duration = end - start;
        count += duration.count();
    }
    cout << "Crout-LU: " << count << "ms" << endl;

    /*cout << "Lower triangular matrix L:" << endl;
    printMatrix(L);

    cout << "Upper triangular matrix U:" << endl;
    printMatrix(U);*/

    return 0;
}