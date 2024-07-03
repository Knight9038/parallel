#include <iostream>
#include <vector>
#include <emmintrin.h>
#include <stdlib.h> // 添加此头文件以使用 rand() 函数
#include <chrono>

using namespace std;


int N = 2048;

vector<vector<double>> m(N, vector<double>(N, 0.0)); // 初始化全局矩阵 m

void printMatrix(const vector<vector<double>>& mat) {
    for (const auto& row : mat) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

void m_reset() {
    // 初始化 m 为单位矩阵加随机数
    srand(time(nullptr)); // 设置随机种子

    for (int i = 0; i < N; i++) {
        m[i][i] = 1.0; // 对角线元素为 1.0

        for (int j = i + 1; j < N; j++) {
            m[i][j] = rand() / (double)RAND_MAX; // 非对角线元素为 0 到 1 之间的随机数
        }
    }

    // 累加操作
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                m[i][j] += m[k][j];
            }
        }
    }
}

void luDecomposition(vector<vector<double>>& mat, vector<vector<double>>& lower, vector<vector<double>>& upper) {
    int n = mat.size();
    lower = vector<vector<double>>(n, vector<double>(n, 0.0));
    upper = vector<vector<double>>(n, vector<double>(n, 0.0));

    // Perform LU decomposition
    for (int i = 0; i < n; i++) {
        lower[i][i] = 1.0;

        for (int j = i; j < n; j++) {
            // Compute upper triangular matrix
            double sum = 0.0;
            int k;
            __m128d sum_vec = _mm_setzero_pd();
            for (k = 0; k < i - 1; k += 2) {
                __m128d lower_vec = _mm_loadu_pd(&lower[i][k]);
                __m128d upper_vec = _mm_loadu_pd(&upper[k][j]);
                __m128d mul_vec = _mm_mul_pd(lower_vec, upper_vec);
                sum_vec = _mm_add_pd(sum_vec, mul_vec);
            }
            double sum_arr[2];
            _mm_storeu_pd(sum_arr, sum_vec);
            sum = sum_arr[0] + sum_arr[1];

            for (; k < i; k++) {
                sum += lower[i][k] * upper[k][j];
            }
            upper[i][j] = mat[i][j] - sum;
        }

        for (int j = i + 1; j < n; j++) {
            // Compute lower triangular matrix
            double sum = 0.0;
            int k;
            __m128d sum_vec = _mm_setzero_pd();
            for (k = 0; k < i - 1; k += 2) { 
                __m128d lower_vec = _mm_loadu_pd(&lower[j][k]);
                __m128d upper_vec = _mm_loadu_pd(&upper[k][i]);
                __m128d mul_vec = _mm_mul_pd(lower_vec, upper_vec);
                sum_vec = _mm_add_pd(sum_vec, mul_vec);
            }
            double sum_arr[2];
            _mm_storeu_pd(sum_arr, sum_vec);
            sum = sum_arr[0] + sum_arr[1];

            for (; k < i; k++) {
                sum += lower[j][k] * upper[k][i];
            }
            lower[j][i] = (mat[j][i] - sum) / upper[i][i];
        }
    }
}

int main() {
    m_reset();
    int iterations = 1;
    vector<vector<double>> L, U;

    using namespace std::chrono;
    double count = 0;
    // Your code for matrix initialization here
    for (int i = 0; i < iterations; ++i)
    {
        m_reset();
        auto start = steady_clock::now();
        luDecomposition(m, L, U);
        auto end = steady_clock::now();
        duration<double, milli> duration = end - start;
        count += duration.count();
    }
    cout << "LU_SSE: " << count << "ms" << endl;
    /*luDecomposition(m, L, U);
    cout << "Lower triangular matrix L:" << endl;
    printMatrix(L);

    cout << "Upper triangular matrix U:" << endl;
    printMatrix(U);*/

    return 0;
}