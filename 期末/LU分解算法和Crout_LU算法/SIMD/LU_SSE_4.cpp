#include <iostream>
#include <vector>
#include <emmintrin.h>
#include <stdlib.h> // 添加此头文件以使用 rand() 函数
#include <chrono>

using namespace std;


int N = 2048;

vector<vector<float>> m(N, vector<float>(N, 0.0)); // 初始化全局矩阵 m

void printMatrix(const vector<vector<float>>& mat) {
    for (const auto& row : mat) {
        for (float val : row) {
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

void luDecomposition(vector<vector<float>>& mat, vector<vector<float>>& lower, vector<vector<float>>& upper) {
    int n = mat.size();
    lower = vector<vector<float>>(n, vector<float>(n, 0.0));
    upper = vector<vector<float>>(n, vector<float>(n, 0.0));

    // Perform LU decomposition
    for (int i = 0; i < n; i++) {
        lower[i][i] = 1.0;

        for (int j = i; j < n; j++) {
            // Compute upper triangular matrix
            float sum = 0.0;
            int k;
            __m128 sum_vec = _mm_setzero_ps();
            for (k = 0; k < i - 1; k += 4) { // 修改此处为4路向量操作
                __m128 lower_vec = _mm_loadu_ps(&lower[i][k]);
                __m128 upper_vec = _mm_loadu_ps(&upper[k][j]);
                __m128 mul_vec = _mm_mul_ps(lower_vec, upper_vec);
                sum_vec = _mm_add_ps(sum_vec, mul_vec);
            }
            float sum_arr[4];
            _mm_storeu_ps(sum_arr, sum_vec);
            sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

            for (; k < i; k++) {
                sum += lower[i][k] * upper[k][j];
            }
            upper[i][j] = mat[i][j] - sum;
        }

        for (int j = i + 1; j < n; j++) {
            // Compute lower triangular matrix
            float sum = 0.0;
            int k;
            __m128 sum_vec = _mm_setzero_ps();
            for (k = 0; k < i - 1; k += 4) { // 修改此处为4路向量操作
                __m128 lower_vec = _mm_loadu_ps(&lower[j][k]);
                __m128 upper_vec = _mm_loadu_ps(&upper[k][i]);
                __m128 mul_vec = _mm_mul_ps(lower_vec, upper_vec);
                sum_vec = _mm_add_ps(sum_vec, mul_vec);
            }
            float sum_arr[4];
            _mm_storeu_ps(sum_arr, sum_vec);
            sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

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
    vector<vector<float>> L, U;
    double count = 0;
    using namespace std::chrono;

    // Your code for matrix initialization here
    for (int i = 0; i < iterations; ++i)
    {
        m_reset();
        auto start = steady_clock::now();
        luDecomposition(m, L, U);
        auto end = steady_clock::now();
        duration<float, milli> duration = end - start;
        count += duration.count();
    }
    cout << "SSE: " << count << "ms" << endl;
    //cout << "Lower triangular matrix L:" << endl;
    //printMatrix(L);

   // cout << "Upper triangular matrix U:" << endl;
    //printMatrix(U);

    return 0;
}