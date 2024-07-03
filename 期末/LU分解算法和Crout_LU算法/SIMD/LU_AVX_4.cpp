#include <iostream>
#include <vector>
#include <immintrin.h> // Include AVX intrinsics
#include <chrono>

using namespace std;

constexpr int N = 2048;
vector<vector<double>> m(N, vector<double>(N, 0.0)); // Initialize matrix m

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
            int k = 0;
            // Compute upper triangular matrix
            __m256d sum_vec = _mm256_setzero_pd();
            for (k = 0; k < i; k += 4) {
                __m256d lower_vec = _mm256_loadu_pd(&lower[i][k]);
                __m256d upper_vec = _mm256_loadu_pd(&upper[k][j]);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(lower_vec, upper_vec));
            }
            double sum_array[4];
            _mm256_storeu_pd(sum_array, sum_vec);
            double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
            for (; k < i; k++) {
                sum += lower[i][k] * upper[k][j];
            }
            upper[i][j] = mat[i][j] - sum;
        }

        for (int j = i + 1; j < n; j++) {
            // Compute lower triangular matrix
            int k;
            __m256d sum_vec = _mm256_setzero_pd();
            for (k = 0; k < i; k += 4) {
                __m256d lower_vec = _mm256_loadu_pd(&lower[j][k]);
                __m256d upper_vec = _mm256_loadu_pd(&upper[k][i]);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(lower_vec, upper_vec));
            }
            double sum_array[4];
            _mm256_storeu_pd(sum_array, sum_vec);
            double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
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
    cout << "AVX: " << count << "ms" << endl;
    /*luDecomposition(m, L, U);
    cout << "Lower triangular matrix L:" << endl;
    printMatrix(L);

    cout << "Upper triangular matrix U:" << endl;
    printMatrix(U);*/

    return 0;
}