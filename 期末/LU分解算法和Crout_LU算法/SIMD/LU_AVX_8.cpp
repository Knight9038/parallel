#include <iostream>
#include <vector>
#include <immintrin.h> // Include AVX intrinsics
#include <chrono>

using namespace std;

constexpr int N = 2048;
vector<vector<float>> m(N, vector<float>(N, 0.0)); // Initialize matrix m

void printMatrix(const vector<vector<float>>& mat) {
    for (const auto& row : mat) {
        for (float val : row) {
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
            m[i][j] = rand() / (float)RAND_MAX; // 非对角线元素为 0 到 1 之间的随机数
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

void luDecomposition(vector<vector<float>>& mat, vector<vector<float>>& lower, vector<vector<float>>& upper) {
    int n = mat.size();
    lower = vector<vector<float>>(n, vector<float>(n, 0.0));
    upper = vector<vector<float>>(n, vector<float>(n, 0.0));

    // Perform LU decomposition
    for (int i = 0; i < n; i++) {
        lower[i][i] = 1.0;

        for (int j = i; j < n; j++) {
            int k = 0;
            // Compute upper triangular matrix
            __m256 sum_vec = _mm256_setzero_ps();
            for (k = 0; k < i; k += 8) {
                __m256 lower_vec = _mm256_loadu_ps(&lower[i][k]);
                __m256 upper_vec = _mm256_loadu_ps(&upper[k][j]);
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(lower_vec, upper_vec));
            }
            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum_vec);
            float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
            for (; k < i; k++) {
                sum += lower[i][k] * upper[k][j];
            }
            upper[i][j] = mat[i][j] - sum;
        }

        for (int j = i + 1; j < n; j++) {
            // Compute lower triangular matrix
            int k;
            __m256 sum_vec = _mm256_setzero_ps();
            for (k = 0; k < i; k += 8) {
                __m256 lower_vec = _mm256_loadu_ps(&lower[j][k]);
                __m256 upper_vec = _mm256_loadu_ps(&upper[k][i]);
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(lower_vec, upper_vec));
            }
            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum_vec);
            float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3]+ sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
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

    using namespace std::chrono;
    float count = 0;
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
    cout << "AVX: " << count << "ms" << endl;
    /*luDecomposition(m, L, U);
    cout << "Lower triangular matrix L:" << endl;
    printMatrix(L);

    cout << "Upper triangular matrix U:" << endl;
    printMatrix(U);*/

    return 0;
}