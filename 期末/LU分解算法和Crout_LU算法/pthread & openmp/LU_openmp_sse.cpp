#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include <immintrin.h> // Include header for SSE4
using namespace std;

const int N = 512;

vector<vector<float>> m(N, vector<float>(N, 0.0f));

void printMatrix(const vector<vector<float>>& mat) {
    for (const auto& row : mat) {
        for (float val : row) {
            cout << int(val) << " ";
        }
        cout << endl;
    }
}

void m_reset() {
    srand(time(nullptr));

    for (int i = 0; i < N; i++) {
        m[i][i] = 1.0f;

        for (int j = i + 1; j < N; j++) {
            m[i][j] = rand() % RAND_MAX + 2;
            m[j][i] = rand() % RAND_MAX + 2;
        }
    }
}

void luDecomposition(vector<vector<float>>& mat, vector<vector<float>>& lower, vector<vector<float>>& upper) {
    int n = mat.size();
    lower = vector<vector<float>>(n, vector<float>(n, 0.0f));
    upper = vector<vector<float>>(n, vector<float>(n, 0.0f));

    #pragma omp parallel num_threads(8)
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
            lower[i][i] = 1.0f;

            for (int j = i; j < n; j++) {
                __m128 sum = _mm_setzero_ps(); // Initialize sum to zero vector
                for (int k = 0; k < i; k += 4) { // Process 4 elements at a time with SSE
                    __m128 l = _mm_loadu_ps(&lower[i][k]); // Load 4 elements from lower
                    __m128 u = _mm_loadu_ps(&upper[k][j]); // Load 4 elements from upper
                    __m128 prod = _mm_mul_ps(l, u); // Multiply corresponding elements
                    sum = _mm_add_ps(sum, prod); // Accumulate sum
                }
                float sum_arr[4];
                _mm_storeu_ps(sum_arr, sum); // Store sum back to array
                float sum_scalar = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3]; // Reduce sum to scalar
                upper[i][j] = mat[i][j] - sum_scalar;
            }

            for (int j = i + 1; j < n; j++) {
                __m128 sum = _mm_setzero_ps();
                for (int k = 0; k < i; k += 4) {
                    __m128 l = _mm_loadu_ps(&lower[j][k]);
                    __m128 u = _mm_loadu_ps(&upper[k][i]);
                    __m128 prod = _mm_mul_ps(l, u);
                    sum = _mm_add_ps(sum, prod);
                }
                float sum_arr[4];
                _mm_storeu_ps(sum_arr, sum);
                float sum_scalar = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
                lower[j][i] = (mat[j][i] - sum_scalar) / upper[i][i];
            }
        }
    }
}

int main() {
    m_reset();
    int iterations = 1;
    vector<vector<float>> L, U;

    using namespace std::chrono;
    float count = 0;
    for (int i = 0; i < iterations; ++i)
    {
        m_reset();
        auto start = steady_clock::now();
        luDecomposition(m, L, U);
        auto end = steady_clock::now();
        duration<float, milli> duration = end - start;
        count += duration.count();
    }
    cout << "OpenMP with 8 threads and SSE4: " << count << "ms" << endl;

    return 0;
}