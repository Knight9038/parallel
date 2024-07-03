#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include <immintrin.h>

int N = 512;

std::vector<std::vector<float>> m(N, std::vector<float>(N, 0.0));

void printMatrix(const std::vector<std::vector<float>>& mat) {
    for (const auto& row : mat) {
        for (float val : row) {
            std::cout << int(val) << " ";
        }
        std::cout << std::endl;
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

void luDecomposition(std::vector<std::vector<float>>& mat, std::vector<std::vector<float>>& lower, std::vector<std::vector<float>>& upper) {
    int n = mat.size();
    lower = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0));
    upper = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0));

    #pragma omp parallel num_threads(8)
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
            lower[i][i] = 1.0;

            for (int j = i; j < n; j++) {
                float sum = 0.0;
                __m512 sum_vec = _mm512_setzero_ps();
                for (int k = 0; k < i; k+=16) {
                    __m512 lower_vec = _mm512_loadu_ps(&lower[i][k]);
                    __m512 upper_vec = _mm512_loadu_ps(&upper[k][j]);
                    sum_vec = _mm512_fmadd_ps(lower_vec, upper_vec, sum_vec);
                }
                sum += _mm512_reduce_add_ps(sum_vec);
                upper[i][j] = mat[i][j] - sum;
            }

            for (int j = i + 1; j < n; j++) {
                float sum = 0.0;
                __m512 sum_vec = _mm512_setzero_ps();
                for (int k = 0; k < i; k+=16) {
                    __m512 lower_vec = _mm512_loadu_ps(&lower[j][k]);
                    __m512 upper_vec = _mm512_loadu_ps(&upper[k][i]);
                    sum_vec = _mm512_fmadd_ps(lower_vec, upper_vec, sum_vec);
                }
                sum += _mm512_reduce_add_ps(sum_vec);
                lower[j][i] = (mat[j][i] - sum) / upper[i][i];
            }
        }
    }
}

int main() {
    m_reset();
    int iterations = 1;
    std::vector<std::vector<float>> L, U;

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
    std::cout << "OpenMP with 8 threads and AVX-512: " << count << "ms" << std::endl;


    return 0;
}