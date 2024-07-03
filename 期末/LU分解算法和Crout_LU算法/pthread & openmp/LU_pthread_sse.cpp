#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <pthread.h>
#include <immintrin.h>

using namespace std;

int N = 1024;
int num_threads = 15; // Number of threads for parallel processing
vector<vector<float>> m(N, vector<float>(N, 0.0));
vector<vector<float>> L(N, vector<float>(N, 0.0));
vector<vector<float>> U(N, vector<float>(N, 0.0));

struct ThreadArgs {
    int start_row;
    int end_row;
};

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
        m[i][i] = 1.0;

        for (int j = i + 1; j < N; j++) {
            m[i][j] = rand() % RAND_MAX + 2;
            m[j][i] = rand() % RAND_MAX + 2;
        }
    }
}

void* luDecompositionParallel(void* args) {
    ThreadArgs* thread_args = (ThreadArgs*)args;
    int start_row = thread_args->start_row;
    int end_row = thread_args->end_row;

    for (int i = start_row; i <= end_row; i++) {
        __m128 sumVec = _mm_set1_ps(0.0);

        for (int k = 0; k < i; k+=4) {
            __m128 lSlice = _mm_loadu_ps(&L[i][k]);
            __m128 uSlice = _mm_loadu_ps(&U[k][0]);
            sumVec = _mm_add_ps(sumVec, _mm_mul_ps(lSlice, uSlice));
        }

        float sum[4];
        _mm_store_ps(sum, sumVec);
        U[i][0] = m[i][0] - sum[0];
        U[i][1] = m[i][1] - sum[1];
        U[i][2] = m[i][2] - sum[2];
        U[i][3] = m[i][3] - sum[3];

        __m128 uSliceDiag = _mm_set1_ps(U[i][i]);

        for (int j = i + 1; j < N; j++) {
            __m128 sumVec2 = _mm_set1_ps(0.0);

            for (int k = 0; k < i; k+=4) {
                __m128 lSlice2 = _mm_loadu_ps(&L[j][k]);
                __m128 uSlice2 = _mm_load1_ps(&U[k][i]);
                sumVec2 = _mm_add_ps(sumVec2, _mm_mul_ps(lSlice2, uSlice2));
            }

            float sum2[4];
            _mm_store_ps(sum2, sumVec2);
            L[j][i] = (m[j][i] - (sum2[0] + sum2[1] + sum2[2] + sum2[3])) / U[i][i];
        }
    }

    pthread_exit(NULL);
}

int main() {
    m_reset();
    int iterations = 1;

    pthread_t threads[num_threads];
    ThreadArgs thread_args[num_threads];

    using namespace std::chrono;
    float count = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        m_reset();

        auto start = steady_clock::now();

        int rows_per_thread = N / num_threads;
        int extra_rows = N % num_threads;
        int start_row = 0;
        for (int i = 0; i < num_threads; i++) {
            thread_args[i].start_row = start_row;
            thread_args[i].end_row = start_row + rows_per_thread - 1 + (i < extra_rows ? 1 : 0);
            pthread_create(&threads[i], NULL, luDecompositionParallel, (void*)&thread_args[i]);
            start_row = thread_args[i].end_row + 1;
        }

        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        auto end = steady_clock::now();
        duration<float, milli> duration = end - start;
        count += duration.count();
    }

    cout << "Parallel: " << count << "ms" << endl;

    return 0;
}
