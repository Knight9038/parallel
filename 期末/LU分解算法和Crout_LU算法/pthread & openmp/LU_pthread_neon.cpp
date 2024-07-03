#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <pthread.h>
#include <arm_neon.h> // Include NEON header

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


void luDecompositionParallel(void* args) {
    ThreadArgs* thread_args = (ThreadArgs*)args;
    int start_row = thread_args->start_row;
    int end_row = thread_args->end_row;

    for (int i = start_row; i <= end_row; i++) {
        L[i][i] = 1.0;

        for (int j = i; j < N; j++) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            for (int k = 0; k < i; k+=4) {
                float32x4_t L_val = vld1q_f32(&L[i][k]);
                float32x4_t U_val = vld1q_f32(&U[k][j]);
                sum_vec = vmlaq_f32(sum_vec, L_val, U_val);
            }
            // Horizontal add of sum_vec
            float sum = vgetq_lane_f32(sum_vec, 0) +
                        vgetq_lane_f32(sum_vec, 1) +
                        vgetq_lane_f32(sum_vec, 2) +
                        vgetq_lane_f32(sum_vec, 3);
            U[i][j] = m[i][j] - sum;
        }

        for (int j = i + 1; j < N; j++) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            for (int k = 0; k < i; k+=4) {
                float32x4_t L_val = vld1q_f32(&L[j][k]);
                float32x4_t U_val = vld1q_f32(&U[k][i]);
                sum_vec = vmlaq_f32(sum_vec, L_val, U_val);
            }
            // Horizontal add of sum_vec
            float sum = vgetq_lane_f32(sum_vec, 0) +
                        vgetq_lane_f32(sum_vec, 1) +
                        vgetq_lane_f32(sum_vec, 2) +
                        vgetq_lane_f32(sum_vec, 3);
            L[j][i] = (m[j][i] - sum) / U[i][i];
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
