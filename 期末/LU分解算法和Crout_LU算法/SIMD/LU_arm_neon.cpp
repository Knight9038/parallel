#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <arm_neon.h>
using namespace std;

int N = 64;

vector<vector<float>> m(N, vector<float>(N, 0.0));

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
            m[i][j] = rand() / (float)RAND_MAX;
        }
    }

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

    for (int i = 0; i < n; i++) {
        lower[i][i] = 1.0;

        for (int j = i; j < n; j++) {
            float32x4_t sum = vdupq_n_f32(0.0);
            for (int k = 0; k < i; k+=4) {
                float32x4_t lower_v = vld1q_f32(&lower[i][k]);
                float32x4_t upper_v = vld1q_f32(&upper[k][j]);
                sum = vmlaq_f32(sum, lower_v, upper_v);
            }
            float sum_arr[4];
            vst1q_f32(sum_arr, sum);
            upper[i][j] = mat[i][j] - (sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3]);
        }

        for (int j = i + 1; j < n; j++) {
            float32x4_t sum = vdupq_n_f32(0.0);
            for (int k = 0; k < i; k+=4) {
                float32x4_t lower_v = vld1q_f32(&lower[j][k]);
                float32x4_t upper_v = vld1q_f32(&upper[k][i]);
                sum = vmlaq_f32(sum, lower_v, upper_v);
            }
            float sum_arr[4];
            vst1q_f32(sum_arr, sum);
            lower[j][i] = (mat[j][i] - (sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3])) / upper[i][i];
        }
    }
}

int main() {
    m_reset();
    int iterations = 100;
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
    cout << "NEON: " << count << "ms" << endl;
    return 0;
}