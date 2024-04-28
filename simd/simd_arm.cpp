#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <arm_neon.h>
using namespace std;
#define N 1024 

float m[N][N];
float b[N];
float x[N];
float m_pivot[N][N];
float b_pivot[N];
float x_pivot[N];;
float m_neon[N][N];
float b_neon[N];
float x_neon[N];
float m_pivot_neon[N][N];
float b_pivot_neon[N];
float x_pivot_neon[N];

void m_reset() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m[i][j] = 0;
            if (i == j)
                m[i][j] = 1.0;
        }
    }

    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            m[i][j] = rand() % 21;

    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                if (m[i][j] < 21)
                    m[i][j] += m[k][j];

    for (int i = 0; i < N; i++)
        b[i] = rand() % 21;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m_pivot[i][j] = m[i][j];
            m_neon[i][j] = m[i][j];
            m_pivot_neon[i][j] = m[i][j];
        }
    }
    for (int i = 0; i < N; i++) {
        b_pivot[i] = b[i];
        b_neon[i] = b[i];
        b_pivot_neon[i] = b[i];
    }
}

void gaussian_elimination() {

    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            float factor = m[i][k] / m[k][k];
            for (int j = k; j < N; j++) {
                m[i][j] -= factor * m[k][j];
            }
            b[i] -= factor * b[k];
        }
    }


    x[N - 1] = b[N - 1] / m[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m[i][j] * x[j];
        }
        x[i] = sum / m[i][i];
    }
}

void gaussian_elimination_neon() {
    for (int k = 0; k < N; k++) {

        float32x4_t mk = vdupq_n_f32(m_neon[k][k]);
        for (int i = k + 1; i < N; i++) {
            float factor = m_neon[i][k] / m_neon[k][k];
            float32x4_t factor_vec = vdupq_n_f32(factor);
            for (int j = k; j < N; j += 4) { 
                float32x4_t mi = vld1q_f32(&m_neon[i][j]);
                float32x4_t mkj = vld1q_f32(&m_neon[k][j]);
                float32x4_t result = vsubq_f32(mi, vmulq_f32(factor_vec, mkj));
                vst1q_f32(&m_neon[i][j], result);
            }
            b_neon[i] -= factor * b_neon[k];
        }
    }

    x_neon[N - 1] = b_neon[N - 1] / m_neon[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_neon[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m_neon[i][j] * x_neon[j];
        }
        x_neon[i] = sum / m_neon[i][i];
    }
}

void gaussian_elimination_with_pivot() {
    for (int k = 0; k < N; k++) {
        int max_index = k;
        float max_value = fabs(m_pivot[k][k]);
        for (int i = k + 1; i < N; i++) {
            if (fabs(m_pivot[i][k]) > max_value) {
                max_value = fabs(m_pivot[i][k]);
                max_index = i;
            }
        }

        if (max_index != k) {
            for (int j = k; j < N; j++) {
                swap(m_pivot[k][j], m_pivot[max_index][j]);
            }
            swap(b_pivot[k], b_pivot[max_index]);
        }

        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot[i][k] / m_pivot[k][k];
            for (int j = k; j < N; j++) {
                m_pivot[i][j] -= factor * m_pivot[k][j];
            }
            b_pivot[i] -= factor * b_pivot[k];
        }
    }

    x_pivot[N - 1] = b_pivot[N - 1] / m_pivot[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_pivot[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m_pivot[i][j] * x_pivot[j];
        }
        x_pivot[i] = sum / m_pivot[i][i];
    }
}


void gaussian_elimination_with_pivot_neon() {
    for (int k = 0; k < N; k++) {
        int max_index = k;
        float32x4_t max_value_vec = vabsq_f32(vld1q_f32(&m_pivot_neon[k][k]));
        for (int i = k + 1; i < N; i++) {
            float32x4_t m_pivot_neon_i_k_vec = vld1q_f32(&m_pivot_neon[i][k]);
            float32x4_t abs_m_pivot_neon_i_k_vec = vabsq_f32(m_pivot_neon_i_k_vec);
            uint32x4_t cmp_result = vcgtq_f32(abs_m_pivot_neon_i_k_vec, max_value_vec);
            if (vgetq_lane_u32(cmp_result, 0) || vgetq_lane_u32(cmp_result, 1) ||
                vgetq_lane_u32(cmp_result, 2) || vgetq_lane_u32(cmp_result, 3)) {
                max_value_vec = abs_m_pivot_neon_i_k_vec;
                max_index = i;
            }
        }

        if (max_index != k) {
            for (int j = k; j < N; j += 4) {
                float32x4_t tmp_vec = vld1q_f32(&m_pivot_neon[k][j]);
                vst1q_f32(&m_pivot_neon[k][j], vld1q_f32(&m_pivot_neon[max_index][j]));
                vst1q_f32(&m_pivot_neon[max_index][j], tmp_vec);
            }
            std::swap(b_pivot_neon[k], b_pivot_neon[max_index]);
        }

        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_neon[i][k] / m_pivot_neon[k][k];
            float32x4_t factor_vec = vdupq_n_f32(factor);
            for (int j = k; j < N; j += 4) {
                float32x4_t m_pivot_neon_i_j_vec = vld1q_f32(&m_pivot_neon[i][j]);
                float32x4_t m_pivot_neon_k_j_vec = vld1q_f32(&m_pivot_neon[k][j]);
                float32x4_t result_vec = vmlsq_f32(m_pivot_neon_i_j_vec, factor_vec, m_pivot_neon_k_j_vec);
                vst1q_f32(&m_pivot_neon[i][j], result_vec);
            }
            b_pivot_neon[i] -= factor * b_pivot_neon[k];
        }
    }

    x_pivot_neon[N - 1] = b_pivot_neon[N - 1] / m_pivot_neon[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_pivot_neon[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m_pivot_neon[i][j] * x_pivot_neon[j];
        }
        x_pivot_neon[i] = sum / m_pivot_neon[i][i];
    }
}

int main() {
    m_reset();

    auto start = chrono::high_resolution_clock::now();
    gaussian_elimination();
    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration = end - start;
    float execution_guass_time = duration.count();
    cout << "Execution GUASS time: " << execution_guass_time << " ms" << endl;

    auto start_neon = chrono::high_resolution_clock::now();
    gaussian_elimination_neon();
    auto end_neon = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_neon = end_neon - start_neon;
    float execution_guass_time_neon = duration_neon.count();
    cout << "Execution GUASS NEON time: " << execution_guass_time_neon << " ms" << endl;

    auto start_pivot = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot();
    auto end_pivot = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot = end_pivot - start_pivot;
    float execution_guass_time_pivot = duration_pivot.count();
    cout << "Execution GUASS PIVOT time: " << execution_guass_time_pivot << " ms" << endl;

    auto start_pivot_neon = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_neon();
    auto end_pivot_neon = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_neon = end_pivot_neon - start_pivot_neon;
    float execution_guass_time_pivot_neon = duration_pivot_neon.count();
    cout << "Execution GUASS PIVOT NEON time: " << execution_guass_time_pivot_neon << " ms" << endl;

    return 0;
}