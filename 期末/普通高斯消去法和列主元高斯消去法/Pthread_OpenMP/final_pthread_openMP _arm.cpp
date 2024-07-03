#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <arm_neon.h>
#include <malloc.h>
#include <pthread.h>
#include <thread>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <cstdlib> 
#include <semaphore.h>

using namespace std;
#define N 16 
#define NUM_THREADS 4

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

float m_pthread_s[N][N];
float b_pthread_s[N];
float x_pthread_s[N];
float m_pthread_neon[N][N];
float b_pthread_neon[N];
float x_pthread_neon[N];
float m_pivot_pthread_s[N][N];
float b_pivot_pthread_s[N];
float x_pivot_pthread_s[N];
float m_pivot_pthread_neon[N][N];
float b_pivot_pthread_neon[N];
float x_pivot_pthread_neon[N];

float m_mp[N][N];
float b_mp[N];
float x_mp[N];
float m_pivot_mp[N][N];
float b_pivot_mp[N];
float x_pivot_mp[N];
float m_mp_neon[N][N];
float b_mp_neon[N];
float x_mp_neon[N];
float m_pivot_mp_neon[N][N];
float b_pivot_mp_neon[N];
float x_pivot_mp_neon[N];

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

            m_pthread_s[i][j] = m[i][j];
            m_pivot_pthread_s[i][j] = m[i][j];
            m_pthread_neon[i][j] = m[i][j];
            m_pivot_pthread_neon[i][j] = m[i][j];

            m_mp[i][j] = m[i][j];
            m_pivot_mp[i][j] = m[i][j];
            m_mp_neon[i][j] = m[i][j];
            m_pivot_mp_neon[i][j] = m[i][j];
        }
    }
    for (int i = 0; i < N; i++) {
        b_pivot[i] = b[i];
        b_neon[i] = b[i];
        b_pivot_neon[i] = b[i];

        b_pthread_s[i] = b[i];
        b_pivot_pthread_s[i] = b[i];
        b_pthread_neon[i] = b[i];
        b_pivot_pthread_neon[i] = b[i];

        b_mp[i] = b[i];
        b_pivot_mp[i] = b[i];
        b_mp_neon[i] = b[i];
        b_pivot_mp_neon[i] = b[i];
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


typedef struct {
    int t_id; 
} threadParam_t_s;

sem_t sem_leader;
sem_t sem_Division[NUM_THREADS - 1];
sem_t sem_Elimination[NUM_THREADS - 1];

void* threadFunc_s(void* param) {
    threadParam_t_s* p = (threadParam_t_s*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
      
        if (t_id == 0) {
            for (int j = k + 1; j < N; ++j) {
                m_pthread_s[k][j] = m_pthread_s[k][j] / m_pthread_s[k][k];
            }
            m_pthread_s[k][k] = 1.0;
        }
        else {
            sem_wait(&sem_Division[t_id - 1]); 
        }

       
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Division[i]);
            }
        }

      
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            for (int j = k + 1; j < N; ++j) {
                m_pthread_s[i][j] -= m_pthread_s[i][k] * m_pthread_s[k][j];
            }
            m_pthread_s[i][k] = 0.0;
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Elimination[i]); 
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]); 
        }
    }

    pthread_exit(NULL);
    return NULL;
}
void gaussian_elimination_pthread_s() {
   
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }

   
    pthread_t handles[NUM_THREADS];
    threadParam_t_s params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_s, (void*)&params[t_id]);
    }

   
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

    sem_destroy(&sem_leader);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Division[i]);
        sem_destroy(&sem_Elimination[i]);
    }

    
    for (int i = N - 1; i >= 0; --i) {
        x_pthread_s[i] = b_pthread_s[i];
        for (int j = i + 1; j < N; ++j) {
            x_pthread_s[i] -= m_pthread_s[i][j] * x_pthread_s[j];
        }
        x_pthread_s[i] /= m_pthread_s[i][i];
    }

}

typedef struct {
    int t_id; 
} threadParam_t_neon;

sem_t sem_leader_neon;
sem_t sem_Division_neon[NUM_THREADS - 1];
sem_t sem_Elimination_neon[NUM_THREADS - 1];

void* threadFunc_neon(void* param) {
    threadParam_t_neon* p = (threadParam_t_neon*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {
            for (int j = k + 1; j < N; ++j) {
                m_pthread_neon[k][j] = m_pthread_neon[k][j] / m_pthread_neon[k][k];
            }
            m_pthread_neon[k][k] = 1.0;
        }
        else {
            sem_wait(&sem_Division_neon[t_id - 1]);
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Division_neon[i]);
            }
        }

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            float32x4_t factor_vec = vdupq_n_f32(m_pthread_neon[i][k] / m_pthread_neon[k][k]);

            for (int j = k; j < N; j += 4) {
                float32x4_t row_i = vld1q_f32(&m_pthread_neon[i][j]);
                float32x4_t row_k = vld1q_f32(&m_pthread_neon[k][j]);
                float32x4_t temp = vmulq_f32(factor_vec, row_k);
                row_i = vmlsq_f32(row_i, temp, row_i);
                vst1q_f32(&m_pthread_neon[i][j], row_i);
            }
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leader_neon);
            }
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Elimination_neon[i]);
            }
        }
        else {
            sem_post(&sem_leader_neon);
            sem_wait(&sem_Elimination_neon[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return NULL;
}

void gaussian_elimination_pthread_neon() {
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Division_neon[i], 0, 0);
        sem_init(&sem_Elimination_neon[i], 0, 0);
    }

    pthread_t handles[NUM_THREADS];
    threadParam_t_neon params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_neon, (void*)&params[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

    sem_destroy(&sem_leader_neon);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Division_neon[i]);
        sem_destroy(&sem_Elimination_neon[i]);
    }

    for (int i = N - 1; i >= 0; --i) {
        x_pthread_neon[i] = b_pthread_neon[i];
        for (int j = i + 1; j < N; ++j) {
            x_pthread_neon[i] -= m_pthread_neon[i][j] * x_pthread_neon[j];
        }
        x_pthread_neon[i] /= m_pthread_neon[i][i];
    }
}

void gaussian_elimination_mp() {
#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            float factor = m_mp[i][k] / m_mp[k][k];
#pragma omp parallel for num_threads(NUM_THREADS)
            for (int j = k; j < N; j++) {
                m_mp[i][j] -= factor * m_mp[k][j];
                
            }
            b_mp[i] -= factor * b_mp[k];
        }
    }

   
    x_mp[N - 1] = b_mp[N - 1] / m_mp[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_mp[i];
#pragma omp parallel for reduction(- : sum) num_threads(NUM_THREADS)
        for (int j = i + 1; j < N; j++) {
            sum -= m_mp[i][j] * x[j];
        }
        x_mp[i] = sum / m_mp[i][i];
    }
}

void gaussian_elimination_mp_neon() {
    
#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            float factor = m_mp_neon[i][k] / m_mp_neon[k][k];

           
#pragma omp simd
            for (int j = k; j < N; j++) {
                m_mp_neon[i][j] -= factor * m_mp_neon[k][j];
            }
            b_mp_neon[i] -= factor * b_mp_neon[k];
        }
    }

    
    x_mp_neon[N - 1] = b_mp_neon[N - 1] / m_mp_neon[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_mp_neon[i];

       
#pragma omp simd
        for (int j = i + 1; j < N; j++) {
            sum -= m_mp_neon[i][j] * x[j];
        }
        x_mp_neon[i] = sum / m_mp_neon[i][i];
    }
}



void gaussian_elimination_with_pivot_mp() {
  
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++) {
      
        int max_index = k;
        float max_value = fabs(m_pivot_mp[k][k]);
#pragma omp simd
        for (int i = k + 1; i < N; i++) {
            if (fabs(m_pivot_mp[i][k]) > max_value) {
                max_value = fabs(m_pivot_mp[i][k]);
                max_index = i;
            }
        }

       
        if (max_index != k) {
#pragma omp critical
            {
                for (int j = k; j < N; j++) {
                    swap(m_pivot_mp[k][j], m_pivot_mp[max_index][j]);
                }
                swap(b_pivot_mp[k], b_pivot_mp[max_index]);
            }
        }

      
#pragma omp simd
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_mp[i][k] / m_pivot_mp[k][k];
            for (int j = k; j < N; j++) {
                m_pivot_mp[i][j] -= factor * m_pivot_mp[k][j];
            }
            b_pivot_mp[i] -= factor * b_pivot_mp[k];
        }
    }

 
    x_pivot_mp[N - 1] = b_pivot_mp[N - 1] / m_pivot_mp[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_pivot_mp[i];
#pragma omp simd
        for (int j = i + 1; j < N; j++) {
            sum -= m_pivot_mp[i][j] * x_pivot_mp[j];
        }
        x_pivot_mp[i] = sum / m_pivot_mp[i][i];
    }
}

void gaussian_elimination_with_pivot_mp_neon() {
 
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++) {
       
        int max_index = k;
        float max_value = fabs(m_pivot_mp_neon[k][k]);

       
#pragma omp simd reduction(max:max_value)
        for (int i = k + 1; i < N; i++) {
            max_value = fmaxf(max_value, fabs(m_pivot_mp_neon[i][k]));
        }

       
#pragma omp simd reduction(min:max_value)
        for (int i = k + 1; i < N; i++) {
            if (fabs(m_pivot_mp_neon[i][k]) == max_value) {
                max_index = i;
                break;
            }
        }

       
        if (max_index != k) {
#pragma omp critical
            {
                for (int j = k; j < N; j++) {
                    float tmp = m_pivot_mp_neon[k][j];
                    m_pivot_mp_neon[k][j] = m_pivot_mp_neon[max_index][j];
                    m_pivot_mp_neon[max_index][j] = tmp;
                }
                float tmp_b = b_pivot_mp_neon[k];
                b_pivot_mp_neon[k] = b_pivot_mp_neon[max_index];
                b_pivot_mp_neon[max_index] = tmp_b;
            }
        }

      
#pragma omp simd
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_mp_neon[i][k] / m_pivot_mp_neon[k][k];
            float32x4_t factor_vec = vdupq_n_f32(factor);
            float32x4_t* m_row_i = (float32x4_t*)(&m_pivot_mp_neon[i][k]);
            float32x4_t* m_row_k = (float32x4_t*)(&m_pivot_mp_neon[k][k]);

#pragma unroll(4)
            for (int j = 0; j < (N - k) / 4; j++) {
                m_row_i[j] = vmlsq_f32(m_row_i[j], factor_vec, m_row_k[j]);
            }

            b_pivot_mp_neon[i] -= factor * b_pivot_mp_neon[k];
        }
    }

    
    x_pivot_mp_neon[N - 1] = b_pivot_mp_neon[N - 1] / m_pivot_mp_neon[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_pivot_mp_neon[i];

#pragma omp simd
        for (int j = i + 1; j < N; j++) {
            sum -= m_pivot_mp_neon[i][j] * x_pivot_mp_neon[j];
        }
        x_pivot_mp_neon[i] = sum / m_pivot_mp_neon[i][i];
    }
}


typedef struct {
    int t_id; 
} threadParam_pivot_pthread_s;


sem_t sem_leader_pivot_pthread_s;
sem_t sem_Division_pivot_pthread_s[NUM_THREADS - 1];
sem_t sem_Elimination_pivot_pthread_s[NUM_THREADS - 1];
void* threadFunc_pivot_pthread_s(void* param) {
    threadParam_pivot_pthread_s* p = (threadParam_pivot_pthread_s*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {
            
            for (int i = k + 1; i < N; ++i) {
                m_pivot_pthread_s[i][k] = m_pivot_pthread_s[i][k] / m_pivot_pthread_s[k][k];
            }
            m_pivot_pthread_s[k][k] = 1.0;
        }
        else {
           
            sem_wait(&sem_Division_pivot_pthread_s[t_id - 1]);
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Division_pivot_pthread_s[i]);
            }
        }

       
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            for (int j = k + 1; j < N; ++j) {
                m_pivot_pthread_s[i][j] -= m_pivot_pthread_s[i][k] * m_pivot_pthread_s[k][j];
            }
            m_pivot_pthread_s[i][k] = 0.0;
        }

        if (t_id == 0) {
           
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leader_pivot_pthread_s);
            }
           
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Elimination_pivot_pthread_s[i]);
            }
        }
        else {
            sem_post(&sem_leader_pivot_pthread_s);
            sem_wait(&sem_Elimination_pivot_pthread_s[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return NULL;
}

void gaussian_elimination_with_pivot_pthread_s() {
   
    sem_init(&sem_leader_pivot_pthread_s, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Division_pivot_pthread_s[i], 0, 0);
        sem_init(&sem_Elimination_pivot_pthread_s[i], 0, 0);
    }

  
    pthread_t handles[NUM_THREADS];
    threadParam_pivot_pthread_s params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_pivot_pthread_s, (void*)&params[t_id]);
    }

   
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

   
    sem_destroy(&sem_leader_pivot_pthread_s);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Division_pivot_pthread_s[i]);
        sem_destroy(&sem_Elimination_pivot_pthread_s[i]);
    }

   
    for (int i = N - 1; i >= 0; --i) {
        x_pivot_pthread_s[i] = b_pivot_pthread_s[i]; 
        for (int j = i + 1; j < N; ++j) {
            x_pivot_pthread_s[i] -= m_pivot_pthread_s[i][j] * x_pivot_pthread_s[j]; 
        }
        x_pivot_pthread_s[i] /= m_pivot_pthread_s[i][i]; 
    }
}



typedef struct {
    int t_id; 
} threadParam_pivot_pthread_neon;

sem_t sem_leader_pivot_pthread_neon;
sem_t sem_Division_pivot_pthread_neon[NUM_THREADS - 1];
sem_t sem_Elimination_pivot_pthread_neon[NUM_THREADS - 1];

void* threadFunc_pivot_pthread_neon(void* param) {
    threadParam_pivot_pthread_neon* p = (threadParam_pivot_pthread_neon*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {
            for (int i = k + 1; i < N; ++i) {
                m_pivot_pthread_neon[i][k] = m_pivot_pthread_neon[i][k] / m_pivot_pthread_neon[k][k];
            }
            m_pivot_pthread_neon[k][k] = 1.0;
        }
        else {
            sem_wait(&sem_Division_pivot_pthread_neon[t_id - 1]);
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Division_pivot_pthread_neon[i]);
            }
        }

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            if (m_pivot_pthread_neon[k][k] == 0.0) {
                continue;
            }
            float32x4_t factor = vdupq_n_f32(m_pivot_pthread_neon[i][k] / m_pivot_pthread_neon[k][k]);
            for (int j = k + 1; j < N; j += 4) {
                float32x4_t m_row = vld1q_f32(&m_pivot_pthread_neon[i][j]);
                float32x4_t mk_mul_mrow = vmulq_f32(factor, m_row);
                float32x4_t m_sub_result = vsubq_f32(vld1q_f32(&m_pivot_pthread_neon[i][j]), mk_mul_mrow);
                vst1q_f32(&m_pivot_pthread_neon[i][j], m_sub_result);
            }
            m_pivot_pthread_neon[i][k] = 0.0;
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leader_pivot_pthread_neon);
            }
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Elimination_pivot_pthread_neon[i]);
            }
        }
        else {
            sem_post(&sem_leader_pivot_pthread_neon);
            sem_wait(&sem_Elimination_pivot_pthread_neon[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return NULL;
}

void gaussian_elimination_with_pivot_pthread_neon() {
    sem_init(&sem_leader_pivot_pthread_neon, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Division_pivot_pthread_neon[i], 0, 0);
        sem_init(&sem_Elimination_pivot_pthread_neon[i], 0, 0);
    }

    pthread_t handles[NUM_THREADS];
    threadParam_pivot_pthread_neon params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_pivot_pthread_neon, (void*)&params[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

    sem_destroy(&sem_leader_pivot_pthread_neon);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Division_pivot_pthread_neon[i]);
        sem_destroy(&sem_Elimination_pivot_pthread_neon[i]);
    }

    for (int i = N - 1; i >= 0; --i) {
        x_pivot_pthread_neon[i] = b_pivot_pthread_neon[i];
        for (int j = i + 1; j < N; ++j) {
            x_pivot_pthread_neon[i] -= m_pivot_pthread_neon[i][j] * x_pivot_pthread_neon[j];
        }
        x_pivot_pthread_neon[i] /= m_pivot_pthread_neon[i][i];
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

    auto start_pthread_s = chrono::high_resolution_clock::now();
    gaussian_elimination_pthread_s();
    auto end_pthread_s = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pthread_s = end_pthread_s - start_pthread_s;
    float execution_guass_time_pthread_s = duration_pthread_s.count();
    cout << "Execution GUASS Pthread time: " << execution_guass_time_pthread_s << " ms" << endl;

    auto start_pthread_neon = chrono::high_resolution_clock::now();
    gaussian_elimination_pthread_neon();
    auto end_pthread_neon = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pthread_neon = end_pthread_neon - start_pthread_neon;
    float execution_guass_time_pthread_neon = duration_pthread_neon.count();
    cout << "Execution GUASS Pthread NEON time: " << execution_guass_time_pthread_neon << " ms" << endl;

    auto start_mp = chrono::high_resolution_clock::now();
    gaussian_elimination_mp();
    auto end_mp = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_mp = end_mp - start_mp;
    float execution_guass_time_mp = duration_mp.count();
    cout << "Execution GUASS MP time: " << execution_guass_time_mp << " ms" << endl;

    auto start_mp_neon = chrono::high_resolution_clock::now();
    gaussian_elimination_mp_neon();
    auto end_mp_neon = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_mp_neon = end_mp_neon - start_mp_neon;
    float execution_guass_time_mp_neon = duration_mp_neon.count();
    cout << "Execution GUASS MP NEON time: " << execution_guass_time_mp_neon << " ms" << endl;




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

    auto start_pivot_pthread_s = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_pthread_s();
    auto end_pivot_pthread_s = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_pthread_s = end_pivot_pthread_s - start_pivot_pthread_s;
    float execution_guass_time_pivot_pthread_s = duration_pivot_pthread_s.count();
    cout << "Execution GUASS PIVOT Pthread time: " << execution_guass_time_pivot_pthread_s << " ms" << endl;

    auto start_pivot_pthread_neon = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_pthread_neon();
    auto end_pivot_pthread_neon = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_pthread_neon = end_pivot_pthread_neon - start_pivot_pthread_neon;
    float execution_guass_time_pivot_pthread_neon = duration_pivot_pthread_neon.count();
    cout << "Execution GUASS PIVOT NEON Pthread time: " << execution_guass_time_pivot_pthread_neon << " ms" << endl;

    auto start_pivot_mp = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_mp();
    auto end_pivot_mp = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_mp = end_pivot_mp - start_pivot_mp;
    float execution_guass_time_pivot_mp = duration_pivot_mp.count();
    cout << "Execution GUASS PIVOT MP time: " << execution_guass_time_pivot_mp << " ms" << endl;

    auto start_pivot_mp_neon = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_mp_neon();
    auto end_pivot_mp_neon = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_mp_neon = end_pivot_mp_neon - start_pivot_mp_neon;
    float execution_guass_time_pivot_mp_neon = duration_pivot_mp_neon.count();
    cout << "Execution GUASS PIVOT MP NEON time: " << execution_guass_time_pivot_mp_neon << " ms" << endl;

    return 0;
}