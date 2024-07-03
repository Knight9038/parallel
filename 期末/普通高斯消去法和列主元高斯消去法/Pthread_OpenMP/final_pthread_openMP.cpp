#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <immintrin.h> // 包含 SSE/AVX 指令集的头文件
#include <stddef.h> // For size_t
#include <stdlib.h>
#include <malloc.h>
#include <pthread.h>
#include <thread>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <cstdlib> // 用于 malloc 和 free
#include <semaphore.h>
using namespace std;
#define N 1280// 矩阵的大小
#define NUM_THREADS 8// 假设使用线程数
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
// 定义线程参数结构


float m[N][N];
float b[N];
float x[N];
float m_pthread[N][N];
float b_pthread[N];
float x_pthread[N];
float m_pthread_hor[N][N];
float b_pthread_hor[N];
float x_pthread_hor[N];
float m_pthread_s[N][N];
float b_pthread_s[N];
float x_pthread_s[N];
float m_pthread_s_hor[N][N];
float b_pthread_s_hor[N];
float x_pthread_s_hor[N];
float m_sse_vec_all[N][N];
float b_sse_vec_all[N];
float x_sse_vec_all[N];
float m_sse_vec_pthread[N][N];
float b_sse_vec_pthread[N];
float x_sse_vec_pthread[N];
float m_avx_vec_all[N][N];
float b_avx_vec_all[N];
float x_avx_vec_all[N];
float m_avx_vec_pthread[N][N];
float b_avx_vec_pthread[N];
float x_avx_vec_pthread[N];
float m_mp[N][N];
float b_mp[N];
float x_mp[N];
float m_mp_sse[N][N];
float b_mp_sse[N];
float x_mp_sse[N];
float m_mp_avx[N][N];
float b_mp_avx[N];
float x_mp_avx[N];

float m_pivot[N][N];
float b_pivot[N];
float x_pivot[N];
float m_pivot_pthread[N][N];
float b_pivot_pthread[N];
float x_pivot_pthread[N];
float m_pivot_pthread_ver[N][N];
float b_pivot_pthread_ver[N];
float x_pivot_pthread_ver[N];
float m_pivot_pthread_s[N][N];
float b_pivot_pthread_s[N];
float x_pivot_pthread_s[N];
float m_pivot_pthread_s_ver[N][N];
float b_pivot_pthread_s_ver[N];
float x_pivot_pthread_s_ver[N];
float m_pivot_sse_vec_all[N][N];
float b_pivot_sse_vec_all[N];
float x_pivot_sse_vec_all[N];
float m_pivot_sse_vec_pthread[N][N];
float b_pivot_sse_vec_pthread[N];
float x_pivot_sse_vec_pthread[N];
float m_pivot_avx_vec_all[N][N];
float b_pivot_avx_vec_all[N];
float x_pivot_avx_vec_all[N];
float m_pivot_avx_vec_pthread[N][N];
float b_pivot_avx_vec_pthread[N];
float x_pivot_avx_vec_pthread[N];
float m_pivot_mp[N][N];
float b_pivot_mp[N];
float x_pivot_mp[N];
float m_pivot_mp_sse[N][N];
float b_pivot_mp_sse[N];
float x_pivot_mp_sse[N];
float m_pivot_mp_avx[N][N];
float b_pivot_mp_avx[N];
float x_pivot_mp_avx[N];

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

    for (int i = 0; i < N; i++)//初始化b[N]
        b[i] = rand() % 21;
    //保证矩阵同样
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m_pthread[i][j] = m[i][j];
            m_pthread_hor[i][j] = m[i][j];
            m_pthread_s[i][j] = m[i][j];
            m_pthread_s_hor[i][j] = m[i][j];
            m_sse_vec_all[i][j] = m[i][j];
            m_sse_vec_pthread[i][j] = m[i][j];
            m_mp[i][j] = m[i][j];
            m_mp_sse[i][j] = m[i][j];
            m_mp_avx[i][j] = m[i][j];
            m_pivot_mp[i][j] = m[i][j];
            m_avx_vec_all[i][j] = m[i][j];
            m_avx_vec_pthread[i][j] = m[i][j];
            m_pivot[i][j] = m[i][j];
            m_pivot_pthread[i][j] = m[i][j];
            m_pivot_pthread_ver[i][j] = m[i][j];
            m_pivot_sse_vec_all[i][j] = m[i][j];
            m_pivot_sse_vec_pthread[i][j] = m[i][j];
            m_pivot_pthread_s[i][j] = m[i][j];
            m_pivot_pthread_s_ver[i][j] = m[i][j];
            m_pivot_mp_sse[i][j] = m[i][j];
            m_pivot_mp_avx[i][j] = m[i][j];
            m_pivot_avx_vec_all[i][j] = m[i][j];
            m_pivot_avx_vec_pthread[i][j] = m[i][j];
           
        }
    }
    for (int i = 0; i < N; i++) {
        b_pthread[i] = b[i];
        b_pthread_hor[i] = b[i];
        b_pthread_s[i] = b[i];
        b_pthread_s_hor[i] = b[i];
        b_sse_vec_all[i] = b[i];
        b_sse_vec_pthread[i] = b[i];
        b_avx_vec_all[i] = b[i];
        b_avx_vec_pthread[i] = b[i];
        b_pivot[i] = b[i];
        b_pivot_mp[i] = b[i];
        b_mp[i] = b[i];
        b_mp_sse[i] = b[i];
        b_mp_avx[i] = b[i];
        b_pivot_pthread[i] = b[i];
        b_pivot_pthread_ver[i] = b[i];
        b_pivot_sse_vec_all[i] = b[i];
        b_pivot_sse_vec_pthread[i] = b[i];
        b_pivot_pthread_s[i] = b[i];
        b_pivot_pthread_s_ver[i] = b[i];
        b_pivot_mp_sse[i] = b[i];
        b_pivot_mp_avx[i] = b[i];
        b_pivot_avx_vec_all[i] = b[i];
        b_pivot_avx_vec_pthread[i] = b[i];
       
    }
}

void gaussian_elimination() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            float factor = m[i][k] / m[k][k];
            for (int j = k; j < N; j++) {
                m[i][j] -= factor * m[k][j];
                //if (fabs(m[i][j]) < 1e-6)
                //    m[i][j] = 0; // 小于 10^-6 的数值赋值为 0
            }
            b[i] -= factor * b[k];
        }
    }

    // 回代过程
    x[N - 1] = b[N - 1] / m[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m[i][j] * x[j];
        }
        x[i] = sum / m[i][i];
    }
}

typedef struct {
    int k;      // 消去的轮次
    int tid;    // 线程 id
} threadParam_t_hor;

// 高斯消元的线程函数
void threadFunc_hor(threadParam_t_hor* param) {
    int k = param->k;
    int tid = param->tid;
    for (int i = k + 1 + tid; i < N; i += NUM_THREADS) {
        float factor = m_pthread_hor[i][k] / m_pthread_hor[k][k];
        for (int j = k; j < N; ++j) {
            m_pthread_hor[i][j] -= factor * m_pthread_hor[k][j];
        }
        b_pthread_hor[i] -= factor * b_pthread_hor[k];
    }
    free(param); // 在线程函数结束后释放内存
}

void gaussian_elimination_pthread_hor() {
    // 主线程进行高斯消元
    for (int k = 0; k < N; ++k) {
        // 创建工作线程进行消元操作
        vector<thread> handles;
        for (int tid = 0; tid < NUM_THREADS; tid++) {
            threadParam_t_hor* param = (threadParam_t_hor*)malloc(sizeof(threadParam_t_hor));
            param->k = k;
            param->tid = tid;
            handles.push_back(thread(threadFunc_hor, param));
        }

        // 主线程等待所有工作线程完成此轮消去工作
        for (auto& handle : handles) {
            handle.join();
        }
    }
    
    // 回代过程
    for (int i = N - 1; i >= 0; --i) {
        x_pthread_hor[i] = b_pthread_hor[i];
        for (int j = i + 1; j < N; ++j) {
            x_pthread_hor[i] -= m_pthread_hor[i][j] * x_pthread_hor[j];
        }
        x_pthread_hor[i] /= m_pthread_hor[i][i];
    }
}

typedef struct {
    int k;          // 消去的轮次
    int thread_id;  // 线程 id
} threadParam_t;
// 垂直划分的高斯消元线程函数
void threadFunc(threadParam_t* param) {
    int k = param->k;
    int thread_id = param->thread_id;

    int start_col = thread_id * (N /NUM_THREADS);
    int end_col = (thread_id + 1) * (N /NUM_THREADS);

    for (int j = start_col; j < end_col; ++j) {
        float factor = m_pthread[k][j] / m_pthread[k][k];
        for (int i = k + 1; i < N; ++i) {
            m_pthread[i][j] -= factor * m_pthread[i][k];
        }
        b_pthread[j] -= factor * b_pthread[k];
    }

    free(param); // 在线程函数结束后释放内存
}
void gaussian_elimination_pthread() {
    // 主线程进行高斯消元
    for (int k = 0; k < N; ++k) {
        // 创建工作线程进行消元操作
        vector<thread> handles;
        for (int tid = 0; tid < NUM_THREADS; tid++) {
            threadParam_t* param = (threadParam_t*)malloc(sizeof(threadParam_t));
            param->k = k;
            param->thread_id = tid;
            handles.push_back(thread(threadFunc, param));
        }

        // 主线程等待所有工作线程完成此轮消去工作
        for (auto& handle : handles) {
            handle.join();
        }
    }

    // 回代过程
    for (int i = N - 1; i >= 0; --i) {
        x_pthread[i] = b_pthread[i];
        for (int j = i + 1; j < N; ++j) {
            x_pthread[i] -= m_pthread[i][j] * x_pthread[j];
        }
        x_pthread[i] /= m_pthread[i][i];
    }
}

// 线程数据结构定义
typedef struct {
    int t_id; // 线程 id
} threadParam_t_s;

// 信号量定义
sem_t sem_leader;
sem_t sem_Division[NUM_THREADS - 1];
sem_t sem_Elimination[NUM_THREADS - 1];

// 线程函数定义
void* threadFunc_s(void* param) {
    threadParam_t_s* p = (threadParam_t_s*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        // t_id 为 0 的线程做除法操作，其它工作线程先等待
        if (t_id == 0) {
            for (int j = k + 1; j < N; ++j) {
                m_pthread_s[k][j] = m_pthread_s[k][j] / m_pthread_s[k][k];
            }
            m_pthread_s[k][k] = 1.0;
        }
        else {
            sem_wait(&sem_Division[t_id - 1]); // 阻塞，等待完成除法操作
        }

        // t_id 为 0 的线程唤醒其它工作线程，进行消去操作
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Division[i]);
            }
        }

        // 循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            for (int j = k + 1; j < N; ++j) {
                m_pthread_s[i][j] -= m_pthread_s[i][k] * m_pthread_s[k][j];
            }
            m_pthread_s[i][k] = 0.0;
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leader); // 等待其它 worker 完成消去
            }
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
            }
        }
        else {
            sem_post(&sem_leader); // 通知 leader，已完成消去任务
            sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
        }
    }

    pthread_exit(NULL);
    return NULL;
}
void gaussian_elimination_pthread_s(){
    // 初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }

    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_t_s params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_s, (void*)&params[t_id]);
    }

    // 主线程等待所有线程完成
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

    // 销毁所有信号量
    sem_destroy(&sem_leader);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Division[i]);
        sem_destroy(&sem_Elimination[i]);
    }

    // 回代求解过程
    for (int i = N - 1; i >= 0; --i) {
        x_pthread_s[i] = b_pthread_s[i]; // 初始化解向量的第 i 个元素为右侧常数向量的第 i 个元素
        for (int j = i + 1; j < N; ++j) {
            x_pthread_s[i] -= m_pthread_s[i][j] * x_pthread_s[j]; // 更新解向量的第 i 个元素
        }
        x_pthread_s[i] /= m_pthread_s[i][i]; // 计算解向量的第 i 个元素
    }

}



typedef struct {
    int t_id; // 线程 id
} threadParam_sse_vec_pthread;

// 信号量定义
sem_t sem_leader_sse_vec_pthread;
sem_t sem_Division_sse_vec_pthread[NUM_THREADS - 1];
sem_t sem_Elimination_sse_vec_pthread[NUM_THREADS - 1];
// 线程函数定义
void* threadFunc_sse_vec_pthread(void* param) {
    threadParam_sse_vec_pthread* p = (threadParam_sse_vec_pthread*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        // t_id 为 0 的线程做除法操作，其它工作线程先等待
        if (t_id == 0) {
            for (int j = k + 1; j < N; ++j) {
                if (m_sse_vec_pthread[k][k] != 0.0) {
                    m_sse_vec_pthread[k][j] = m_sse_vec_pthread[k][j] / m_sse_vec_pthread[k][k];
                }
                else {
                    //printf("Error: Division by zero in row %d, column %d.\n", k, k);
                    m_sse_vec_pthread[k][j] = 0.0; // 为避免NaN值，设定为0
                }
            }
            m_sse_vec_pthread[k][k] = 1.0;
        }
        else {
            sem_wait(&sem_Division_sse_vec_pthread[t_id - 1]); // 阻塞，等待完成除法操作
        }

        // t_id 为 0 的线程唤醒其它工作线程，进行消去操作
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Division_sse_vec_pthread[i]);
            }
        }

        // 循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            if (m_sse_vec_pthread[k][k] == 0.0) {
               // printf("Error: Division by zero in row %d, column %d.\n", k, k);
                continue; // 跳过当前操作
            }
            float factor = m_sse_vec_pthread[i][k] / m_sse_vec_pthread[k][k];
            for (int j = k + 1; j < N; ++j) {
                m_sse_vec_pthread[i][j] -= factor * m_sse_vec_pthread[k][j];
            }
            m_sse_vec_pthread[i][k] = 0.0;
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leader_sse_vec_pthread); // 等待其它 worker 完成消去
            }
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Elimination_sse_vec_pthread[i]); // 通知其它 worker 进入下一轮
            }
        }
        else {
            sem_post(&sem_leader_sse_vec_pthread); // 通知 leader，已完成消去任务
            sem_wait(&sem_Elimination_sse_vec_pthread[t_id - 1]); // 等待通知，进入下一轮
        }
    }

    pthread_exit(NULL);
    return NULL;
}

void gaussian_elimination_sse_vec_pthread() {
    // 初始化信号量
    sem_init(&sem_leader_sse_vec_pthread, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Division_sse_vec_pthread[i], 0, 0);
        sem_init(&sem_Elimination_sse_vec_pthread[i], 0, 0);
    }

    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_sse_vec_pthread params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_sse_vec_pthread, (void*)&params[t_id]);
    }

    // 主线程等待所有线程完成
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

    // 销毁所有信号量
    sem_destroy(&sem_leader_sse_vec_pthread);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Division_sse_vec_pthread[i]);
        sem_destroy(&sem_Elimination_sse_vec_pthread[i]);
    }

    // 回代求解过程
    for (int i = N - 1; i >= 0; --i) {
        x_sse_vec_pthread[i] = b_sse_vec_pthread[i]; // 初始化解向量的第 i 个元素为右侧常数向量的第 i 个元素
        for (int j = i + 1; j < N; ++j) {
            x_sse_vec_pthread[i] -= m_sse_vec_pthread[i][j] * x_sse_vec_pthread[j]; // 更新解向量的第 i 个元素
        }
        if (m_sse_vec_pthread[i][i] == 0.0) {
            // 处理除数为零的情况
            //printf("Error: Division by 0 in row %d, column %d during back substitution.\n", i, i);
            x_sse_vec_pthread[i] = 0.0; // 为避免NaN值，设定为0
        }
        else {
          x_sse_vec_pthread[i] /= m_sse_vec_pthread[i][i]; // 计算解向量的第 i 个元素
        }
    }
}




typedef struct {
    int t_id; // 线程 id
} threadParam_avx_vec_pthread;

// 信号量定义
sem_t sem_leader_avx_vec_pthread;
sem_t sem_Division_avx_vec_pthread[NUM_THREADS - 1];
sem_t sem_Elimination_avx_vec_pthread[NUM_THREADS - 1];

// 线程函数定义
void* threadFunc_avx_vec_pthread(void* param) {
    threadParam_avx_vec_pthread* p = (threadParam_avx_vec_pthread*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        // t_id 为 0 的线程做除法操作，其它工作线程先等待
        if (t_id == 0) {
            for (int j = k + 1; j < N; ++j) {
                if (m_avx_vec_pthread[k][k] != 0.0) {
                    m_avx_vec_pthread[k][j] = m_avx_vec_pthread[k][j] / m_avx_vec_pthread[k][k];
                }
                else {
                    //printf("Error: Division by zero in row %d, column %d.\n", k, k);
                    m_avx_vec_pthread[k][j] = 0.0; // 为避免NaN值，设定为0
                }
            }
            m_avx_vec_pthread[k][k] = 1.0;
        }
        else {
            sem_wait(&sem_Division_avx_vec_pthread[t_id - 1]); // 阻塞，等待完成除法操作
        }

        // t_id 为 0 的线程唤醒其它工作线程，进行消去操作
        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Division_avx_vec_pthread[i]);
            }
        }

        // 循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            if (m_avx_vec_pthread[k][k] == 0.0) {
               // printf("Error: Division by zero in row %d, column %d.\n", k, k);
                continue; // 跳过当前操作
            }
            float factor = m_avx_vec_pthread[i][k] / m_avx_vec_pthread[k][k];
            __m256 factor_avx = _mm256_set1_ps(factor); // 生成AVX向量
            for (int j = k + 1; j < N; j += 8) {
                __m256 m_row = _mm256_loadu_ps(&m_avx_vec_pthread[i][j]); // 使用未对齐加载
                __m256 m_k = _mm256_loadu_ps(&m_avx_vec_pthread[k][j]); // 使用未对齐加载
                __m256 m_mul = _mm256_mul_ps(factor_avx, m_k); // 执行乘法操作
                __m256 m_sub_result = _mm256_sub_ps(m_row, m_mul); // 执行减法操作
                _mm256_storeu_ps(&m_avx_vec_pthread[i][j], m_sub_result); // 使用未对齐存储
            }
            m_avx_vec_pthread[i][k] = 0.0;
        }

        if (t_id == 0) {
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leader_avx_vec_pthread); // 等待其它 worker 完成消去
            }
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Elimination_avx_vec_pthread[i]); // 通知其它 worker 进入下一轮
            }
        }
        else {
            sem_post(&sem_leader_avx_vec_pthread); // 通知 leader，已完成消去任务
            sem_wait(&sem_Elimination_avx_vec_pthread[t_id - 1]); // 等待通知，进入下一轮
        }
    }

    pthread_exit(NULL);
    return NULL;
}

void gaussian_elimination_avx_vec_pthread() {
    // 初始化信号量
    sem_init(&sem_leader_avx_vec_pthread, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Division_avx_vec_pthread[i], 0, 0);
        sem_init(&sem_Elimination_avx_vec_pthread[i], 0, 0);
    }

    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_avx_vec_pthread params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_avx_vec_pthread, (void*)&params[t_id]);
    }

    // 主线程等待所有线程完成
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

    // 销毁所有信号量
    sem_destroy(&sem_leader_avx_vec_pthread);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Division_avx_vec_pthread[i]);
        sem_destroy(&sem_Elimination_avx_vec_pthread[i]);
    }

    // 回代求解过程
    for (int i = N - 1; i >= 0; --i) {
        x_avx_vec_pthread[i] = b_avx_vec_pthread[i]; for (int j = i + 1; j < N; ++j) {
            x_avx_vec_pthread[i] -= m_avx_vec_pthread[i][j] * x_avx_vec_pthread[j]; // 更新解向量的第 i 个元素
        }
        if (m_avx_vec_pthread[i][i] == 0.0) {
            // 处理除数为零的情况
            //printf("Error: Division by 0 in row %d, column %d during back substitution.\n", i, i);
            x_avx_vec_pthread[i] = 0.0; // 为避免NaN值，设定为0
        }
        else {
            x_avx_vec_pthread[i] /= m_avx_vec_pthread[i][i]; // 计算解向量的第 i 个元素
        }
    }
}






// 线程数据结构定义
typedef struct {
    int t_id; // 线程 id
} threadParam_t_s_hor;

// 信号量定义
sem_t sem_leader_hor;
sem_t sem_Division_hor[NUM_THREADS - 1];
sem_t sem_Elimination_hor[NUM_THREADS - 1];

// 线程函数定义
void* threadFunc_s_hor(void* param) {
    threadParam_t_s_hor* p = (threadParam_t_s_hor*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {
            // 等待 t_id 为 0 的线程完成除法操作
            for (int i = k + 1; i < N; ++i) {
                m_pthread_s_hor[i][k] = m_pthread_s_hor[i][k] / m_pthread_s_hor[k][k];
            }
            m_pthread_s_hor[k][k] = 1.0;
        }
        else {
            // 其他线程等待除法操作完成
            sem_wait(&sem_Division_hor[t_id - 1]);
        }

        if (t_id == 0) {
            // 唤醒其他线程，开始消去操作
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Division_hor[i]);
            }
        }

        // 循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            for (int j = k + 1; j < N; ++j) {
                m_pthread_s_hor[i][j] -= m_pthread_s_hor[i][k] * m_pthread_s_hor[k][j];
            }
            m_pthread_s_hor[i][k] = 0.0;
        }

        if (t_id == 0) {
            // t_id 为 0 的线程等待其他线程完成消去操作
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leader_hor);
            }
            // t_id 为 0 的线程通知其他线程进入下一轮循环
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Elimination_hor[i]);
            }
        }
        else {
            sem_post(&sem_leader_hor);// 其他线程通知 t_id 为 0 的线程已完成消去任务
            // 其他线程等待通知，进入下一轮循环
            sem_wait(&sem_Elimination_hor[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return NULL;
}

void gaussian_elimination_pthread_s_hor() {
    // 初始化信号量
    sem_init(&sem_leader_hor, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Division_hor[i], 0, 0);
        sem_init(&sem_Elimination_hor[i], 0, 0);
    }

    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_t_s_hor params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_s_hor, (void*)&params[t_id]);
    }

    // 主线程等待所有线程完成
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

    // 销毁所有信号量
    sem_destroy(&sem_leader_hor);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Division_hor[i]);
        sem_destroy(&sem_Elimination_hor[i]);
    }

    // 回代求解过程
    for (int i = N - 1; i >= 0; --i) {
        x_pthread_s_hor[i] = b_pthread_s_hor[i]; // 初始化解向量的第 i 个元素为右侧常数向量的第 i 个元素
        for (int j = i + 1; j < N; ++j) {
            x_pthread_s_hor[i] -= m_pthread_s_hor[i][j] * x_pthread_s_hor[j]; // 更新解向量的第 i 个元素
        }
        x_pthread_s_hor[i] /= m_pthread_s_hor[i][i]; // 计算解向量的第 i 个元素
    }
}


void gaussian_elimination_sse_vec_all() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        __m128 mk = _mm_set1_ps(m_sse_vec_all[k][k]); // 将m[k][k]的值扩展到一个128位的寄存器mk中
        for (int i = k + 1; i < N; i++) {
            float factor = m_sse_vec_all[i][k] / m_sse_vec_all[k][k];
            __m128 factor_vec = _mm_set1_ps(factor); // 将factor的值扩展到一个128位的寄存器factor_vec中
            int j;
            for (j = k; j < N - 4; j += 4) { // 使用SSE指令，每次处理4个元素
                __m128 mi = _mm_loadu_ps(&m_sse_vec_all[i][j]); // 将m[i][j]~m[i][j+3]加载到一个128位的寄存器mi中
                __m128 mkj = _mm_loadu_ps(&m_sse_vec_all[k][j]); // 将m[k][j]~m[k][j+3]加载到一个128位的寄存器mkj中
                __m128 result = _mm_sub_ps(mi, _mm_mul_ps(factor_vec, mkj)); // 使用SSE指令进行减法和乘法
                _mm_storeu_ps(&m_sse_vec_all[i][j], result); // 将结果存回m[i][j]~m[i][j+3]
            }
            // 处理剩余的不足4个元素，使用串行方式
            for (; j < N; j++) {
                m_sse_vec_all[i][j] -= factor * m_sse_vec_all[k][j];
            }
            b_sse_vec_all[i] -= factor * b_sse_vec_all[k];
        }
    }

    // 回代过程
    x_sse_vec_all[N - 1] = b_sse_vec_all[N - 1] / m_sse_vec_all[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        __m128 sum = _mm_set1_ps(b_sse_vec_all[i]); // 将b[i]的值扩展到一个128位的寄存器sum中
        int j;
        for (j = i + 1; j < N - 4; j += 4) { // 使用SSE指令，每次处理4个元素
            __m128 mjxj = _mm_loadu_ps(&m_sse_vec_all[i][j]); // 将m[i][j]~m[i][j+3]加载到一个128位的寄存器mjxj中
            __m128 xj = _mm_loadu_ps(&x_sse_vec_all[j]); // 将x[j]~x[j+3]加载到一个128位的寄存器xj中
            sum = _mm_sub_ps(sum, _mm_mul_ps(mjxj, xj)); // 使用SSE指令进行减法和乘法
        }
        // 处理剩余的不足4个元素，使用串行方式
        for (; j < N; j++) {
            // 使用SSE指令，每次处理一个元素
            __m128 mjxj = _mm_set_ps(0, 0, 0, m_sse_vec_all[i][j]); // 将m[i][j]扩展到一个128位的寄存器mjxj中
            __m128 xj = _mm_set_ps(0, 0, 0, x_sse_vec_all[j]); // 将x[j]扩展到一个128位的寄存器xj中
            __m128 product = _mm_mul_ps(mjxj, xj); // 使用SSE指令进行乘法
            sum = _mm_sub_ps(sum, product); // 使用SSE指令进行减法
        }
        float result[4]; // 用于存储寄存器sum中的值
        _mm_storeu_ps(result, sum); // 将寄存器sum中的值存入result数组中
        float final_sum = result[0] + result[1] + result[2] + result[3]; // 对result数组中的值进行求和
        x_sse_vec_all[i] = final_sum / m_sse_vec_all[i][i];
    }
}





void gaussian_elimination_avx_vec_all() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        __m256 mk = _mm256_set1_ps(m_avx_vec_all[k][k]); // 将 m[k][k] 的值扩展到一个256位的 AVX 寄存器 mk 中
        for (int i = k + 1; i < N; i++) {
            float factor = m_avx_vec_all[i][k] / m_avx_vec_all[k][k];
            __m256 factor_vec = _mm256_set1_ps(factor); // 将 factor 的值扩展到一个256位的 AVX 寄存器 factor_vec 中
            int j;
            for (j = k; j < N - 8; j += 8) { // 使用 AVX 指令，每次处理8个元素
                __m256 mi = _mm256_loadu_ps(&m_avx_vec_all[i][j]); // 将 m[i][j]~m[i][j+7] 加载到一个256位的 AVX 寄存器 mi 中
                __m256 mkj = _mm256_loadu_ps(&m_avx_vec_all[k][j]); // 将 m[k][j]~m[k][j+7] 加载到一个256位的 AVX 寄存器 mkj 中
                __m256 result = _mm256_sub_ps(mi, _mm256_mul_ps(factor_vec, mkj)); // 使用 AVX 指令进行减法和乘法
                _mm256_storeu_ps(&m_avx_vec_all[i][j], result); // 将结果存回 m[i][j]~m[i][j+7]
            }
            // 处理剩余的不足8个元素，使用串行方式
            for (; j < N; j++) {
                m_avx_vec_all[i][j] -= factor * m_avx_vec_all[k][j];
            }
            b_avx_vec_all[i] -= factor * b_avx_vec_all[k];
        }
    }

    // 回代过程
    x_avx_vec_all[N - 1] = b_avx_vec_all[N - 1] / m_avx_vec_all[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        __m256 sum = _mm256_set1_ps(b_avx_vec_all[i]); // 将 b[i] 的值扩展到一个256位的 AVX 寄存器 sum 中
        int j;
        for (j = i + 1; j < N - 8; j += 8) { // 使用 AVX 指令，每次处理8个元素
            __m256 mjxj = _mm256_loadu_ps(&m_avx_vec_all[i][j]); // 将 m[i][j]~m[i][j+7] 加载到一个256位的 AVX 寄存器 mjxj 中
            __m256 xj = _mm256_loadu_ps(&x_avx_vec_all[j]); // 将 x[j]~x[j+7] 加载到一个256位的 AVX 寄存器 xj 中
            sum = _mm256_sub_ps(sum, _mm256_mul_ps(mjxj, xj)); // 使用 AVX 指令进行减法和乘法
        }
        // 处理剩余的不足8个元素，使用串行方式
        for (; j < N; j++) {
            sum = _mm256_sub_ps(sum, _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, m_avx_vec_all[i][j] * x_avx_vec_all[j])); // 使用 AVX 指令进行减法
        }
        float result[8]; // 用于存储 AVX 寄存器 sum 中的值
        _mm256_storeu_ps(result, sum); // 将 AVX 寄存器 sum 中的值存入 result 数组中
        float final_sum = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7]; // 对 result 数组中的值进行求和
        x_avx_vec_all[i] = final_sum / m_avx_vec_all[i][i];
    }
}


void gaussian_elimination_mp() {
    // 消去过程
#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            float factor = m_mp[i][k] / m_mp[k][k];
#pragma omp parallel for num_threads(NUM_THREADS)
            for (int j = k; j < N; j++) {
                m_mp[i][j] -= factor * m_mp[k][j];
                //if (fabs(m[i][j]) < 1e-6)
                //    m[i][j] = 0; // 小于 10^-6 的数值赋值为 0
            }
            b_mp[i] -= factor * b_mp[k];
        }
    }

    // 回代过程
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


void gaussian_elimination_mp_sse() {
    // 消去过程
#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            float factor = m_mp_sse[i][k] / m_mp_sse[k][k];
            __m128 factor_vec = _mm_set1_ps(factor); // 使用 SSE 加速乘法
#pragma omp simd
            for (int j = k; j < N; j += 4) {
                __m128 m_vec = _mm_loadu_ps(&m_mp_sse[k][j]); // 加载系数矩阵数据
                __m128 result = _mm_loadu_ps(&m_mp_sse[i][j]); // 加载当前行数据
                result = _mm_sub_ps(result, _mm_mul_ps(factor_vec, m_vec)); // 执行向量化的减法和乘法
                _mm_storeu_ps(&m_mp_sse[i][j], result); // 存储结果
            }
            b_mp_sse[i] -= factor * b_mp_sse[k];
        }
    }

    // 回代过程
    x_mp_sse[N - 1] = b_mp_sse[N - 1] / m_mp_sse[N - 1][N - 1];
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_mp_sse[i];
        __m128 x_vec = _mm_loadu_ps(&x_mp_sse[i + 1]); // 加载解向量数据
#pragma omp simd
        for (int j = i + 1; j < N; j += 4) {
            __m128 m_vec = _mm_loadu_ps(&m_mp_sse[i][j]); // 加载系数矩阵数据
            __m128 result = _mm_mul_ps(m_vec, x_vec); // 执行向量化的乘法
            sum -= _mm_cvtss_f32(_mm_hadd_ps(result, result)); // 计算横向加法结果
        }
        x_mp_sse[i] = sum / m_mp_sse[i][i];
    }
}

void gaussian_elimination_mp_avx() {
    // Forward elimination
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            float factor = m_mp_avx[i][k] / m_mp_avx[k][k];

            // Load factors into AVX registers
            __m256 factor_vec = _mm256_set1_ps(factor);
            int j;

            for (j = k; j <= N - 8; j += 8) {
                // Load data into AVX registers
                __m256 row_k = _mm256_load_ps(&m_mp_avx[k][j]);
                __m256 row_i = _mm256_load_ps(&m_mp_avx[i][j]);

                // Perform the elimination
                __m256 temp = _mm256_mul_ps(factor_vec, row_k);
                row_i = _mm256_sub_ps(row_i, temp);

                // Store the result back into the matrix
                _mm256_store_ps(&m_mp_avx[i][j], row_i);
            }

            // Handle the remaining elements
            for (; j < N; j++) {
                m_mp_avx[i][j] -= factor * m_mp_avx[k][j];
            }

            // Update the right-hand side vector
            b_mp_avx[i] -= factor * b_mp_avx[k];
        }
    }

    // Back substitution
    x_mp_avx[N - 1] = b_mp_avx[N - 1] / m_mp_avx[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        __m256 sum_vec = _mm256_setzero_ps();
        int j;

        for (j = i + 1; j <= N - 8; j += 8) {
            __m256 m_vec = _mm256_load_ps(&m_mp_avx[i][j]);
            __m256 x_vec = _mm256_load_ps(&x_mp_avx[j]);

            __m256 temp = _mm256_mul_ps(m_vec, x_vec);
            sum_vec = _mm256_add_ps(sum_vec, temp);
        }

        // Sum the elements of the vector
        float sum_array[8];
        _mm256_store_ps(sum_array, sum_vec);
        float sum = 0.0f;
        for (int k = 0; k < 8; k++) {
            sum += sum_array[k];
        }

        // Handle the remaining elements
        for (; j < N; j++) {
            sum += m_mp_avx[i][j] * x_mp_avx[j];
        }

        x_mp_avx[i] = (b_mp_avx[i] - sum) / m_mp_avx[i][i];
    }
}








void gaussian_elimination_with_pivot() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        // 选择主元
        int max_index = k;
        float max_value = fabs(m_pivot[k][k]);
        for (int i = k + 1; i < N; i++) {
            if (fabs(m_pivot[i][k]) > max_value) {
                max_value = fabs(m_pivot[i][k]);
                max_index = i;
            }
        }

        // 交换主元所在行与当前行
        if (max_index != k) {
            for (int j = k; j < N; j++) {
                swap(m_pivot[k][j], m_pivot[max_index][j]);
            }
            swap(b_pivot[k], b_pivot[max_index]);
        }

        // 消元
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot[i][k] / m_pivot[k][k];
            for (int j = k; j < N; j++) {
                m_pivot[i][j] -= factor * m_pivot[k][j];
            }
            b_pivot[i] -= factor * b_pivot[k];
        }
    }

    // 回代过程
    x_pivot[N - 1] = b_pivot[N - 1] / m_pivot[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_pivot[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m_pivot[i][j] * x_pivot[j];
        }
        x_pivot[i] = sum / m_pivot[i][i];
    }
}


typedef struct {
    int k;      // 消去的轮次
    int tid;    // 线程 id
} ThreadParam_pivot_pthread;

void* threadFunc_pivot_pthread(void* param) {
    ThreadParam_pivot_pthread* p = (ThreadParam_pivot_pthread*)param;
    int k = p->k;       // 消去的轮次
    int tid = p->tid;   // 线程编号

    int start_col = tid * (N / NUM_THREADS);
    int end_col = (tid + 1) * (N / NUM_THREADS);

    for (int j = start_col; j < end_col; ++j) {
        float factor = m_pivot_pthread[k][j] / m_pivot_pthread[k][k];
        for (int i = k + 1; i < N; ++i) {
            m_pivot_pthread[i][j] -= factor * m_pivot_pthread[i][k];
        }
        b_pivot_pthread[j] -= factor * b_pivot_pthread[k];
    }

    pthread_exit(NULL);
    return NULL;
}

void gaussian_elimination_with_pivot_pthread() {
    pthread_t handles[NUM_THREADS];  // 创建对应的 Handle
    ThreadParam_pivot_pthread param[NUM_THREADS];  // 创建对应的线程数据结构

    // 进行高斯消去法计算
    for (int k = 0; k < N; ++k) {
        // 主线程做除法操作
        for (int j = k + 1; j < N; ++j) {
            m_pivot_pthread[k][j] = m_pivot_pthread[k][j] / m_pivot_pthread[k][k];
        }
        m_pivot_pthread[k][k] = 1.0;

        // 创建工作线程，进行消去操作
        int worker_count = N - 1 - k;  // 工作线程数量

        // 分配任务
        for (int tid = 0; tid < NUM_THREADS; ++tid) {
            param[tid].k = k;
            param[tid].tid = tid;
            pthread_create(&handles[tid], NULL, threadFunc_pivot_pthread, (void*)&param[tid]);
        }

        // 主线程挂起等待所有的工作线程完成此轮消去工作
        for (int tid = 0; tid < NUM_THREADS; ++tid) {
            pthread_join(handles[tid], NULL);
        }
    }

    // 回代过程
    x_pivot_pthread[N - 1] = b_pivot_pthread[N - 1] / m_pivot_pthread[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_pivot_pthread[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m_pivot_pthread[i][j] * x_pivot_pthread[j];
        }
        x_pivot_pthread[i] = sum / m_pivot_pthread[i][i];
    }
}



typedef struct {
    int k;      // 消去的轮次
    int tid;    // 线程 id
} ThreadParam_pivot_pthread_ver;
void* threadFunc_pivot_pthread_ver(void* param) {
    ThreadParam_pivot_pthread_ver* p = (ThreadParam_pivot_pthread_ver*)param;
    int k = p->k;       // 消去的轮次
    int tid = p->tid;   // 线程编号

    int start_row = tid * (N / NUM_THREADS);
    int end_row = (tid + 1) * (N / NUM_THREADS);

    for (int i = start_row; i < end_row; ++i) {
        float factor = m_pivot_pthread_ver[i][k] / m_pivot_pthread_ver[k][k];
        for (int j = k + 1; j < N; ++j) {
            m_pivot_pthread_ver[i][j] -= factor * m_pivot_pthread_ver[k][j];
        }
        b_pivot_pthread_ver[i] -= factor * b_pivot_pthread_ver[k];
    }

    pthread_exit(NULL);
    return NULL;
}
void gaussian_elimination_with_pivot_pthread_ver() {
    pthread_t handles[NUM_THREADS];  // 创建对应的 Handle
    ThreadParam_pivot_pthread_ver param[NUM_THREADS];  // 创建对应的线程数据结构

    // 进行高斯消去法计算
    for (int k = 0; k < N; ++k) {
        // 执行部分主元消去
        int max_row = k;
        for (int i = k + 1; i < N; ++i) {
            if (fabs(m_pivot_pthread_ver[i][k]) > fabs(m_pivot_pthread_ver[max_row][k])) {
                max_row = i;
            }
        }

        // 若需要，则交换行
        if (max_row != k) {
            for (int j = k; j < N; ++j) {
                float temp = m_pivot_pthread_ver[k][j];
                m_pivot_pthread_ver[k][j] = m_pivot_pthread_ver[max_row][j];
                m_pivot_pthread_ver[max_row][j] = temp;
            }
            float temp = b_pivot_pthread_ver[k];
            b_pivot_pthread_ver[k] = b_pivot_pthread_ver[max_row];
            b_pivot_pthread_ver[max_row] = temp;
        }

        // 检查部分主元是否为零
        if (fabs(m_pivot_pthread_ver[k][k]) < 1e-7) {
            continue;
        }

        // 主线程做除法操作
        for (int j = k + 1; j < N; ++j) {
            m_pivot_pthread_ver[k][j] = m_pivot_pthread_ver[k][j] / m_pivot_pthread_ver[k][k];
        }
        m_pivot_pthread_ver[k][k] = 1.0;

        // 创建工作线程，进行消去操作
        int worker_count = N - 1 - k;  // 工作线程数量

        // 分配任务
        for (int tid = 0; tid < NUM_THREADS; ++tid) {
            param[tid].k = k;
            param[tid].tid = tid;
            pthread_create(&handles[tid], NULL, threadFunc_pivot_pthread_ver, (void*)&param[tid]);
        }

        // 主线程挂起等待所有的工作线程完成此轮消去工作
        for (int tid = 0; tid < NUM_THREADS; ++tid) {
            pthread_join(handles[tid], NULL);
        }
    }

    // 回代过程
    x_pivot_pthread_ver[N - 1] = b_pivot_pthread_ver[N - 1] / m_pivot_pthread_ver[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_pivot_pthread_ver[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m_pivot_pthread_ver[i][j] * x_pivot_pthread_ver[j];
        }
        x_pivot_pthread_ver[i] = sum / m_pivot_pthread_ver[i][i];
    }
}




// 线程数据结构定义
typedef struct {
    int t_id; // 线程 id
} threadParam_pivot_pthread_s;

// 信号量定义
sem_t sem_leader_pivot_pthread_s;
sem_t sem_Division_pivot_pthread_s[NUM_THREADS - 1];
sem_t sem_Elimination_pivot_pthread_s[NUM_THREADS - 1];
void* threadFunc_pivot_pthread_s(void* param) {
    threadParam_pivot_pthread_s* p = (threadParam_pivot_pthread_s*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {
            // 等待 t_id 为 0 的线程完成除法操作
            for (int i = k + 1; i < N; ++i) {
                m_pivot_pthread_s[i][k] = m_pivot_pthread_s[i][k] / m_pivot_pthread_s[k][k];
            }
            m_pivot_pthread_s[k][k] = 1.0;
        }
        else {
            // 其他线程等待除法操作完成
            sem_wait(&sem_Division_pivot_pthread_s[t_id - 1]);
        }

        if (t_id == 0) {
            // 唤醒其他线程，开始消去操作
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Division_pivot_pthread_s[i]);
            }
        }

        // 循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            for (int j = k + 1; j < N; ++j) {
                m_pivot_pthread_s[i][j] -= m_pivot_pthread_s[i][k] * m_pivot_pthread_s[k][j];
            }
            m_pivot_pthread_s[i][k] = 0.0;
        }

        if (t_id == 0) {
            // t_id 为 0 的线程等待其他线程完成消去操作
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leader_pivot_pthread_s);
            }
            // t_id 为 0 的线程通知其他线程进入下一轮循环
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Elimination_pivot_pthread_s[i]);
            }
        }
        else {
            sem_post(&sem_leader_pivot_pthread_s); // 其他线程通知 t_id 为 0 的线程已完成消去任务
            // 其他线程等待通知，进入下一轮循环
            sem_wait(&sem_Elimination_pivot_pthread_s[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return NULL;
}

void gaussian_elimination_with_pivot_pthread_s() {
    // 初始化信号量
    sem_init(&sem_leader_pivot_pthread_s, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Division_pivot_pthread_s[i], 0, 0);
        sem_init(&sem_Elimination_pivot_pthread_s[i], 0, 0);
    }

    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_pivot_pthread_s params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_pivot_pthread_s, (void*)&params[t_id]);
    }

    // 主线程等待所有线程完成
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

    // 销毁所有信号量
    sem_destroy(&sem_leader_pivot_pthread_s);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Division_pivot_pthread_s[i]);
        sem_destroy(&sem_Elimination_pivot_pthread_s[i]);
    }

    // 回代求解过程
    for (int i = N - 1; i >= 0; --i) {
        x_pivot_pthread_s[i] = b_pivot_pthread_s[i]; // 初始化解向量的第 i 个元素为右侧常数向量的第 i 个元素
        for (int j = i + 1; j < N; ++j) {
            x_pivot_pthread_s[i] -= m_pivot_pthread_s[i][j] * x_pivot_pthread_s[j]; // 更新解向量的第 i 个元素
        }
        x_pivot_pthread_s[i] /= m_pivot_pthread_s[i][i]; // 计算解向量的第 i 个元素
    }
}

typedef struct {
    int t_id; // 线程 id
} threadParam_pivot_sse_vec_pthread;

// 信号量定义
sem_t sem_leader_pivot_sse_vec_pthread;
sem_t sem_Division_pivot_sse_vec_pthread[NUM_THREADS - 1];
sem_t sem_Elimination_pivot_sse_vec_pthread[NUM_THREADS - 1];

void* threadFunc_pivot_sse_vec_pthread(void* param) {
    threadParam_pivot_sse_vec_pthread* p = (threadParam_pivot_sse_vec_pthread*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {
            // 等待 t_id 为 0 的线程完成除法操作
            for (int i = k + 1; i < N; ++i) {
                if (m_pivot_sse_vec_pthread[k][k] != 0.0) {
                    m_pivot_sse_vec_pthread[i][k] = m_pivot_sse_vec_pthread[i][k] / m_pivot_sse_vec_pthread[k][k];
                }
                else {
                   // printf("Error: Division by zero in row %d, column %d.\n", i, k);
                    // 为避免NaN值，设定为0或其他默认值
                    m_pivot_sse_vec_pthread[i][k] = 0.0;
                }
            }
            m_pivot_sse_vec_pthread[k][k] = 1.0;
        }
        else {
            // 其他线程等待除法操作完成
            sem_wait(&sem_Division_pivot_sse_vec_pthread[t_id - 1]);
        }

        if (t_id == 0) {
            // 唤醒其他线程，开始消去操作
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Division_pivot_sse_vec_pthread[i]);
            }
        }

        // 循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            if (m_pivot_sse_vec_pthread[k][k] == 0.0) {
                // 除数为零，跳过当前操作
                continue;
            }
            float factor = m_pivot_sse_vec_pthread[i][k] / m_pivot_sse_vec_pthread[k][k];
            for (int j = k + 1; j < N; j += 8) { // 使用 AVX 寄存器每次处理 8 个元素
                __m256 mk = _mm256_set1_ps(factor);
                __m256 m_row = _mm256_loadu_ps(&m_pivot_sse_vec_pthread[i][j]);
                __m256 mk_mul_mrow = _mm256_mul_ps(mk, m_row);
                __m256 m_sub_result = _mm256_sub_ps(_mm256_loadu_ps(&m_pivot_sse_vec_pthread[i][j]), mk_mul_mrow);
                _mm256_storeu_ps(&m_pivot_sse_vec_pthread[i][j], m_sub_result);
            }
            m_pivot_sse_vec_pthread[i][k] = 0.0;
        }

        if (t_id == 0) {
            // t_id 为 0 的线程等待其他线程完成消去操作
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leader_pivot_sse_vec_pthread);
            }
            // t_id 为 0 的线程通知其他线程进入下一轮循环
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Elimination_pivot_sse_vec_pthread[i]);
            }
        }
        else {
            sem_post(&sem_leader_pivot_sse_vec_pthread); // 其他线程通知 t_id 为 0 的线程已完成消去任务
            // 其他线程等待通知，进入下一轮循环
            sem_wait(&sem_Elimination_pivot_sse_vec_pthread[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return NULL;
}

void gaussian_elimination_with_pivot_sse_vec_pthread() {
    // 初始化信号量
    sem_init(&sem_leader_pivot_sse_vec_pthread, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Division_pivot_sse_vec_pthread[i], 0, 0);
        sem_init(&sem_Elimination_pivot_sse_vec_pthread[i], 0, 0);
    }

    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_pivot_sse_vec_pthread params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_pivot_sse_vec_pthread, (void*)&params[t_id]);
    }

    // 主线程等待所有线程完成
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

    // 销毁所有信号量
    sem_destroy(&sem_leader_pivot_sse_vec_pthread);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Division_pivot_sse_vec_pthread[i]);
        sem_destroy(&sem_Elimination_pivot_sse_vec_pthread[i]);
    }

    // 回代求解过程
    for (int i = N - 1; i >= 0; --i) {
        x_pivot_sse_vec_pthread[i] = b_pivot_sse_vec_pthread[i]; // 初始化解向量的第 i 个元素为右侧常数向量的第 i 个元素
        for (int j = i + 1; j < N; ++j) {
            x_pivot_sse_vec_pthread[i] -= m_pivot_sse_vec_pthread[i][j] * x_pivot_sse_vec_pthread[j]; // 更新解向量的第 i 个元素
        }
        if (m_pivot_sse_vec_pthread[i][i] == 0.0) {
            // 处理除数为零的情况
           // printf("Error: Division by 0 in row %d, column %d during back substitution.\n", i, i);
            x_pivot_sse_vec_pthread[i] = 0.0; // 为避免NaN值，设定为0
        }
        else {
            x_pivot_sse_vec_pthread[i] /= m_pivot_sse_vec_pthread[i][i]; // 计算解向量的第 i 个元素
        }
    }
}



typedef struct {
    int t_id; // 线程 id
} threadParam_pivot_avx_vec_pthread;

// 信号量定义
sem_t sem_leader_pivot_avx_vec_pthread;
sem_t sem_Division_pivot_avx_vec_pthread[NUM_THREADS - 1];
sem_t sem_Elimination_pivot_avx_vec_pthread[NUM_THREADS - 1];

// 线程函数：进行高斯消元算法的一部分
void* threadFunc_pivot_avx_vec_pthread(void* param) {
    threadParam_pivot_avx_vec_pthread* p = (threadParam_pivot_avx_vec_pthread*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {
            // 等待 t_id 为 0 的线程完成除法操作
            for (int i = k + 1; i < N; ++i) {
                // 检查除数是否为零，避免除以零错误
                if (m_pivot_avx_vec_pthread[k][k] != 0.0) {
                    m_pivot_avx_vec_pthread[i][k] = m_pivot_avx_vec_pthread[i][k] / m_pivot_avx_vec_pthread[k][k];
                }
                else {
                    // 处理除以零的情况
                    m_pivot_avx_vec_pthread[i][k] = 0.0;
                }
            }
            m_pivot_avx_vec_pthread[k][k] = 1.0;
        }
        else {
            // 其他线程等待除法操作完成
            sem_wait(&sem_Division_pivot_avx_vec_pthread[t_id - 1]);
        }

        if (t_id == 0) {
            // 唤醒其他线程，开始消去操作
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Division_pivot_avx_vec_pthread[i]);
            }
        }

        // 循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            if (m_pivot_avx_vec_pthread[k][k] == 0.0) {
                // 如果除数为零，跳过当前操作
                continue;
            }
            float factor = m_pivot_avx_vec_pthread[i][k] / m_pivot_avx_vec_pthread[k][k];
            __m256 mk = _mm256_set1_ps(factor); // 初始化 AVX 寄存器
            for (int j = k + 1; j < N; j += 8) { // 使用 AVX 寄存器每次处理 8 个元素
                __m256 m_row = _mm256_loadu_ps(&m_pivot_avx_vec_pthread[i][j]);
                __m256 mk_mul_mrow = _mm256_mul_ps(mk, m_row);
                __m256 m_sub_result = _mm256_sub_ps(_mm256_loadu_ps(&m_pivot_avx_vec_pthread[i][j]), mk_mul_mrow);
                _mm256_storeu_ps(&m_pivot_avx_vec_pthread[i][j], m_sub_result);
            }
            m_pivot_avx_vec_pthread[i][k] = 0.0;
        }

        if (t_id == 0) {
            // t_id 为 0 的线程等待其他线程完成消去操作
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_wait(&sem_leader_pivot_avx_vec_pthread);
            }
            // t_id 为 0 的线程通知其他线程进入下一轮循环
            for (int i = 0; i < NUM_THREADS - 1; ++i) {
                sem_post(&sem_Elimination_pivot_avx_vec_pthread[i]);
            }
        }
        else {
            sem_post(&sem_leader_pivot_avx_vec_pthread); // 其他线程通知 t_id 为 0 的线程已完成消去任务
            // 其他线程等待通知，进入下一轮循环
            sem_wait(&sem_Elimination_pivot_avx_vec_pthread[t_id - 1]);
        }
    }

    pthread_exit(NULL);
    return NULL;
}

// 执行高斯消元算法的函数
void gaussian_elimination_with_pivot_avx_vec_pthread() {
    // 初始化信号量
    sem_init(&sem_leader_pivot_avx_vec_pthread, 0, 0);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_init(&sem_Division_pivot_avx_vec_pthread[i], 0, 0);
        sem_init(&sem_Elimination_pivot_avx_vec_pthread[i], 0, 0);
    }

    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_pivot_avx_vec_pthread params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_pivot_avx_vec_pthread, (void*)&params[t_id]);
    }

    // 主线程等待所有线程完成
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

    // 销毁所有信号量
    sem_destroy(&sem_leader_pivot_avx_vec_pthread);
    for (int i = 0; i < NUM_THREADS - 1; ++i) {
        sem_destroy(&sem_Division_pivot_avx_vec_pthread[i]);
        sem_destroy(&sem_Elimination_pivot_avx_vec_pthread[i]);
    }

    // 回代求解过程
    for (int i = N - 1; i >= 0; --i) {
        x_pivot_avx_vec_pthread[i] = b_pivot_avx_vec_pthread[i]; // 初始化解向量的第 i 个元素为右侧常数向量的第 i 个元素
        for (int j = i + 1; j < N; ++j) {
            x_pivot_avx_vec_pthread[i] -= m_pivot_avx_vec_pthread[i][j] * x_pivot_avx_vec_pthread[j]; // 更新解向量的第 i 个元素
        }
        if (m_pivot_avx_vec_pthread[i][i] == 0.0) {
            // 处理除数为零的情况
            x_pivot_avx_vec_pthread[i] = 0.0; // 为避免 NaN 值，设定为 0
        }
        else {
            x_pivot_avx_vec_pthread[i] /= m_pivot_avx_vec_pthread[i][i]; // 计算解向量的第 i 个元素
        }
    }
}





typedef struct {
    int t_id; // 线程 id
} threadParam_pivot_pthread_s_ver;


// 信号量定义
sem_t sem_Division_pivot_pthread_s_ver[NUM_THREADS];
sem_t sem_Elimination_pivot_pthread_s_ver[NUM_THREADS];
sem_t sem_Continue_pivot_pthread_s_ver;

void* threadFunc_pivot_pthread_s_ver(void* param) {
    threadParam_pivot_pthread_s_ver* p = (threadParam_pivot_pthread_s_ver*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        // 等待除法操作完成
        sem_wait(&sem_Division_pivot_pthread_s_ver[t_id]);

        // 消去操作
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
            for (int j = k + 1; j < N; ++j) {
                m_pivot_pthread_s_ver[i][j] -= m_pivot_pthread_s_ver[i][k] * m_pivot_pthread_s_ver[k][j];
            }
            m_pivot_pthread_s_ver[i][k] = 0.0;
        }

        // 等待其他线程完成消去操作
        sem_post(&sem_Continue_pivot_pthread_s_ver);

        // 等待通知，进入下一轮循环
        sem_wait(&sem_Elimination_pivot_pthread_s_ver[t_id]);
    }

    pthread_exit(NULL);
    return NULL;
}

void gaussian_elimination_with_pivot_pthread_s_ver() {
    // 初始化信号量
    sem_init(&sem_Continue_pivot_pthread_s_ver, 0, 0);
    for (int i = 0; i < NUM_THREADS; ++i) {
        sem_init(&sem_Division_pivot_pthread_s_ver[i], 0, 0);
        sem_init(&sem_Elimination_pivot_pthread_s_ver[i], 0, 0);
    }

    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_pivot_pthread_s_ver params[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_pivot_pthread_s_ver, (void*)&params[t_id]);
    }

    for (int k = 0; k < N; ++k) {
        // 完成除法操作
        for (int i = k + 1; i < N; ++i) {
            m_pivot_pthread_s_ver[i][k] = m_pivot_pthread_s_ver[i][k] / m_pivot_pthread_s_ver[k][k];
        }
        m_pivot_pthread_s_ver[k][k] = 1.0;

        // 通知所有线程开始消去操作
        for (int i = 0; i < NUM_THREADS; ++i) {
            sem_post(&sem_Division_pivot_pthread_s_ver[i]);
        }

        // 等待所有线程完成消去操作
        for (int i = 0; i < NUM_THREADS; ++i) {
            sem_wait(&sem_Continue_pivot_pthread_s_ver);
        }

        // 通知所有线程进入下一轮循环
        for (int i = 0; i < NUM_THREADS; ++i) {
            sem_post(&sem_Elimination_pivot_pthread_s_ver[i]);
        }
    }

    // 主线程等待所有线程完成
    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

    // 销毁所有信号量
    sem_destroy(&sem_Continue_pivot_pthread_s_ver);
    for (int i = 0; i < NUM_THREADS; ++i) {
        sem_destroy(&sem_Division_pivot_pthread_s_ver[i]);
        sem_destroy(&sem_Elimination_pivot_pthread_s_ver[i]);
    }

    // 回代求解过程
    for (int i = N - 1; i >= 0; --i) {
        x_pivot_pthread_s_ver[i] = b_pivot_pthread_s_ver[i];
        for (int j = i + 1; j < N; ++j) {
            x_pivot_pthread_s_ver[i] -= m_pivot_pthread_s_ver[j][i] * x_pivot_pthread_s_ver[j];
        }
        x_pivot_pthread_s_ver[i] /= m_pivot_pthread_s_ver[i][i];
    }
}


void gaussian_elimination_with_pivot_sse_vec_all() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        // 选择主元
        int max_index = k;
        float max_value = fabsf(m_pivot_sse_vec_all[k][k]);
        for (int i = k + 1; i < N; i++) {
            if (fabsf(m_pivot_sse_vec_all[i][k]) > max_value) {
                max_value = fabsf(m_pivot_sse_vec_all[i][k]);
                max_index = i;
            }
        }

        // 交换主元所在行与当前行
        if (max_index != k) {
            // 使用 SSE 寄存器进行交换
            __m128i mask = _mm_castps_si128(_mm_cmpeq_ps(_mm_set1_ps(0.0f), _mm_set1_ps(0.0f)));
            for (int j = k; j < N; j += 4) {
                __m128 temp_m0 = _mm_loadu_ps(&m_pivot_sse_vec_all[k][j]);
                __m128 temp_m1 = _mm_loadu_ps(&m_pivot_sse_vec_all[max_index][j]);
                _mm_maskstore_ps(&m_pivot_sse_vec_all[k][j], mask, temp_m1);
                _mm_maskstore_ps(&m_pivot_sse_vec_all[max_index][j], mask, temp_m0);
            }
            swap(b_pivot_sse_vec_all[k], b_pivot_sse_vec_all[max_index]);
        }

        // 消元
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_sse_vec_all[i][k] / m_pivot_sse_vec_all[k][k];
            __m128 factor_vec = _mm_set1_ps(factor);
            __m128* m_i = (__m128*) & m_pivot_sse_vec_all[i][k];
            __m128* m_k = (__m128*) & m_pivot_sse_vec_all[k][k];
            for (int j = k; j < N; j += 4) {
                __m128 m_kj_vec = _mm_loadu_ps(&m_pivot_sse_vec_all[k][j]);
                __m128 result = _mm_sub_ps(_mm_loadu_ps(&m_pivot_sse_vec_all[i][j]), _mm_mul_ps(factor_vec, m_kj_vec));
                _mm_storeu_ps(&m_pivot_sse_vec_all[i][j], result);
            }
            b_pivot_sse_vec_all[i] -= factor * b_pivot_sse_vec_all[k];
        }

    }

    __m128 x_temp = _mm_set1_ps(0.0f);
    for (int j = N - 1; j >= 0; j--) {
        x_temp = _mm_set1_ps(b_pivot_sse_vec_all[j]);
        for (int k = j + 1; k < N; k++) {
            __m128 x_k = _mm_loadu_ps(&x_pivot_sse_vec_all[k]);
            __m128 m_jk = _mm_set1_ps(m_pivot_sse_vec_all[j][k]);
            x_temp = _mm_sub_ps(x_temp, _mm_mul_ps(x_k, m_jk));
        }
        x_pivot_sse_vec_all[j] = _mm_cvtss_f32(_mm_div_ss(x_temp, _mm_set1_ps(m_pivot_sse_vec_all[j][j])));
    }
}





void gaussian_elimination_with_pivot_avx_vec_all() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        // 选择主元
        int max_index = k;
        float max_value = fabs(m_pivot_avx_vec_all[k][k]);
        for (int i = k + 1; i < N; i++) {
            if (fabs(m_pivot_avx_vec_all[i][k]) > max_value) {
                max_value = fabs(m_pivot_avx_vec_all[i][k]);
                max_index = i;
            }
        }

        // 交换主元所在行与当前行
        if (max_index != k) {
            __m256 swap_temp;
            for (int j = k; j < N; j += 8) { // 使用 AVX 指令集进行交换
                swap_temp = _mm256_load_ps(&m_pivot_avx_vec_all[k][j]);
                _mm256_store_ps(&m_pivot_avx_vec_all[k][j], _mm256_load_ps(&m_pivot_avx_vec_all[max_index][j]));
                _mm256_store_ps(&m_pivot_avx_vec_all[max_index][j], swap_temp);
            }
            swap(b_pivot_avx_vec_all[k], b_pivot_avx_vec_all[max_index]);
        }

        // 消元
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_avx_vec_all[i][k] / m_pivot_avx_vec_all[k][k];
            __m256 factor_vec = _mm256_set1_ps(factor);
            __m256* m_i = (__m256*) & m_pivot_avx_vec_all[i][k];
            __m256* m_k = (__m256*) & m_pivot_avx_vec_all[k][k];
            for (int j = k; j < N; j += 8) { // 使用 AVX 指令集进行消元
                __m256 m_kj_vec = _mm256_load_ps(&m_pivot_avx_vec_all[k][j]);
                __m256 result = _mm256_sub_ps(_mm256_load_ps(&m_pivot_avx_vec_all[i][j]), _mm256_mul_ps(factor_vec, m_kj_vec));
                _mm256_store_ps(&m_pivot_avx_vec_all[i][j], result);
            }
            b_pivot_avx_vec_all[i] -= factor * b_pivot_avx_vec_all[k];
        }
    }

    // 回代过程
    x_pivot_avx_vec_all[N - 1] = b_pivot_avx_vec_all[N - 1] / m_pivot_avx_vec_all[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        __m256 sum = _mm256_set1_ps(b_pivot_avx_vec_all[i]);
        for (int j = i + 1; j < N; j += 8) { // 使用 AVX 指令集进行回代
            __m256 mjxj = _mm256_load_ps(&m_pivot_avx_vec_all[i][j]);
            __m256 xj = _mm256_load_ps(&x_pivot_avx_vec_all[j]);
            sum = _mm256_sub_ps(sum, _mm256_mul_ps(mjxj, xj));
        }
        float result[8]; // 用于存储 AVX 寄存器 sum 中的值
        _mm256_storeu_ps(result, sum); // 将 AVX 寄存器 sum 中的值存入 result 数组中
        float final_sum = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7]; // 对 result 数组中的值进行求和
        x_pivot_avx_vec_all[i] = final_sum / m_pivot_avx_vec_all[i][i];
    }
}



void gaussian_elimination_with_pivot_mp() {
    // 消去过程
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++) {
        // 选择主元
        int max_index = k;
        float max_value = fabs(m_pivot_mp[k][k]);
#pragma omp simd
        for (int i = k + 1; i < N; i++) {
            if (fabs(m_pivot_mp[i][k]) > max_value) {
                max_value = fabs(m_pivot_mp[i][k]);
                max_index = i;
            }
        }

        // 交换主元所在行与当前行
        if (max_index != k) {
#pragma omp critical
            {
                for (int j = k; j < N; j++) {
                    swap(m_pivot_mp[k][j], m_pivot_mp[max_index][j]);
                }
                swap(b_pivot_mp[k], b_pivot_mp[max_index]);
            }
        }

        // 消元
#pragma omp simd
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_mp[i][k] / m_pivot_mp[k][k];
            for (int j = k; j < N; j++) {
                m_pivot_mp[i][j] -= factor * m_pivot_mp[k][j];
            }
            b_pivot_mp[i] -= factor * b_pivot_mp[k];
        }
    }

    // 回代过程
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

void gaussian_elimination_with_pivot_mp_sse() {
    // 消去过程
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++) {
        // 选择主元
        int max_index = k;
        float max_value = fabs(m_pivot_mp_sse[k][k]);
#pragma omp simd
        for (int i = k + 1; i < N; i++) {
            if (fabs(m_pivot_mp_sse[i][k]) > max_value) {
                max_value = fabs(m_pivot_mp_sse[i][k]);
                max_index = i;
            }
        }

        // 交换主元所在行与当前行
        if (max_index != k) {
#pragma omp critical
            {
                for (int j = k; j < N; j++) {
                    swap(m_pivot_mp_sse[k][j], m_pivot_mp_sse[max_index][j]);
                }
                swap(b_pivot_mp_sse[k], b_pivot_mp_sse[max_index]);
            }
        }

        // 消元
#pragma omp simd
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_mp_sse[i][k] / m_pivot_mp_sse[k][k];
            __m128 factor_vec = _mm_set1_ps(factor); // 使用 SSE 加速乘法
            for (int j = k; j < N; j += 4) {
                __m128 m_vec = _mm_loadu_ps(&m_pivot_mp_sse[k][j]); // 加载系数矩阵数据
                __m128 result = _mm_loadu_ps(&m_pivot_mp_sse[i][j]); // 加载当前行数据
                result = _mm_sub_ps(result, _mm_mul_ps(factor_vec, m_vec)); // 执行向量化的减法和乘法
                _mm_storeu_ps(&m_pivot_mp_sse[i][j], result); // 存储结果
            }
            b_pivot_mp_sse[i] -= factor * b_pivot_mp_sse[k];
        }
    }

    // 回代过程
    x_pivot_mp_sse[N - 1] = b_pivot_mp_sse[N - 1] / m_pivot_mp_sse[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_pivot_mp_sse[i];
#pragma omp simd
        for (int j = i + 1; j < N; j++) {
            sum -= m_pivot_mp_sse[i][j] * x_pivot_mp_sse[j];
        }
        x_pivot_mp_sse[i] = sum / m_pivot_mp_sse[i][i];
    }
}

void gaussian_elimination_with_pivot_mp_avx() {
    // 消去过程
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int k = 0; k < N; k++) {
        // 选择主元
        int max_index = k;
        float max_value = fabs(m_pivot_mp_avx[k][k]);
#pragma omp simd
        for (int i = k + 1; i < N; i++) {
            if (fabs(m_pivot_mp_avx[i][k]) > max_value) {
                max_value = fabs(m_pivot_mp_avx[i][k]);
                max_index = i;
            }
        }

        // 交换主元所在行与当前行
        if (max_index != k) {
#pragma omp critical
            {
                for (int j = k; j < N; j++) {
                    swap(m_pivot_mp_avx[k][j], m_pivot_mp_avx[max_index][j]);
                }
                swap(b_pivot_mp_avx[k], b_pivot_mp_avx[max_index]);
            }
        }

        // 消元
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_mp_avx[i][k] / m_pivot_mp_avx[k][k];
            __m256 factor_vec = _mm256_set1_ps(factor); // 使用 AVX 加速乘法
            for (int j = k; j < N; j += 8) {
                __m256 m_vec = _mm256_loadu_ps(&m_pivot_mp_avx[k][j]); // 加载系数矩阵数据
                __m256 result = _mm256_loadu_ps(&m_pivot_mp_avx[i][j]); // 加载当前行数据
                result = _mm256_sub_ps(result, _mm256_mul_ps(factor_vec, m_vec)); // 执行向量化的减法和乘法
                _mm256_storeu_ps(&m_pivot_mp_avx[i][j], result); // 存储结果
            }
            b_pivot_mp_avx[i] -= factor * b_pivot_mp_avx[k];
        }
    }

    // 回代过程
    x_pivot_mp_avx[N - 1] = b_pivot_mp_avx[N - 1] / m_pivot_mp_avx[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_pivot_mp_avx[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m_pivot_mp_avx[i][j] * x_pivot_mp_avx[j];
        }
        x_pivot_mp_avx[i] = sum / m_pivot_mp_avx[i][i];
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



    auto start_hor = chrono::high_resolution_clock::now();
    gaussian_elimination_pthread_hor();
    auto end_hor = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_hor = end_hor - start_hor;
    float execution_guass_time_hor = duration_hor.count();
    cout << "Execution GUASS D HOR Pthread time: " << execution_guass_time_hor << " ms" << endl;

    auto start_pthread = chrono::high_resolution_clock::now();
    gaussian_elimination_pthread();
    auto end_pthread = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pthread = end_pthread - start_pthread;
    float execution_guass_time_pthread = duration_pthread.count();
    cout << "Execution GUASS D Pthread time: " << execution_guass_time_pthread << " ms" << endl;

    auto start_pthread_s = chrono::high_resolution_clock::now();
    gaussian_elimination_pthread_s();
    auto end_pthread_s = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pthread_s = end_pthread_s - start_pthread_s;
    float execution_guass_time_pthread_s = duration_pthread_s.count();
    cout << "Execution GUASS S Pthread time: " << execution_guass_time_pthread_s << " ms" << endl;

    auto start_pthread_s_hor = chrono::high_resolution_clock::now();
    gaussian_elimination_pthread_s_hor();
    auto end_pthread_s_hor = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pthread_s_hor = end_pthread_s_hor - start_pthread_s_hor;
    float execution_guass_time_pthread_s_hor = duration_pthread_s_hor.count();
    cout << "Execution GUASS S HOR Pthread time: " << execution_guass_time_pthread_s_hor << " ms" << endl;

    auto start_sse_vec_all = chrono::high_resolution_clock::now();
    gaussian_elimination_sse_vec_all();
    auto end_sse_vec_all = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_sse_vec_all = end_sse_vec_all - start_sse_vec_all;
    float execution_guass_time_sse_vec_all = duration_sse_vec_all.count();
    cout << "Execution GUASS SSE VEC ALL time: " << execution_guass_time_sse_vec_all << " ms" << endl;

    auto start_sse_vec_pthread = chrono::high_resolution_clock::now();
    gaussian_elimination_sse_vec_pthread();
    auto end_sse_vec_pthread = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_sse_vec_pthread = end_sse_vec_pthread - start_sse_vec_pthread;
    float execution_guass_time_sse_vec_pthread = duration_sse_vec_pthread.count();
    cout << "Execution GUASS SSE VEC Pthread time: " << execution_guass_time_sse_vec_pthread << " ms" << endl;

   

    auto start_avx_vec_all = chrono::high_resolution_clock::now();
    gaussian_elimination_avx_vec_all();
    auto end_avx_vec_all = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_avx_vec_all = end_avx_vec_all - start_avx_vec_all;
    float execution_guass_time_avx_vec_all = duration_avx_vec_all.count();
    cout << "Execution GUASS AVX VEC ALL time: " << execution_guass_time_avx_vec_all << " ms" << endl;

    auto start_avx_vec_pthread = chrono::high_resolution_clock::now();
    gaussian_elimination_avx_vec_pthread();
    auto end_avx_vec_pthread = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_avx_vec_pthread = end_avx_vec_pthread - start_avx_vec_pthread;
    float execution_guass_time_avx_vec_pthread = duration_avx_vec_pthread.count();
    cout << "Execution GUASS AVX VEC Pthread time: " << execution_guass_time_avx_vec_pthread << " ms" << endl;

    
    auto start_mp = chrono::high_resolution_clock::now();
    gaussian_elimination_mp();
    auto end_mp = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_mp = end_mp - start_mp;
    float execution_guass_time_mp = duration_mp.count();
    cout << "Execution GUASS MP time: " << execution_guass_time_mp << " ms" << endl;

    auto start_mp_sse = chrono::high_resolution_clock::now();
    gaussian_elimination_mp_sse();
    auto end_mp_sse = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_mp_sse = end_mp_sse - start_mp_sse;
    float execution_guass_time_mp_sse = duration_mp_sse.count();
    cout << "Execution GUASS MP SSE time: " << execution_guass_time_mp_sse << " ms" << endl;

    auto start_mp_avx = chrono::high_resolution_clock::now();
    gaussian_elimination_mp_avx();
    auto end_mp_avx = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_mp_avx = end_mp_avx - start_mp_avx;
    float execution_guass_time_mp_avx = duration_mp_avx.count();
    cout << "Execution GUASS MP AVX time: " << execution_guass_time_mp_avx << " ms" << endl;




    auto start_pivot = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot();
    auto end_pivot = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot = end_pivot - start_pivot;
    float execution_guass_time_pivot = duration_pivot.count();
    cout << "Execution GUASS PIVOT time: " << execution_guass_time_pivot << " ms" << endl;

    auto start_pivot_pthread = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_pthread();
    auto end_pivot_pthread = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_pthread = end_pivot_pthread - start_pivot_pthread;
    float execution_guass_time_pivot_pthread = duration_pivot_pthread.count();
    cout << "Execution GUASS PIVOT D Pthread time: " << execution_guass_time_pivot_pthread << " ms" << endl;

    auto start_pivot_pthread_ver = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_pthread_ver();
    auto end_pivot_pthread_ver = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_pthread_ver = end_pivot_pthread_ver - start_pivot_pthread_ver;
    float execution_guass_time_pivot_pthread_ver = duration_pivot_pthread_ver.count();
    cout << "Execution GUASS PIVOT D VER Pthread time: " << execution_guass_time_pivot_pthread_ver << " ms" << endl;

    auto start_pivot_pthread_s = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_pthread_s();
    auto end_pivot_pthread_s = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_pthread_s = end_pivot_pthread_s - start_pivot_pthread_s;
    float execution_guass_time_pivot_pthread_s = duration_pivot_pthread_s.count();
    cout << "Execution GUASS PIVOT S Pthread time: " << execution_guass_time_pivot_pthread_s << " ms" << endl;

    auto start_pivot_pthread_s_ver = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_pthread_s_ver();
    auto end_pivot_pthread_s_ver = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_pthread_s_ver = end_pivot_pthread_s_ver - start_pivot_pthread_s_ver;
    float execution_guass_time_pivot_pthread_s_ver = duration_pivot_pthread_s_ver.count();
    cout << "Execution GUASS PIVOT S VER Pthread time: " << execution_guass_time_pivot_pthread_s_ver << " ms" << endl;

    auto start_pivot_sse_vec_all = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_sse_vec_all();
    auto end_pivot_sse_vec_all = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_sse_vec_all = end_pivot_sse_vec_all - start_pivot_sse_vec_all;
    float execution_guass_time_pivot_sse_vec_all = duration_pivot_sse_vec_all.count();
    cout << "Execution GUASS PIVOT SSE VEC all time: " << execution_guass_time_pivot_sse_vec_all << " ms" << endl;

    auto start_pivot_sse_vec_pthread = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_sse_vec_pthread();
    auto end_pivot_sse_vec_pthread = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_sse_vec_pthread = end_pivot_sse_vec_pthread - start_pivot_sse_vec_pthread;
    float execution_guass_time_pivot_sse_vec_pthread = duration_pivot_sse_vec_pthread.count();
    cout << "Execution GUASS PIVOT SSE VEC Pthread time: " << execution_guass_time_pivot_sse_vec_pthread << " ms" << endl;

   
    auto start_pivot_avx_vec_all = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_avx_vec_all();
    auto end_pivot_avx_vec_all = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_avx_vec_all = end_pivot_avx_vec_all - start_pivot_avx_vec_all;
    float execution_guass_time_pivot_avx_vec_all = duration_pivot_avx_vec_all.count();
    cout << "Execution GUASS PIVOT AVX VEC all time: " << execution_guass_time_pivot_avx_vec_all << " ms" << endl;

    auto start_pivot_avx_vec_pthread = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_avx_vec_pthread();
    auto end_pivot_avx_vec_pthread = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_avx_vec_pthread = end_pivot_avx_vec_pthread - start_pivot_avx_vec_pthread;
    float execution_guass_time_pivot_avx_vec_pthread = duration_pivot_avx_vec_pthread.count();
    cout << "Execution GUASS PIVOT AVX  Pthread time: " << execution_guass_time_pivot_avx_vec_pthread << " ms" << endl;
   

    auto start_pivot_mp = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_mp();
    auto end_pivot_mp = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_mp = end_pivot_mp - start_pivot_mp;
    float execution_guass_time_pivot_mp = duration_pivot_mp.count();
    cout << "Execution GUASS PIVOT MP time: " << execution_guass_time_pivot_mp << " ms" << endl;

    auto start_pivot_mp_sse = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_mp_sse();
    auto end_pivot_mp_sse = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_mp_sse = end_pivot_mp_sse - start_pivot_mp_sse;
    float execution_guass_time_pivot_mp_sse = duration_pivot_mp_sse.count();
    cout << "Execution GUASS PIVOT MP SSE time: " << execution_guass_time_pivot_mp_sse << " ms" << endl;

    auto start_pivot_mp_avx = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_mp_avx();
    auto end_pivot_mp_avx = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_mp_avx = end_pivot_mp_avx - start_pivot_mp_avx;
    float execution_guass_time_pivot_mp_avx = duration_pivot_mp_avx.count();
    cout << "Execution GUASS PIVOT MP AVX time: " << execution_guass_time_pivot_mp_avx << " ms" << endl;

    //     //输出结果
    //cout << "Results:" << endl;
    //for (int i = 0; i < N; i++) {
    //    for (int j = 0; j < N; j++) {
    //        // cout << "m[" << i << "]["<<j<<"] = " << m[i][j] << " ";
    //        cout << m_mp_avx[i][j] << " ";
    //    }
    //    cout << endl;}
    
    //
    //检验解决-nan(ind)
    /*for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << m_pivot_avx_aligned[i][j] << " ";
        }
        cout << endl;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << m_pivot_avx_vec_all[i][j] << " ";
        }
        cout << endl;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << m_pivot_avx_vec_e[i][j] << " ";
        }
        cout << endl;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << m_pivot_avx_vec_b[i][j] << " ";
        }
        cout << endl;
    }*/
    /*   for (int i = 0; i < N; i++) {
           for (int j = 0; j < N; j++) {
               cout << m_avx_aligned[i][j] << " ";
           }
           cout << endl;
       }
       for (int i = 0; i < N; i++) {
           for (int j = 0; j < N; j++) {
               cout << m_sse_vec_all[i][j] << " ";
           }
           cout << endl;
       }
       for (int i = 0; i < N; i++) {
           for (int j = 0; j < N; j++) {
               cout << m_sse_aligned[i][j] << " ";
           }
           cout << endl;
       }*/
       //for (int i = 0; i < N; i++) {
      //     //cout << "x[" << i << "] = " << x[i] << endl;
      //    cout << "x[" << i << "] = " << x[i] << " ";
      // }检验能正确输出
      /* _aligned_free(m_sse_aligned);
       _aligned_free(b_sse_aligned);
       _aligned_free(x_sse_aligned);
       _aligned_free(m_avx_aligned);
       _aligned_free(b_avx_aligned);
       _aligned_free(x_avx_aligned);
       _aligned_free(m_pivot_sse_aligned);
       _aligned_free(b_pivot_sse_aligned);
       _aligned_free(x_pivot_sse_aligned);
       _aligned_free(m_pivot_avx_aligned);
       _aligned_free(b_pivot_avx_aligned);
       _aligned_free(x_pivot_avx_aligned);*/
    return 0;
}