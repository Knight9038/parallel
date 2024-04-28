#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <immintrin.h> // 包含 SSE/AVX 指令集的头文件
#include <stddef.h> // For size_t
#include <stdlib.h>
#include <malloc.h>
using namespace std;
#define N 160// 矩阵的大小

float m[N][N];
float b[N];
float x[N]; 
float m_sse_vec_all[N][N];
float b_sse_vec_all[N];
float x_sse_vec_all[N];
alignas(16)float m_sse_aligned[N][N];
alignas(16)float b_sse_aligned[N];
alignas(16)float x_sse_aligned[N];
//float(*m_sse_aligned)[N] = (float(*)[N])_aligned_malloc(N * sizeof(float[N]), 16);// 为 m_sse_aligned 分配内存并按照 16 字节对齐
//float* b_sse_aligned = (float*)_aligned_malloc(N * sizeof(float), 16);// 为 b_sse_aligned 分配内存并按照 16 字节对齐
//float* x_sse_aligned = (float*)_aligned_malloc(N * sizeof(float), 16);// 为 x_sse_aligned 分配内存并按照 16 字节对齐
float m_sse_vec_e[N][N];
float b_sse_vec_e[N];
float x_sse_vec_e[N];
float m_sse_vec_b[N][N];
float b_sse_vec_b[N];
float x_sse_vec_b[N];
float m_avx_vec_all[N][N];
float b_avx_vec_all[N];
float x_avx_vec_all[N]; 
alignas(32)float m_avx_aligned[N][N];
alignas(32)float b_avx_aligned[N];
alignas(32)float x_avx_aligned[N];
float m_avx_vec_e[N][N];
float b_avx_vec_e[N];
float x_avx_vec_e[N];
float m_avx_vec_b[N][N];
float b_avx_vec_b[N];
float x_avx_vec_b[N];
float m_pivot[N][N];
float b_pivot[N];
float x_pivot[N];
float m_pivot_sse_vec_all[N][N];
float b_pivot_sse_vec_all[N];
float x_pivot_sse_vec_all[N];
alignas(16)float m_pivot_sse_aligned[N][N];
alignas(16)float b_pivot_sse_aligned[N];
alignas(16)float x_pivot_sse_aligned[N];
float m_pivot_sse_vec_b[N][N];
float b_pivot_sse_vec_b[N];
float x_pivot_sse_vec_b[N];
float m_pivot_sse_vec_e[N][N];
float b_pivot_sse_vec_e[N];
float x_pivot_sse_vec_e[N];
float m_pivot_avx_vec_all[N][N];
float b_pivot_avx_vec_all[N];
float x_pivot_avx_vec_all[N]; 
alignas(32)float m_pivot_avx_aligned[N][N];
alignas(32)float b_pivot_avx_aligned[N];
alignas(32)float x_pivot_avx_aligned[N];
float m_pivot_avx_vec_e[N][N];
float b_pivot_avx_vec_e[N];
float x_pivot_avx_vec_e[N];
float m_pivot_avx_vec_b[N][N];
float b_pivot_avx_vec_b[N];
float x_pivot_avx_vec_b[N];

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
            m[i][j] = rand()%21;

    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                if(m[i][j]<21)
                m[i][j] += m[k][j];

    for (int i = 0; i < N; i++)//初始化b[N]
        b[i] = rand() % 21;
    //保证矩阵同样
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m_sse_vec_all[i][j] = m[i][j];
            m_sse_vec_b[i][j] = m[i][j];
            m_sse_vec_e[i][j] = m[i][j];
            m_avx_vec_all[i][j] = m[i][j];
            m_avx_vec_b[i][j] = m[i][j];
            m_avx_vec_e[i][j] = m[i][j];
            m_pivot[i][j] = m[i][j];
            m_pivot_sse_vec_all[i][j] = m[i][j];
            m_pivot_sse_vec_b[i][j] = m[i][j];
            m_pivot_sse_vec_e[i][j] = m[i][j];
            m_pivot_avx_vec_all[i][j] = m[i][j];
            m_pivot_avx_vec_b[i][j] = m[i][j];
            m_pivot_avx_vec_e[i][j] = m[i][j];
            m_sse_aligned[i][j] = m[i][j];
            m_avx_aligned[i][j] = m[i][j];
            m_pivot_sse_aligned[i][j] = m[i][j];
            m_pivot_avx_aligned[i][j] = m[i][j];
        }
    }
    for (int i = 0; i < N; i++) {
        b_sse_vec_all[i] = b[i];
        b_sse_vec_b[i] = b[i];
        b_sse_vec_e[i] = b[i];
        b_avx_vec_all[i] = b[i];
        b_avx_vec_b[i] = b[i];
        b_avx_vec_e[i] = b[i];
        b_pivot[i] = b[i];
        b_pivot_sse_vec_all[i] = b[i];
        b_pivot_sse_vec_b[i] = b[i];
        b_pivot_sse_vec_e[i] = b[i];
        b_pivot_avx_vec_all[i] = b[i];
        b_pivot_avx_vec_b[i] = b[i];
        b_pivot_avx_vec_e[i] = b[i];
        b_sse_aligned[i] = b[i];
        b_avx_aligned[i] = b[i];
        b_pivot_sse_aligned[i] = b[i];
        b_pivot_avx_aligned[i] = b[i];
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

void gaussian_elimination_sse_aligned() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        __m128 mk = _mm_set1_ps(m_sse_aligned[k][k]); // 将m[k][k]的值扩展到一个128位的寄存器mk中
        for (int i = k + 1; i < N; i++) {
            float factor = m_sse_aligned[i][k] / m_sse_aligned[k][k];
            __m128 factor_vec = _mm_set1_ps(factor); // 将factor的值扩展到一个128位的寄存器factor_vec中
            int j;
            for (j = k; j < N - 4; j += 4) { // 使用SSE指令，每次处理4个元素
                __m128 mi = _mm_load_ps(&m_sse_aligned[i][j]); // 将m[i][j]~m[i][j+3]加载到一个128位的寄存器mi中
                __m128 mkj = _mm_load_ps(&m_sse_aligned[k][j]); // 将m[k][j]~m[k][j+3]加载到一个128位的寄存器mkj中
                __m128 result = _mm_sub_ps(mi, _mm_mul_ps(factor_vec, mkj)); // 使用SSE指令进行减法和乘法
                _mm_store_ps(&m_sse_aligned[i][j], result); // 将结果存回m[i][j]~m[i][j+3]
            }
            // 处理剩余的不足4个元素，使用串行方式
            for (; j < N; j++) {
                m_sse_aligned[i][j] -= factor * m_sse_aligned[k][j];
            }
            b_sse_aligned[i] -= factor * b_sse_aligned[k];
        }
    }

    // 回代过程
    x_sse_aligned[N - 1] = b_sse_aligned[N - 1] / m_sse_aligned[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        __m128 sum = _mm_set1_ps(b_sse_aligned[i]); // 将b[i]的值扩展到一个128位的寄存器sum中
        int j;
        for (j = i + 1; j < N - 4; j += 4) { // 使用SSE指令，每次处理4个元素
            __m128 mjxj = _mm_load_ps(&m_sse_aligned[i][j]); // 将m[i][j]~m[i][j+3]加载到一个128位的寄存器mjxj中
            __m128 xj = _mm_load_ps(&x_sse_aligned[j]); // 将x[j]~x[j+3]加载到一个128位的寄存器xj中
            sum = _mm_sub_ps(sum, _mm_mul_ps(mjxj, xj)); // 使用SSE指令进行减法和乘法
        }
        // 处理剩余的不足4个元素，使用串行方式
        for (; j < N; j++) {
            // 使用SSE指令，每次处理一个元素
            __m128 mjxj = _mm_set_ps(0, 0, 0, m_sse_aligned[i][j]); // 将m[i][j]扩展到一个128位的寄存器mjxj中
            __m128 xj = _mm_set_ps(0, 0, 0, x_sse_aligned[j]); // 将x[j]扩展到一个128位的寄存器xj中
            __m128 product = _mm_mul_ps(mjxj, xj); // 使用SSE指令进行乘法
            sum = _mm_sub_ps(sum, product); // 使用SSE指令进行减法
        }
        float result[4]; // 用于存储寄存器sum中的值
        _mm_storeu_ps(result, sum); // 将寄存器sum中的值存入result数组中
        float final_sum = result[0] + result[1] + result[2] + result[3]; // 对result数组中的值进行求和
        x_sse_aligned[i] = final_sum / m_sse_aligned[i][i];
    }
}

void gaussian_elimination_sse_vec_e() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        __m128 mk = _mm_set1_ps(m_sse_vec_e[k][k]); // 将m[k][k]的值扩展到一个128位的寄存器mk中
        for (int i = k + 1; i < N; i++) {
            float factor = m_sse_vec_e[i][k] / m_sse_vec_e[k][k];
            __m128 factor_vec = _mm_set1_ps(factor); // 将factor的值扩展到一个128位的寄存器factor_vec中
            int j;
            for (j = k; j < N - 4; j += 4) { // 使用SSE指令，每次处理4个元素
                __m128 mi = _mm_loadu_ps(&m_sse_vec_e[i][j]); // 将m[i][j]~m[i][j+3]加载到一个128位的寄存器mi中
                __m128 mkj = _mm_loadu_ps(&m_sse_vec_e[k][j]); // 将m[k][j]~m[k][j+3]加载到一个128位的寄存器mkj中
                __m128 result = _mm_sub_ps(mi, _mm_mul_ps(factor_vec, mkj)); // 使用SSE指令进行减法和乘法
                _mm_storeu_ps(&m_sse_vec_e[i][j], result); // 将结果存回m[i][j]~m[i][j+3]
            }
            // 处理剩余的不足4个元素，使用串行方式
            for (; j < N; j++) {
                m_sse_vec_e[i][j] -= factor * m_sse_vec_e[k][j];
            }
            b_sse_vec_e[i] -= factor * b_sse_vec_e[k];
        }
    }

    // 回代过程
    x_sse_vec_e[N - 1] = b_sse_vec_e[N - 1] / m_sse_vec_e[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_sse_vec_e[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m_sse_vec_e[i][j] * x_sse_vec_e[j];
        }
        x[i] = sum / m_sse_vec_e[i][i];
    }
}

void gaussian_elimination_sse_vec_b() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            float factor = m_sse_vec_b[i][k] / m_sse_vec_b[k][k];
            for (int j = k; j < N; j++) {
                m_sse_vec_b[i][j] -= factor * m_sse_vec_b[k][j];
            }
            b_sse_vec_b[i] -= factor * b_sse_vec_b[k];
        }
    }

    // 回代过程
    x_sse_vec_b[N - 1] = b_sse_vec_b[N - 1] / m_sse_vec_b[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        __m128 sum = _mm_set1_ps(b_sse_vec_b[i]); // 将b[i]的值扩展到一个128位的寄存器sum中
        int j;
        for (j = i + 1; j < N - 4; j += 4) { // 使用SSE指令，每次处理4个元素
            __m128 mjxj = _mm_loadu_ps(&m_sse_vec_b[i][j]); // 将m[i][j]~m[i][j+3]加载到一个128位的寄存器mjxj中
            __m128 xj = _mm_loadu_ps(&x_sse_vec_b[j]); // 将x[j]~x[j+3]加载到一个128位的寄存器xj中
            sum = _mm_sub_ps(sum, _mm_mul_ps(mjxj, xj)); // 使用SSE指令进行减法和乘法
        }
        // 处理剩余的不足4个元素，使用串行方式
        for (; j < N; j++) {
            // 使用SSE指令，每次处理一个元素
            __m128 mjxj = _mm_set_ps(0, 0, 0, m_sse_vec_b[i][j]); // 将m[i][j]扩展到一个128位的寄存器mjxj中
            __m128 xj = _mm_set_ps(0, 0, 0, x_sse_vec_b[j]); // 将x[j]扩展到一个128位的寄存器xj中
            __m128 product = _mm_mul_ps(mjxj, xj); // 使用SSE指令进行乘法
            sum = _mm_sub_ps(sum, product); // 使用SSE指令进行减法
        }
        float result[4]; // 用于存储寄存器sum中的值
        _mm_storeu_ps(result, sum); // 将寄存器sum中的值存入result数组中
        float final_sum = result[0] + result[1] + result[2] + result[3]; // 对result数组中的值进行求和
        x_sse_vec_b[i] = final_sum / m_sse_vec_b[i][i];
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

void gaussian_elimination_avx_aligned() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        __m256 mk = _mm256_set1_ps(m_avx_aligned[k][k]); // 将 m[k][k] 的值扩展到一个256位的 AVX 寄存器 mk 中
        for (int i = k + 1; i < N; i++) {
            float factor = m_avx_aligned[i][k] / m_avx_aligned[k][k];
            __m256 factor_vec = _mm256_set1_ps(factor); // 将 factor 的值扩展到一个256位的 AVX 寄存器 factor_vec 中
            int j;
            for (j = k; j < N - 8; j += 8) { // 使用 AVX 指令，每次处理8个元素
                __m256 mi = _mm256_load_ps(&m_avx_aligned[i][j]); // 将 m[i][j]~m[i][j+7] 加载到一个256位的 AVX 寄存器 mi 中
                __m256 mkj = _mm256_load_ps(&m_avx_aligned[k][j]); // 将 m[k][j]~m[k][j+7] 加载到一个256位的 AVX 寄存器 mkj 中
                __m256 result = _mm256_sub_ps(mi, _mm256_mul_ps(factor_vec, mkj)); // 使用 AVX 指令进行减法和乘法
                _mm256_store_ps(&m_avx_aligned[i][j], result); // 将结果存回 m[i][j]~m[i][j+7]
            }
            // 处理剩余的不足8个元素，使用串行方式
            for (; j < N; j++) {
                m_avx_aligned[i][j] -= factor * m_avx_aligned[k][j];
            }
            b_avx_aligned[i] -= factor * b_avx_aligned[k];
        }
    }

    // 回代过程
    x_avx_aligned[N - 1] = b_avx_aligned[N - 1] / m_avx_aligned[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        __m256 sum = _mm256_set1_ps(b_avx_aligned[i]); // 将 b[i] 的值扩展到一个256位的 AVX 寄存器 sum 中
        int j;
        for (j = i + 1; j < N - 8; j += 8) { // 使用 AVX 指令，每次处理8个元素
            __m256 mjxj = _mm256_load_ps(&m_avx_aligned[i][j]); // 将 m[i][j]~m[i][j+7] 加载到一个256位的 AVX 寄存器 mjxj 中
            __m256 xj = _mm256_load_ps(&x_avx_aligned[j]); // 将 x[j]~x[j+7] 加载到一个256位的 AVX 寄存器 xj 中
            sum = _mm256_sub_ps(sum, _mm256_mul_ps(mjxj, xj)); // 使用 AVX 指令进行减法和乘法
        }
        // 处理剩余的不足8个元素，使用串行方式
        for (; j < N; j++) {
            sum = _mm256_sub_ps(sum, _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, m_avx_aligned[i][j] * x_avx_aligned[j])); // 使用 AVX 指令进行减法
        }
        float result[8]; // 用于存储 AVX 寄存器 sum 中的值
        _mm256_store_ps(result, sum); // 将 AVX 寄存器 sum 中的值存入 result 数组中
        float final_sum = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7]; // 对 result 数组中的值进行求和
        x_avx_aligned[i] = final_sum / m_avx_aligned[i][i];
    }
}

void gaussian_elimination_avx_vec_e() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        __m256 mk = _mm256_set1_ps(m_avx_vec_e[k][k]); // 将 m[k][k] 的值扩展到一个256位的 AVX 寄存器 mk 中
        for (int i = k + 1; i < N; i++) {
            float factor = m_avx_vec_e[i][k] / m_avx_vec_e[k][k];
            __m256 factor_vec = _mm256_set1_ps(factor); // 将 factor 的值扩展到一个256位的 AVX 寄存器 factor_vec 中
            int j;
            for (j = k; j < N - 8; j += 8) { // 使用 AVX 指令，每次处理8个元素
                __m256 mi = _mm256_loadu_ps(&m_avx_vec_e[i][j]); // 将 m[i][j]~m[i][j+7] 加载到一个256位的 AVX 寄存器 mi 中
                __m256 mkj = _mm256_loadu_ps(&m_avx_vec_e[k][j]); // 将 m[k][j]~m[k][j+7] 加载到一个256位的 AVX 寄存器 mkj 中
                __m256 result = _mm256_sub_ps(mi, _mm256_mul_ps(factor_vec, mkj)); // 使用 AVX 指令进行减法和乘法
                _mm256_storeu_ps(&m_avx_vec_e[i][j], result); // 将结果存回 m[i][j]~m[i][j+7]
            }
            // 处理剩余的不足8个元素，使用串行方式
            for (; j < N; j++) {
                m_avx_vec_e[i][j] -= factor * m_avx_vec_e[k][j];
            }
            b_avx_vec_e[i] -= factor * b_avx_vec_e[k];
        }
    }

    // 回代过程
    x_avx_vec_e[N - 1] = b_avx_vec_e[N - 1] / m_avx_vec_e[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_avx_vec_e[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m_avx_vec_e[i][j] * x[j];
        }
        x_avx_vec_e[i] = sum / m_avx_vec_e[i][i];
    }

}

void gaussian_elimination_avx_vec_b() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            float factor = m_avx_vec_b[i][k] / m_avx_vec_b[k][k];
            for (int j = k; j < N; j++) {
                m_avx_vec_b[i][j] -= factor * m_avx_vec_b[k][j];
                //if (fabs(m[i][j]) < 1e-6)
                //    m[i][j] = 0; // 小于 10^-6 的数值赋值为 0
            }
            b_avx_vec_b[i] -= factor * b_avx_vec_b[k];
        }
    }
    // 回代过程 - 向量化
    __m256 x_vec[N / 8]; // 创建一个 AVX 向量数组，用于存储回代结果

    // 回代过程
    x_avx_vec_b[N - 1] = b_avx_vec_b[N - 1] / m_avx_vec_b[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        __m256 sum = _mm256_set1_ps(b_avx_vec_b[i]); // 将 b[i] 的值扩展到一个256位的 AVX 寄存器 sum 中
        int j;
        for (j = i + 1; j < N - 8; j += 8) { // 使用 AVX 指令，每次处理8个元素
            __m256 mjxj = _mm256_loadu_ps(&m_avx_vec_b[i][j]); // 将 m[i][j]~m[i][j+7] 加载到一个256位的 AVX 寄存器 mjxj 中
            __m256 xj = _mm256_loadu_ps(&x_avx_vec_b[j]); // 将 x[j]~x[j+7] 加载到一个256位的 AVX 寄存器 xj 中
            sum = _mm256_sub_ps(sum, _mm256_mul_ps(mjxj, xj)); // 使用 AVX 指令进行减法和乘法
        }
        // 处理剩余的不足8个元素，使用串行方式
        for (; j < N; j++) {
            sum = _mm256_sub_ps(sum, _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, m_avx_vec_b[i][j] * x_avx_vec_b[j])); // 使用 AVX 指令进行减法
        }
        float result[8]; // 用于存储 AVX 寄存器 sum 中的值
        _mm256_storeu_ps(result, sum); // 将 AVX 寄存器 sum 中的值存入 result 数组中
        float final_sum = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7]; // 对 result 数组中的值进行求和
        x_avx_vec_b[i] = final_sum / m_avx_vec_b[i][i];
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

void gaussian_elimination_with_pivot_sse_aligned() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        // 选择主元
        int max_index = k;
        float max_value = fabsf(m_pivot_sse_aligned[k][k]);
        for (int i = k + 1; i < N; i++) {
            if (fabsf(m_pivot_sse_aligned[i][k]) > max_value) {
                max_value = fabsf(m_pivot_sse_aligned[i][k]);
                max_index = i;
            }
        }

        // 交换主元所在行与当前行
        if (max_index != k) {
            // 使用 SSE 寄存器进行交换
            __m128i mask = _mm_castps_si128(_mm_cmpeq_ps(_mm_set1_ps(0.0f), _mm_set1_ps(0.0f)));
            for (int j = k; j < N; j += 4) {
                __m128 temp_m0 = _mm_load_ps(&m_pivot_sse_aligned[k][j]);
                __m128 temp_m1 = _mm_load_ps(&m_pivot_sse_aligned[max_index][j]);
                _mm_maskstore_ps(&m_pivot_sse_aligned[k][j], mask, temp_m1);
                _mm_maskstore_ps(&m_pivot_sse_aligned[max_index][j], mask, temp_m0);
            }
            swap(b_pivot_sse_aligned[k], b_pivot_sse_aligned[max_index]);
        }

        // 消元
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_sse_aligned[i][k] / m_pivot_sse_aligned[k][k];
            __m128 factor_vec = _mm_set1_ps(factor);
            __m128* m_i = (__m128*) & m_pivot_sse_aligned[i][k];
            __m128* m_k = (__m128*) & m_pivot_sse_aligned[k][k];
            for (int j = k; j < N; j += 4) {
                __m128 m_kj_vec = _mm_load_ps(&m_pivot_sse_aligned[k][j]);
                __m128 result = _mm_sub_ps(_mm_load_ps(&m_pivot_sse_aligned[i][j]), _mm_mul_ps(factor_vec, m_kj_vec));
                _mm_store_ps(&m_pivot_sse_aligned[i][j], result);
            }
            b_pivot_sse_aligned[i] -= factor * b_pivot_sse_aligned[k];
        }
    }

    // 回代过程
    __m128 x_temp = _mm_set1_ps(0.0f);
    for (int j = N - 1; j >= 0; j--) {
        x_temp = _mm_set1_ps(b_pivot_sse_aligned[j]);
        for (int k = j + 1; k < N; k++) {
            __m128 x_k = _mm_load_ps(&x_pivot_sse_aligned[k]);
            __m128 m_jk = _mm_set1_ps(m_pivot_sse_aligned[j][k]);
            x_temp = _mm_sub_ps(x_temp, _mm_mul_ps(x_k, m_jk));
        }
        x_pivot_sse_aligned[j] = _mm_cvtss_f32(_mm_div_ss(x_temp, _mm_set1_ps(m_pivot_sse_aligned[j][j])));
    }
}

void gaussian_elimination_with_pivot_sse_vec_e() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        // 选择主元
        int max_index = k;
        float max_value = fabsf(m_pivot_sse_vec_e[k][k]);
        for (int i = k + 1; i < N; i++) {
            if (fabsf(m_pivot_sse_vec_e[i][k]) > max_value) {
                max_value = fabsf(m_pivot_sse_vec_e[i][k]);
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
            float factor = m_pivot_sse_vec_e[i][k] / m_pivot_sse_vec_e[k][k];
            __m128 factor_vec = _mm_set1_ps(factor);
            __m128* m_i = (__m128*) & m_pivot_sse_vec_e[i][k];
            __m128* m_k = (__m128*) & m_pivot_sse_vec_e[k][k];
            for (int j = k; j < N; j += 4) {
                __m128 m_kj_vec = _mm_loadu_ps(&m_pivot_sse_vec_e[k][j]);
                __m128 result = _mm_sub_ps(_mm_loadu_ps(&m_pivot_sse_vec_e[i][j]), _mm_mul_ps(factor_vec, m_kj_vec));
                _mm_storeu_ps(&m_pivot_sse_vec_e[i][j], result);
            }
            b_pivot_sse_vec_e[i] -= factor * b_pivot_sse_vec_e[k];
        }

    }

    // 回代过程
    x_pivot_sse_vec_e[N - 1] = b_pivot_sse_vec_e[N - 1] / m_pivot_sse_vec_e[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_pivot_sse_vec_e[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m_pivot_sse_vec_e[i][j] * x_pivot_sse_vec_e[j];
        }
        x_pivot_sse_vec_e[i] = sum / m_pivot_sse_vec_e[i][i];
    }
}

void gaussian_elimination_with_pivot_sse_vec_b() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        // 选择主元
        int max_index = k;
        float max_value = fabsf(m_pivot_sse_vec_b[k][k]);
        for (int i = k + 1; i < N; i++) {
            if (fabsf(m_pivot_sse_vec_b[i][k]) > max_value) {
                max_value = fabsf(m_pivot_sse_vec_b[i][k]);
                max_index = i;
            }
        }

        // 交换主元所在行与当前行
        if (max_index != k) {
            for (int j = k; j < N; j++) {
                swap(m_pivot_sse_vec_b[k][j], m_pivot_sse_vec_b[max_index][j]);
            }
            swap(b_pivot_sse_vec_b[k], b_pivot_sse_vec_b[max_index]);
        }

        // 消元
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_sse_vec_b[i][k] / m_pivot_sse_vec_b[k][k];
            for (int j = k; j < N; j++) {
                m_pivot_sse_vec_b[i][j] -= factor * m_pivot_sse_vec_b[k][j];
            }
            b_pivot_sse_vec_b[i] -= factor * b_pivot_sse_vec_b[k];
        }

    }

    __m128 x_temp = _mm_set1_ps(0.0f);
    for (int j = N - 1; j >= 0; j--) {
        x_temp = _mm_set1_ps(b_pivot_sse_vec_b[j]);
        for (int k = j + 1; k < N; k++) {
            __m128 x_k = _mm_loadu_ps(&x_pivot_sse_vec_b[k]);
            __m128 m_jk = _mm_set1_ps(m_pivot_sse_vec_b[j][k]);
            x_temp = _mm_sub_ps(x_temp, _mm_mul_ps(x_k, m_jk));
        }
        x_pivot_sse_vec_b[j] = _mm_cvtss_f32(_mm_div_ss(x_temp, _mm_set1_ps(m_pivot_sse_vec_b[j][j])));
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

void gaussian_elimination_with_pivot_avx_aligned() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        // 选择主元
        int max_index = k;
        float max_value = fabs(m_pivot_avx_aligned[k][k]);
        for (int i = k + 1; i < N; i++) {
            if (fabs(m_pivot_avx_aligned[i][k]) > max_value) {
                max_value = fabs(m_pivot_avx_aligned[i][k]);
                max_index = i;
            }
        }

        // 交换主元所在行与当前行
        if (max_index != k) {
            __m256 swap_temp;
            for (int j = k; j < N; j += 8) { // 使用 AVX 指令集进行交换
                swap_temp = _mm256_load_ps(&m_pivot_avx_aligned[k][j]);
                _mm256_store_ps(&m_pivot_avx_aligned[k][j], _mm256_load_ps(&m_pivot_avx_aligned[max_index][j]));
                _mm256_store_ps(&m_pivot_avx_aligned[max_index][j], swap_temp);
            }
            swap(b_pivot_avx_aligned[k], b_pivot_avx_aligned[max_index]);
        }

        // 消元
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_avx_aligned[i][k] / m_pivot_avx_aligned[k][k];
            __m256 factor_vec = _mm256_set1_ps(factor);
            __m256* m_i = (__m256*) & m_pivot_avx_aligned[i][k];
            __m256* m_k = (__m256*) & m_pivot_avx_aligned[k][k];
            for (int j = k; j < N; j += 8) { // 使用 AVX 指令集进行消元
                __m256 m_kj_vec = _mm256_load_ps(&m_pivot_avx_aligned[k][j]);
                __m256 result = _mm256_sub_ps(_mm256_load_ps(&m_pivot_avx_aligned[i][j]), _mm256_mul_ps(factor_vec, m_kj_vec));
                _mm256_store_ps(&m_pivot_avx_aligned[i][j], result);
            }
            b_pivot_avx_aligned[i] -= factor * b_pivot_avx_aligned[k];
        }
    }

    // 回代过程
    x_pivot_avx_aligned[N - 1] = b_pivot_avx_aligned[N - 1] / m_pivot_avx_aligned[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        __m256 sum = _mm256_set1_ps(b_pivot_avx_aligned[i]);
        for (int j = i + 1; j < N; j += 8) { // 使用 AVX 指令集进行回代
            __m256 mjxj = _mm256_load_ps(&m_pivot_avx_aligned[i][j]);
            __m256 xj = _mm256_load_ps(&x_pivot_avx_aligned[j]);
            sum = _mm256_sub_ps(sum, _mm256_mul_ps(mjxj, xj));
        }
        float result[8]; // 用于存储 AVX 寄存器 sum 中的值
        _mm256_storeu_ps(result, sum); // 将 AVX 寄存器 sum 中的值存入 result 数组中
        float final_sum = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7]; // 对 result 数组中的值进行求和
        x_pivot_avx_aligned[i] = final_sum / m_pivot_avx_aligned[i][i];
    }
}

void gaussian_elimination_with_pivot_avx_vec_e() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        // 选择主元
        int max_index = k;
        float max_value = fabs(m_pivot_avx_vec_e[k][k]);
        for (int i = k + 1; i < N; i++) {
            if (fabs(m_pivot_avx_vec_e[i][k]) > max_value) {
                max_value = fabs(m_pivot_avx_vec_e[i][k]);
                max_index = i;
            }
        }

        // 交换主元所在行与当前行
        if (max_index != k) {
            __m256 swap_temp;
            for (int j = k; j < N; j += 8) { // 使用 AVX 指令集进行交换
                swap_temp = _mm256_load_ps(&m_pivot_avx_vec_e[k][j]);
                _mm256_store_ps(&m_pivot_avx_vec_e[k][j], _mm256_load_ps(&m_pivot_avx_vec_e[max_index][j]));
                _mm256_store_ps(&m_pivot_avx_vec_e[max_index][j], swap_temp);
            }
            swap(b_pivot_avx_vec_e[k], b_pivot_avx_vec_e[max_index]);
        }

        // 消元
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_avx_vec_e[i][k] / m_pivot_avx_vec_e[k][k];
            __m256 factor_vec = _mm256_set1_ps(factor);
            __m256* m_i = (__m256*) & m_pivot_avx_vec_e[i][k];
            __m256* m_k = (__m256*) & m_pivot_avx_vec_e[k][k];
            for (int j = k; j < N; j += 8) { // 使用 AVX 指令集进行消元
                __m256 m_kj_vec = _mm256_load_ps(&m_pivot_avx_vec_e[k][j]);
                __m256 result = _mm256_sub_ps(_mm256_load_ps(&m_pivot_avx_vec_e[i][j]), _mm256_mul_ps(factor_vec, m_kj_vec));
                _mm256_store_ps(&m_pivot_avx_vec_e[i][j], result);
            }
            b_pivot_avx_vec_e[i] -= factor * b_pivot_avx_vec_e[k];
        }
    }

    // 回代过程
    x_pivot_avx_vec_e[N - 1] = b_pivot_avx_vec_e[N - 1] / m_pivot_avx_vec_e[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--) {
        float sum = b_pivot_avx_vec_e[i];
        for (int j = i + 1; j < N; j++) {
            sum -= m_pivot_avx_vec_e[i][j] * x_pivot_avx_vec_e[j];
        }
        x_pivot_avx_vec_e[i] = sum / m_pivot_avx_vec_e[i][i];
    }
}

void gaussian_elimination_with_pivot_avx_vec_b() {
    // 消去过程
    for (int k = 0; k < N; k++) {
        // 选择主元
        int max_index = k;
        float max_value = fabs(m_pivot_avx_vec_b[k][k]);
        for (int i = k + 1; i < N; i++) {
            if (fabs(m_pivot_avx_vec_b[i][k]) > max_value) {
                max_value = fabs(m_pivot_avx_vec_b[i][k]);
                max_index = i;
            }
        }

        // 交换主元所在行与当前行
        if (max_index != k) {
            for (int j = k; j < N; j++) {
                swap(m_pivot_avx_vec_b[k][j], m_pivot_avx_vec_b[max_index][j]);
            }
            swap(b_pivot_avx_vec_b[k], b_pivot_avx_vec_b[max_index]);
        }

        // 消元
        for (int i = k + 1; i < N; i++) {
            float factor = m_pivot_avx_vec_b[i][k] / m_pivot_avx_vec_b[k][k];
            for (int j = k; j < N; j++) {
                m_pivot_avx_vec_b[i][j] -= factor * m_pivot_avx_vec_b[k][j];
            }
            b_pivot_avx_vec_b[i] -= factor * b_pivot_avx_vec_b[k];
        }
    }

    // 回代过程
    __m256 x_temp = _mm256_set1_ps(0.0f);
    for (int j = N - 1; j >= 0; j--) {
        x_temp = _mm256_set1_ps(b_pivot_avx_vec_b[j]);
        for (int k = j + 1; k < N; k++) {
            __m256 x_k = _mm256_loadu_ps(&x_pivot_avx_vec_b[k]);
            __m256 m_jk = _mm256_set1_ps(m_pivot_avx_vec_b[j][k]);
            x_temp = _mm256_sub_ps(x_temp, _mm256_mul_ps(x_k, m_jk));
        }
        _mm256_storeu_ps(&x_pivot_avx_vec_b[j], _mm256_div_ps(x_temp, _mm256_set1_ps(m_pivot_avx_vec_b[j][j])));
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

    auto start_sse_vec_all = chrono::high_resolution_clock::now();
    gaussian_elimination_sse_vec_all();
    auto end_sse_vec_all = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_sse_vec_all = end_sse_vec_all - start_sse_vec_all;
    float execution_guass_time_sse_vec_all = duration_sse_vec_all.count();
    cout << "Execution GUASS SSE VEC ALL time: " << execution_guass_time_sse_vec_all << " ms" << endl;

    auto start_sse_aligned = chrono::high_resolution_clock::now();
    gaussian_elimination_sse_aligned();
    auto end_sse_aligned = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_sse_aligned = end_sse_aligned - start_sse_aligned;
    float execution_guass_time_sse_aligned = duration_sse_aligned.count();
    cout << "Execution GUASS SSE aligned time: " << execution_guass_time_sse_aligned << " ms" << endl;

    auto start_sse_vec_e = chrono::high_resolution_clock::now();
    gaussian_elimination_sse_vec_e();
    auto end_sse_vec_e = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_sse_vec_e = end_sse_vec_e - start_sse_vec_e;
    float execution_guass_time_sse_vec_e = duration_sse_vec_e.count();
    cout << "Execution GUASS SSE VEC e time: " << execution_guass_time_sse_vec_e << " ms" << endl;

    auto start_sse_vec_b = chrono::high_resolution_clock::now();
    gaussian_elimination_sse_vec_b();
    auto end_sse_vec_b = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_sse_vec_b = end_sse_vec_b - start_sse_vec_b;
    float execution_guass_time_sse_vec_b = duration_sse_vec_b.count();
    cout << "Execution GUASS SSE VEC b time: " << execution_guass_time_sse_vec_b << " ms" << endl;

    auto start_avx_vec_all = chrono::high_resolution_clock::now();
    gaussian_elimination_avx_vec_all();
    auto end_avx_vec_all = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_avx_vec_all = end_avx_vec_all - start_avx_vec_all;
    float execution_guass_time_avx_vec_all = duration_avx_vec_all.count();
    cout << "Execution GUASS AVX VEC ALL time: " << execution_guass_time_avx_vec_all << " ms" << endl;

    auto start_avx_aligned = chrono::high_resolution_clock::now();
    gaussian_elimination_avx_aligned();
    auto end_avx_aligned = std::chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_avx_aligned = end_avx_aligned - start_avx_aligned;
    float execution_guass_time_avx_aligned = duration_avx_aligned.count();
    cout << "Execution GUASS AVX aligned time: " << execution_guass_time_avx_aligned << " ms" << endl;

    auto start_avx_vec_e = chrono::high_resolution_clock::now();
    gaussian_elimination_avx_vec_e();
    auto end_avx_vec_e = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_avx_vec_e = end_avx_vec_e - start_avx_vec_e;
    float execution_guass_time_avx_vec_e = duration_avx_vec_e.count();
    cout << "Execution GUASS AVX VEC e time: " << execution_guass_time_avx_vec_e << " ms" << endl;

    auto start_avx_vec_b = chrono::high_resolution_clock::now();
    gaussian_elimination_avx_vec_b();
    auto end_avx_vec_b = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_avx_vec_b = end_avx_vec_b - start_avx_vec_b;
    float execution_guass_time_avx_vec_b = duration_avx_vec_b.count();
    cout << "Execution GUASS AVX VEC b time: " << execution_guass_time_avx_vec_b << " ms" << endl;

    auto start_pivot = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot();
    auto end_pivot = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot = end_pivot - start_pivot;
    float execution_guass_time_pivot = duration_pivot.count();
    cout << "Execution GUASS PIVOT time: " << execution_guass_time_pivot << " ms" << endl;
    
    auto start_pivot_sse_vec_all = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_sse_vec_all();
    auto end_pivot_sse_vec_all = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_sse_vec_all = end_pivot_sse_vec_all - start_pivot_sse_vec_all;
    float execution_guass_time_pivot_sse_vec_all = duration_pivot_sse_vec_all.count();
    cout << "Execution GUASS PIVOT SSE VEC all time: " << execution_guass_time_pivot_sse_vec_all << " ms" << endl;

    auto start_pivot_sse_aligned = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_sse_aligned();
    auto end_pivot_sse_aligned = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_sse_aligned = end_pivot_sse_aligned - start_pivot_sse_aligned;
    float execution_guass_time_pivot_sse_aligned = duration_pivot_sse_aligned.count();
    cout << "Execution GUASS PIVOT SSE aligned time: " << execution_guass_time_pivot_sse_aligned << " ms" << endl;

    auto start_pivot_sse_vec_e = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_sse_vec_e();
    auto end_pivot_sse_vec_e = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_sse_vec_e = end_pivot_sse_vec_e - start_pivot_sse_vec_e;
    float execution_guass_time_pivot_sse_vec_e = duration_pivot_sse_vec_e.count();
    cout << "Execution GUASS PIVOT SSE VEC e time: " << execution_guass_time_pivot_sse_vec_e << " ms" << endl;

    auto start_pivot_sse_vec_b = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_sse_vec_b();
    auto end_pivot_sse_vec_b = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_sse_vec_b = end_pivot_sse_vec_b - start_pivot_sse_vec_b;
    float execution_guass_time_pivot_sse_vec_b = duration_pivot_sse_vec_b.count();
    cout << "Execution GUASS PIVOT SSE VEC b time: " << execution_guass_time_pivot_sse_vec_b << " ms" << endl;

    auto start_pivot_avx_vec_all = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_avx_vec_all();
    auto end_pivot_avx_vec_all = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_avx_vec_all = end_pivot_avx_vec_all - start_pivot_avx_vec_all;
    float execution_guass_time_pivot_avx_vec_all = duration_pivot_avx_vec_all.count();
    cout << "Execution GUASS PIVOT AVX VEC all time: " << execution_guass_time_pivot_avx_vec_all << " ms" << endl;

    auto start_pivot_avx_aligned = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_avx_aligned();
    auto end_pivot_avx_aligned = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_avx_aligned = end_pivot_avx_aligned - start_pivot_avx_aligned;
    float execution_guass_time_pivot_avx_aligned = duration_pivot_avx_aligned.count();
    cout << "Execution GUASS PIVOT AVX aligned time: " << execution_guass_time_pivot_avx_aligned << " ms" << endl;

    auto start_pivot_avx_vec_e = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_avx_vec_e();
    auto end_pivot_avx_vec_e = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_avx_vec_e = end_pivot_avx_vec_e - start_pivot_avx_vec_e;
    float execution_guass_time_pivot_avx_vec_e = duration_pivot_avx_vec_e.count();
    cout << "Execution GUASS PIVOT AVX VEC e time: " << execution_guass_time_pivot_avx_vec_e << " ms" << endl;

    auto start_pivot_avx_vec_b = chrono::high_resolution_clock::now();
    gaussian_elimination_with_pivot_avx_vec_b();
    auto end_pivot_avx_vec_b = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration_pivot_avx_vec_b = end_pivot_avx_vec_b - start_pivot_avx_vec_b;
    float execution_guass_time_pivot_avx_vec_b = duration_pivot_avx_vec_b.count();
    cout << "Execution GUASS PIVOT AVX VEC b time: " << execution_guass_time_pivot_avx_vec_b << " ms" << endl;

    ////输出结果
    //cout << "Results:" << endl;
    //for (int i = 0; i < N; i++) {
    //    for (int j = 0; j < N; j++) {
    //       // cout << "m[" << i << "]["<<j<<"] = " << m[i][j] << " ";
    //        cout  << m_sse_aligned[i][j] << " ";
    //    }
    // cout << endl;
    //}//检验解决-nan(ind)
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