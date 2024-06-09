#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <immintrin.h>
#include <stddef.h>
#include <stdlib.h>
#include <malloc.h>
#include <pthread.h>
#include <thread>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <cstdlib> 
#include <semaphore.h>
#include <mpi.h>

using namespace std;
#define NUM_THREADS 4
#define N 32 

float m[N][N];
float b[N];
float x[N];
float m_mpi[N][N];
float b_mpi[N];
float x_mpi[N];
float m_mpi_async[N][N];
float b_mpi_async[N];
float x_mpi_async[N];
float m_mpi_circle[N][N];
float b_mpi_circle[N];
float x_mpi_circle[N];
float m_mpi_circle_b[N][N];
float b_mpi_circle_b[N];
float x_mpi_circle_b[N];
float m_mpi_async_omp[N][N];
float b_mpi_async_omp[N];
float x_mpi_async_omp[N];
float m_mpi_async_neon[N][N];
float b_mpi_async_neon[N];
float x_mpi_async_neon[N];
float m_mpi_async_omp_neon[N][N];
float b_mpi_async_omp_neon[N];
float x_mpi_async_omp_neon[N];


void m_reset()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
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


    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            m_mpi[i][j] = m[i][j];
            m_mpi_async[i][j] = m[i][j];
            m_mpi_circle[i][j] = m[i][j];
            m_mpi_circle_b[i][j] = m[i][j];
            m_mpi_async_omp[i][j] = m[i][j];
            m_mpi_async_neon[i][j] = m[i][j];
            m_mpi_async_omp_neon[i][j] = m[i][j];


        }
    }
    for (int i = 0; i < N; i++)
    {
        b_mpi[i] = b[i];
        b_mpi_async[i] = b[i];
        b_mpi_circle[i] = b[i];
        b_mpi_circle_b[i] = b[i];
        b_mpi_async_omp[i] = b[i];
        b_mpi_async_neon[i] = b[i];
        b_mpi_async_omp_neon[i] = b[i];


    }
}

void gaussian_elimination()
{

    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            float factor = m[i][k] / m[k][k];
            for (int j = k; j < N; j++)
            {
                m[i][j] -= factor * m[k][j];
            }
            b[i] -= factor * b[k];
        }
    }



    x[N - 1] = b[N - 1] / m[N - 1][N - 1];
    for (int i = N - 2; i >= 0; i--)
    {
        float sum = b[i];
        for (int j = i + 1; j < N; j++)
        {
            sum -= m[i][j] * x[j];
        }
        x[i] = sum / m[i][i];
    }
}

void gaussian_elimination_mpi(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);



    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                m_mpi[k][j] = m_mpi[k][j] / m_mpi[k][k];
            }
            m_mpi[k][k] = 1.0;
            for (j = 0; j < total; j++) {
                if (j != rank)
                    MPI_Send(&m_mpi[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
        }
        else {
            int src;
            if (k < N / total * total)
                src = k / (N / total);
            else
                src = total - 1;
            MPI_Recv(&m_mpi[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (i = (begin > k + 1 ? begin : k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                m_mpi[i][j] = m_mpi[i][j] - m_mpi[i][k] * m_mpi[k][j];
            }
            m_mpi[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("MPI ：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();

    return;
}

void gaussian_elimination_mpi_async(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    if (rank == 0) {
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&m_mpi_async[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE);

    }
    else {
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m_mpi_async[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                m_mpi_async[k][j] = m_mpi_async[k][j] / m_mpi_async[k][k];
            }
            m_mpi_async[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[total - 1 - rank];
            for (j = rank + 1; j < total; j++) {

                MPI_Isend(&m_mpi_async[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);
            }
            MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break;
        }
        else {
            int src = k / (N / total);
            MPI_Request request;
            MPI_Irecv(&m_mpi_async[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
        for (i = std::max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                m_mpi_async[i][j] = m_mpi_async[i][j] - m_mpi_async[i][k] * m_mpi_async[k][j];
            }
            m_mpi_async[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("MPI 块划分 非阻塞：%.4lf ms\n", 1000 * (end_time - start_time));

    }
    MPI_Finalize();

    return;
}
void gaussian_elimination_mpi_async_neon_c(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    // 进程0发送矩阵数据给其他进程
    if (rank == 0) {
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&m_mpi_async[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE);
    }
    else {
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m_mpi_async[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            // NEON优化的除法步骤
            float32x4_t v_inv = vdupq_n_f32(1.0f / m_mpi_async[k][k]);
            for (j = k + 1; j < N; j += 4) {
                float32x4_t v_data = vld1q_f32(&m_mpi_async[k][j]);
                v_data = vmulq_f32(v_data, v_inv);
                vst1q_f32(&m_mpi_async[k][j], v_data);
            }
            for (; j < N; j++) {
                m_mpi_async[k][j] = m_mpi_async[k][j] / m_mpi_async[k][k];
            }
            m_mpi_async[k][k] = 1.0;

            // 非阻塞发送
            MPI_Request* request = new MPI_Request[total - 1 - rank];
            for (j = rank + 1; j < total; j++) {
                MPI_Isend(&m_mpi_async[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);
            }
            MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break;
        }
        else {
            int src = k / (N / total);
            MPI_Request request;
            MPI_Irecv(&m_mpi_async[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }

        // NEON优化的减法步骤
        for (i = std::max(begin, k + 1); i < end; i++) {
            float32x4_t v_k_row = vld1q_f32(&m_mpi_async[k][k + 1]);
            for (j = k + 1; j < N; j += 4) {
                float32x4_t v_i_row = vld1q_f32(&m_mpi_async[i][j]);
                float32x4_t v_factor = vdupq_n_f32(m_mpi_async[i][k]);
                v_i_row = vmlsq_f32(v_i_row, v_k_row, v_factor);
                vst1q_f32(&m_mpi_async[i][j], v_i_row);
            }
            for (; j < N; j++) {
                m_mpi_async[i][j] = m_mpi_async[i][j] - m_mpi_async[i][k] * m_mpi_async[k][j];
            }
            m_mpi_async[i][k] = 0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("MPI 块划分 非阻塞：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
}

void gaussian_elimination_mpi_circle(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {


        for (j = 1; j < total; j++) {
            for (i = j; i < N; i += total) {
                MPI_Send(&m_mpi_circle[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);
            }
        }
    }
    else {
        for (i = rank; i < N; i += total) {
            MPI_Recv(&m_mpi_circle[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if (k % total == rank) {
            for (j = k + 1; j < N; j++) {
                m_mpi_circle[k][j] = m_mpi_circle[k][j] / m_mpi_circle[k][k];
            }
            m_mpi_circle[k][k] = 1.0;
            for (j = 0; j < total; j++) {
                if (j != rank)
                    MPI_Send(&m_mpi_circle[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
        }
        else {
            int src = k % total;
            MPI_Recv(&m_mpi_circle[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        int begin = k;
        while (begin % total != rank)
            begin++;
        for (i = begin; i < N; i += total) {
            for (j = k + 1; j < N; j++) {
                m_mpi_circle[i][j] = m_mpi_circle[i][j] - m_mpi_circle[i][k] * m_mpi_circle[k][j];
            }
            m_mpi_circle[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("MPI 循环划分：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();

}
void gaussian_elimination_mpi_circle_b(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Request request;

    if (rank == 0) { 
        for (j = 1; j < total; j++) {
            for (i = j; i < N; i += total) {
                MPI_Isend(&m_mpi_circle_b[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request); 
                MPI_Wait(&request, &status); 
            }
        }
    }
    else {
        for (i = rank; i < N; i += total) {
            MPI_Irecv(&m_mpi_circle_b[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request); 
            MPI_Wait(&request, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    for (k = 0; k < N; k++) {
        if (k % total == rank) {
            for (j = k + 1; j < N; j++) {
                m_mpi_circle_b[k][j] = m_mpi_circle_b[k][j] / m_mpi_circle_b[k][k];
            }
            m_mpi_circle_b[k][k] = 1.0;
            for (j = 0; j < total; j++) {
                if (j != rank) {
                    MPI_Isend(&m_mpi_circle_b[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request); 
                    MPI_Wait(&request, &status); 
                }
            }
        }
        else {
            int src = k % total;
            MPI_Irecv(&m_mpi_circle_b[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
        }
        int begin = k;
        while (begin % total != rank)
            begin++;
        for (i = begin; i < N; i += total) {
            for (j = k + 1; j < N; j++) {
                m_mpi_circle_b[i][j] = m_mpi_circle_b[i][j] - m_mpi_circle_b[i][k] * m_mpi_circle_b[k][j];
            }
            m_mpi_circle_b[i][k] = 0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    if (rank == 0) { 
        end_time = MPI_Wtime();
        printf("MPI 循环划分 非阻塞：%.4lf ms\n", 1000 * (end_time - start_time));
    }

    MPI_Finalize();
}
void gaussian_elimination_mpi_async_omp(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    if (rank == 0) {
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&m_mpi_async_omp[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE);

    }
    else {
        
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m_mpi_async_omp[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
#pragma omp parallel num_threads(NUM_THREADS) private(i, j, k)
    {
        for (k = 0; k < N; k++) {
#pragma omp single
            {
                if ((begin <= k && k < end)) {
                    for (j = k + 1; j < N; j++) {
                        m_mpi_async_omp[k][j] = m_mpi_async_omp[k][j] / m_mpi_async_omp[k][k];
                    }
                    m_mpi_async_omp[k][k] = 1.0;
                    MPI_Request* request = new MPI_Request[total - 1 - rank];
                    for (j = 0; j < total; j++) {
                        MPI_Isend(&m_mpi_async_omp[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);
                    }
                    MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
                }
                else {
                    int src;
                    if (k < N / total * total)
                        src = k / (N / total);
                    else
                        src = total - 1;
                    MPI_Request request;
                    MPI_Irecv(&m_mpi_async_omp[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUS_IGNORE);
                }
            }
#pragma omp for schedule(guided)  
            for (i = std::max(begin, k + 1); i < end; i++) {
                for (j = k + 1; j < N; j++) {
                    m_mpi_async_omp[i][j] = m_mpi_async_omp[i][j] - m_mpi_async_omp[i][k] * m_mpi_async_omp[k][j];
                }
                m_mpi_async_omp[i][k] = 0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("MPI 块划分 非阻塞 OpenMP：%.4lf ms\n", 1000 * (end_time - start_time));

    }
    MPI_Finalize();
    return;
}
void gaussian_elimination_mpi_async_neon(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    MPI_Wtick();
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    float m_mpi_async_neon[N][N];
    float b_mpi_async_neon[N];
    float x_mpi_async_neon[N];

    if (rank == 0) {
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Isend(&m_mpi_async_neon[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE);
    }
    else {
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m_mpi_async_neon[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        {
            if ((begin <= k && k < end)) {
                float t1 = m_mpi_async_neon[k][k];
                for (j = k + 1; j < N; j += 4) {
                    float32x4_t t2 = vld1q_f32(&m_mpi_async_neon[k][j]);
                    t2 = vdivq_f32(t2, vdupq_n_f32(t1));
                    vst1q_f32(&m_mpi_async_neon[k][j], t2);
                }
                for (; j < N; j++) {
                    m_mpi_async_neon[k][j] = m_mpi_async_neon[k][j] / m_mpi_async_neon[k][k];
                }
                m_mpi_async_neon[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[total - 1 - rank];
                for (j = 0; j < total; j++) {
                    MPI_Isend(&m_mpi_async_neon[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);
                }
                MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < N / total * total)
                    src = k / (N / total);
                else
                    src = total - 1;
                MPI_Request request;
                MPI_Irecv(&m_mpi_async_neon[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
        }
        for (i = max(begin, k + 1); i < end; i++) {
            float32x4_t vik = vdupq_n_f32(m_mpi_async_neon[i][k]);
            for (j = k + 1; j < N; j += 4) {
                float32x4_t vkj = vld1q_f32(&m_mpi_async_neon[k][j]);
                float32x4_t vij = vld1q_f32(&m_mpi_async_neon[i][j]);
                float32x4_t vx = vmulq_f32(vik, vkj);
                vij = vsubq_f32(vij, vx);
                vst1q_f32(&m_mpi_async_neon[i][j], vij);
            }
            for (; j < N; j++) {
                m_mpi_async_neon[i][j] = m_mpi_async_neon[i][j] - m_mpi_async_neon[i][k] * m_mpi_async_neon[k][j];
            }
            m_mpi_async_neon[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("MPI 块划分 非阻塞 NEON耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    return;
}
void gaussian_elimination_mpi_async_omp_neon(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    MPI_Wtick();
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    float m_mpi_async_omp_neon[N][N];
    float b_mpi_async_omp_neon[N];
    float x_mpi_async_omp_neon[N];

    if (rank == 0) {
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Isend(&m_mpi_async_omp_neon[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE);
    }
    else {
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m_mpi_async_omp_neon[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            if ((begin <= k && k < end)) {
                float t1 = m_mpi_async_omp_neon[k][k];
                for (j = k + 1; j < N; j += 4) {
                    float32x4_t t2 = vld1q_f32(&m_mpi_async_omp_neon[k][j]);
                    t2 = vdivq_f32(t2, vdupq_n_f32(t1));
                    vst1q_f32(&m_mpi_async_omp_neon[k][j], t2);
                }
                for (; j < N; j++) {
                    m_mpi_async_omp_neon[k][j] = m_mpi_async_omp_neon[k][j] / m_mpi_async_omp_neon[k][k];
                }
                m_mpi_async_omp_neon[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[total - 1 - rank];
                for (j = 0; j < total; j++) {
                    MPI_Isend(&m_mpi_async_omp_neon[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);
                }
                MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            }
            else {
                int src;
                if (k < N / total * total)
                    src = k / (N / total);
                else
                    src = total - 1;
                MPI_Request request;
                MPI_Irecv(&m_mpi_async_omp_neon[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
        }
#pragma omp for schedule(guided)
        for (i = max(begin, k + 1); i < end; i++) {
            float32x4_t vik = vdupq_n_f32(m_mpi_async_omp_neon[i][k]);
            for (j = k + 1; j < N; j += 4) {
                float32x4_t vkj = vld1q_f32(&m_mpi_async_omp_neon[k][j]);
                float32x4_t vij = vld1q_f32(&m_mpi_async_omp_neon[i][j]);
                float32x4_t vx = vmulq_f32(vik, vkj);
                vij = vsubq_f32(vij, vx);
                vst1q_f32(&m_mpi_async_omp_neon[i][j], vij);
            }
            for (; j < N; j++) {
                m_mpi_async_omp_neon[i][j] = m_mpi_async_omp_neon[i][j] - m_mpi_async_omp_neon[i][k] * m_mpi_async_omp_neon[k][j];
            }
            m_mpi_async_omp_neon[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("MPI 块划分 非阻塞 OpenMP NEON ：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    return;
}
void gaussian_elimination_mpi_async_neon_c(int argc, char* argv[]) {
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    if (rank == 0) {
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Isend(&m_mpi_async[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE);
    }
    else {
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m_mpi_async[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            float32x4_t v_inv = vdupq_n_f32(1.0f / m_mpi_async[k][k]);
            for (j = k + 1; j < N; j += 4) {
                float32x4_t v_data = vld1q_f32(&m_mpi_async[k][j]);
                v_data = vmulq_f32(v_data, v_inv);
                vst1q_f32(&m_mpi_async[k][j], v_data);
            }
            for (; j < N; j++) {
                m_mpi_async[k][j] = m_mpi_async[k][j] / m_mpi_async[k][k];
            }
            m_mpi_async[k][k] = 1.0;

            MPI_Request* request = new MPI_Request[total - 1 - rank];
            for (j = rank + 1; j < total; j++) {
                MPI_Isend(&m_mpi_async[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);
            }
            MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break;
        }
        else {
            int src = k / (N / total);
            MPI_Request request;
            MPI_Irecv(&m_mpi_async[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }

        for (i = std::max(begin, k + 1); i < end; i++) {
            float32x4_t v_k_row = vld1q_f32(&m_mpi_async[k][k + 1]);
            for (j = k + 1; j < N; j += 4) {
                float32x4_t v_i_row = vld1q_f32(&m_mpi_async[i][j]);
                float32x4_t v_factor = vdupq_n_f32(m_mpi_async[i][k]);
                v_i_row = vmlsq_f32(v_i_row, v_k_row, v_factor);
                vst1q_f32(&m_mpi_async[i][j], v_i_row);
            }
            for (; j < N; j++) {
                m_mpi_async[i][j] = m_mpi_async[i][j] - m_mpi_async[i][k] * m_mpi_async[k][j];
            }
            m_mpi_async[i][k] = 0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("MPI 块划分 非阻塞：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
}

int main(int argc, char* argv[]) {
    gaussian_elimination_mpi_async(argc, argv);
    return 0;
}

int main(int argc, char* argv[])
{

    m_reset();

    /*gaussian_elimination_mpi(argc, argv);*/
    /*gaussian_elimination_mpi_async(argc, argv);*/
    /*gaussian_elimination_mpi_circle(argc, argv);*/
    /*gaussian_elimination_mpi_circle_b(argc, argv);*/
    /*gaussian_elimination_mpi_async_omp(argc, argv);*/
    /*gaussian_elimination_mpi_async_omp_neon_c(argc, argv);*/
    /*gaussian_elimination_mpi_async_neon(argc, argv);*/
    

    return 0;
}

