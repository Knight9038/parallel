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
#include <mpi.h>

using namespace std;
#define NUM_THREADS 4
#define N 1536 // 矩阵大小32 128 256 512 768 1024 1280 1536

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
float m_mpi_async_avx[N][N];
float b_mpi_async_avx[N];
float x_mpi_async_avx[N];
float m_mpi_async_omp_avx[N][N];
float b_mpi_async_omp_avx[N];
float x_mpi_async_omp_avx[N];
float m_mpi_async_sse[N][N];
float b_mpi_async_sse[N];
float x_mpi_async_sse[N];
float m_mpi_async_omp_sse[N][N];
float b_mpi_async_omp_sse[N];
float x_mpi_async_omp_sse[N];


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

    for (int i = 0; i < N; i++) // 初始化 b[N]
        b[i] = rand() % 21;

    // 确保矩阵相同
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            m_mpi[i][j] = m[i][j];
            m_mpi_async[i][j] = m[i][j];
            m_mpi_circle[i][j] = m[i][j];
            m_mpi_circle_b[i][j] = m[i][j];
            m_mpi_async_omp[i][j] = m[i][j];
            m_mpi_async_avx[i][j] = m[i][j];
            m_mpi_async_omp_avx[i][j] = m[i][j];
            m_mpi_async_sse[i][j] = m[i][j];
            m_mpi_async_omp_sse[i][j] = m[i][j];

        }
    }
    for (int i = 0; i < N; i++)
    {
        b_mpi[i] = b[i];
        b_mpi_async[i] = b[i];
        b_mpi_circle[i] = b[i];
        b_mpi_circle_b[i] = b[i];
        b_mpi_async_omp[i] = b[i];
        b_mpi_async_avx[i] = b[i];
        b_mpi_async_omp_avx[i] = b[i];
        b_mpi_async_sse[i] = b[i];
        b_mpi_async_omp_sse[i] = b[i];

    }
}

void gaussian_elimination()
{
    // 消去过程
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

    // 回代过程
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

void gaussian_elimination_mpi(int argc, char* argv[]) {//快划分
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

  

    MPI_Barrier(MPI_COMM_WORLD);  // 确保所有进程已接收数据
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                m_mpi[k][j] = m_mpi[k][j] / m_mpi[k][k];
            }
            m_mpi[k][k] = 1.0;
            for (j = 0; j < total; j++) {
                if (j != rank)
                    MPI_Send(&m_mpi[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);  // 发送第k行到其他进程
            }
        }
        else {
            int src;
            if (k < N / total * total)  // 确定数据源进程
                src = k / (N / total);
            else
                src = total - 1;
            MPI_Recv(&m_mpi[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);  // 从源进程接收第k行
        }
        for (i = (begin > k + 1 ? begin : k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                m_mpi[i][j] = m_mpi[i][j] - m_mpi[i][k] * m_mpi[k][j];
            }
            m_mpi[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);  // 同步所有进程
    if (rank == 0) {  // 0号进程计算并打印耗时
        end_time = MPI_Wtime();
        printf("MPI 块划分：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    /*if (rank == 0)
    {
        cout << "Resultant Matrix (m_mpi):" << endl;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << m_mpi[i][j] << " ";
            }
            cout << endl;
        }
    }*/
    return ;
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
                MPI_Isend(&m_mpi_async[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//非阻塞传递矩阵数据
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); //等待传递

    }
    else {
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m_mpi_async[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //非阻塞接收
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
            MPI_Request* request = new MPI_Request[total - 1 - rank];  //非阻塞传递
            for (j = rank + 1; j < total; j++) { //块划分中，已经消元好且进行了除法置1的行向量仅

                MPI_Isend(&m_mpi_async[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
            }
            MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; //若执行完自身的任务，可直接跳出
        }
        else {
            int src = k / (N / total);
            MPI_Request request;
            MPI_Irecv(&m_mpi_async[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
        }
        for (i = std::max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                m_mpi_async[i][j] = m_mpi_async[i][j] - m_mpi_async[i][k] * m_mpi_async[k][j];
            }
            m_mpi_async[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf(" MPI 块划分 非阻塞耗时：%.4lf ms\n", 1000 * (end_time - start_time));
       /* for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << m_mpi_async[i][j] << " ";
            }
            cout << endl;
        };*/
    }
    MPI_Finalize();
    
    return;
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

    if (rank == 0) {  // 0号进程初始化矩阵
       

        for (j = 1; j < total; j++) {
            for (i = j; i < N; i += total) {
                MPI_Send(&m_mpi_circle[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD); // 1是初始矩阵信息，向每个进程发送数据
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
                    MPI_Send(&m_mpi_circle[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD); // 0号消息表示除法完毕
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
    MPI_Barrier(MPI_COMM_WORLD); // 各进程同步
    if (rank == 0) { // 0号进程中存有最终结果
        end_time = MPI_Wtime();
        printf("MPI 循环划分：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    /*if (rank == 0)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << m_mpi_circle[i][j] << " ";
            }
            cout << endl;
        }
    }*/
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

    if (rank == 0) {  // 0号进程初始化矩阵
        for (j = 1; j < total; j++) {
            for (i = j; i < N; i += total) {
                MPI_Isend(&m_mpi_circle_b[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request); // 非阻塞发送
                MPI_Wait(&request, &status); // 等待发送完成
            }
        }
    }
    else {
        for (i = rank; i < N; i += total) {
            MPI_Irecv(&m_mpi_circle_b[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request); // 非阻塞接收
            MPI_Wait(&request, &status); // 等待接收完成
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
                    MPI_Isend(&m_mpi_circle_b[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request); // 非阻塞发送
                    MPI_Wait(&request, &status); // 等待发送完成
                }
            }
        }
        else {
            int src = k % total;
            MPI_Irecv(&m_mpi_circle_b[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request); // 非阻塞接收
            MPI_Wait(&request, &status); // 等待接收完成
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

    MPI_Barrier(MPI_COMM_WORLD); // 各进程同步

    if (rank == 0) { // 0号进程中存有最终结果
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

    if (rank == 0) {  //0号进程初始化矩阵
        // 假设这里初始化了矩阵 m_mpi_async_omp
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&m_mpi_async_omp[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//非阻塞传递矩阵数据
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); //等待传递

    }
    else {
        // 假设这里初始化了矩阵 m_mpi_async_omp
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m_mpi_async_omp[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //非阻塞接收
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
                    MPI_Request* request = new MPI_Request[total - 1 - rank];  //非阻塞传递
                    for (j = 0; j < total; j++) { //块划分中，已经消元好且进行了除法置1的行向量仅
                        MPI_Isend(&m_mpi_async_omp[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
                    }
                    MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
                }
                else {
                    int src;
                    if (k < N / total * total)//在可均分的任务量内
                        src = k / (N / total);
                    else
                        src = total - 1;
                    MPI_Request request;
                    MPI_Irecv(&m_mpi_async_omp[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
                }
            }
#pragma omp for schedule(guided)  //开始多线程
            for (i = std::max(begin, k + 1); i < end; i++) {
                for (j = k + 1; j < N; j++) {
                    m_mpi_async_omp[i][j] = m_mpi_async_omp[i][j] - m_mpi_async_omp[i][k] * m_mpi_async_omp[k][j];
                }
                m_mpi_async_omp[i][k] = 0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("MPI 块划分 非阻塞 OpenMP：%.4lf ms\n", 1000 * (end_time - start_time));
        /*for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << m_mpi_async_omp[i][j] << " ";
            }
            cout << endl;
        };*/
    }
    MPI_Finalize();
    return;
}
void gaussian_elimination_mpi_async_avx(int argc, char* argv[]) {
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

    float m_mpi_async_avx[N][N];
    float b_mpi_async_avx[N];
    float x_mpi_async_avx[N];

    if (rank == 0) {
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Isend(&m_mpi_async_avx[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE);
    }
    else {
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m_mpi_async_avx[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        {
            if ((begin <= k && k < end)) {
                __m256 t1 = _mm256_set1_ps(m_mpi_async_avx[k][k]);
                for (j = k + 1; j + 8 <= N; j += 8) {
                    __m256 t2 = _mm256_loadu_ps(&m_mpi_async_avx[k][j]);  //AVX优化除法部分
                    t2 = _mm256_div_ps(t2, t1);
                    _mm256_storeu_ps(&m_mpi_async_avx[k][j], t2);
                }
                for (; j < N; j++) {
                    m_mpi_async_avx[k][j] = m_mpi_async_avx[k][j] / m_mpi_async_avx[k][k];
                }
                m_mpi_async_avx[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[total - 1 - rank];  //非阻塞传递
                for (j = 0; j < total; j++) { //块划分中，已经消元好且进行了除法置1的行向量仅
                    MPI_Isend(&m_mpi_async_avx[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
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
                MPI_Irecv(&m_mpi_async_avx[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
            }
        }
        for (i = max(begin, k + 1); i < end; i++) {
            __m256 vik = _mm256_set1_ps(m_mpi_async_avx[i][k]);   //AVX优化消去部分
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&m_mpi_async_avx[k][j]);
                __m256 vij = _mm256_loadu_ps(&m_mpi_async_avx[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&m_mpi_async_avx[i][j], vij);
            }
            for (; j < N; j++) {
                m_mpi_async_avx[i][j] = m_mpi_async_avx[i][j] - m_mpi_async_avx[i][k] * m_mpi_async_avx[k][j];
            }
            m_mpi_async_avx[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);    //各进程同步
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("MPI 块划分 非阻塞 AVX：%.4lf ms\n", 1000 * (end_time - start_time));
       /* for (int i = 0; i < N; i++)
       {
           for (int j = 0; j < N; j++)
           {
               cout << m_mpi_async_avx[i][j] << " ";
           }
           cout << endl;
       };*/
    }
    MPI_Finalize();
    return ;
}
void gaussian_elimination_mpi_async_omp_avx(int argc, char* argv[]) {
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

    float m_mpi_async_omp_avx[N][N];
    float b_mpi_async_omp_avx[N];
    float x_mpi_async_omp_avx[N];

    if (rank == 0) {
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Isend(&m_mpi_async_omp_avx[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);
            }
        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE);
    }
    else {
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m_mpi_async_omp_avx[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);
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
                __m256 t1 = _mm256_set1_ps(m_mpi_async_omp_avx[k][k]);
                for (j = k + 1; j + 8 <= N; j += 8) {
                    __m256 t2 = _mm256_loadu_ps(&m_mpi_async_omp_avx[k][j]);
                    t2 = _mm256_div_ps(t2, t1);
                    _mm256_storeu_ps(&m_mpi_async_omp_avx[k][j], t2);
                }
                for (; j < N; j++) {
                    m_mpi_async_omp_avx[k][j] = m_mpi_async_omp_avx[k][j] / m_mpi_async_omp_avx[k][k];
                }
                m_mpi_async_omp_avx[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[total - 1 - rank];
                for (j = 0; j < total; j++) {
                    MPI_Isend(&m_mpi_async_omp_avx[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);
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
                MPI_Irecv(&m_mpi_async_omp_avx[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
            }
        }
#pragma omp for schedule(guided)
        for (i = max(begin, k + 1); i < end; i++) {
            __m256 vik = _mm256_set1_ps(m_mpi_async_omp_avx[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&m_mpi_async_omp_avx[k][j]);
                __m256 vij = _mm256_loadu_ps(&m_mpi_async_omp_avx[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&m_mpi_async_omp_avx[i][j], vij);
            }
            for (; j < N; j++) {
                m_mpi_async_omp_avx[i][j] = m_mpi_async_omp_avx[i][j] - m_mpi_async_omp_avx[i][k] * m_mpi_async_omp_avx[k][j];
            }
            m_mpi_async_omp_avx[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("MPI 块划分 非阻塞 OpenMP AVX：%.4lf ms\n", 1000 * (end_time - start_time));
        /*for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << m_mpi_async_omp_avx[i][j] << " ";
            }
            cout << endl;
        };*/
    }
    MPI_Finalize();
    return;
}

void gaussian_elimination_mpi_async_sse(int argc, char* argv[]) {
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

    float m_mpi_async_sse[N][N];
    float b_mpi_async_sse[N];
    float x_mpi_async_sse[N];

    if (rank == 0) {
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&m_mpi_async_sse[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // 非阻塞传递矩阵数据
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); // 等待传递
        delete[] request;
    }
    else {
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m_mpi_async_sse[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  // 非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
        delete[] request;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                m_mpi_async_sse[k][j] = m_mpi_async_sse[k][j] / m_mpi_async_sse[k][k];
            }
            m_mpi_async_sse[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[total - 1 - rank];  // 非阻塞传递
            for (j = rank + 1; j < total; j++) { // 块划分中，已经消元好且进行了除法置1的行向量仅

                MPI_Isend(&m_mpi_async_sse[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);// 0号消息表示除法完毕
            }
            MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; // 若执行完自身的任务，可直接跳出
            delete[] request;
        }
        else {
            int src = k / (N / total);
            MPI_Request request;
            MPI_Irecv(&m_mpi_async_sse[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);         // 实际上仍然是阻塞接收，因为接下来的操作需要这些数据
        }
        for (i = std::max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                m_mpi_async_sse[i][j] = m_mpi_async_sse[i][j] - m_mpi_async_sse[i][k] * m_mpi_async_sse[k][j];
            }
            m_mpi_async_sse[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 各进程同步
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("MPI 块划分 非阻塞耗时 SSE：%.4lf ms\n", 1000 * (end_time - start_time));
         /*for (int i = 0; i < N; i++)
         {
             for (int j = 0; j < N; j++)
             {
                 cout << m_mpi_async_sse[i][j] << " ";
             }
             cout << endl;
         };*/
    }
    MPI_Finalize();

    return;
}
void gaussian_elimination_mpi_async_omp_sse(int argc, char* argv[]) {
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

    float m_mpi_async_omp_sse[N][N];
    float b_mpi_async_omp_sse[N];
    float x_mpi_async_omp_sse[N];

    if (rank == 0) {
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&m_mpi_async_omp_sse[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]); // 非阻塞传递矩阵数据
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); // 等待传递
        delete[] request;
    }
    else {
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&m_mpi_async_omp_sse[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  // 非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);
        delete[] request;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
#pragma omp parallel num_threads(NUM_THREADS) private(i,j,k)
    for (k = 0; k < N; k++) {
#pragma omp single
        {
            if ((begin <= k && k < end)) {
                for (j = k + 1; j < N; j++) {
                    m_mpi_async_omp_sse[k][j] = m_mpi_async_omp_sse[k][j] / m_mpi_async_omp_sse[k][k];
                }
                m_mpi_async_omp_sse[k][k] = 1.0;
                MPI_Request* request = new MPI_Request[total - 1 - rank];  // 非阻塞传递
                for (j = rank + 1; j < total; j++) { // 块划分中，已经消元好且进行了除法置1的行向量仅

                    MPI_Isend(&m_mpi_async_omp_sse[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);// 0号消息表示除法完毕
                }
                MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
                if (k == end - 1)
                    break; // 若执行完自身的任务，可直接跳出
                delete[] request;
            }
            else {
                int src = k / (N / total);
                MPI_Request request;
                MPI_Irecv(&m_mpi_async_omp_sse[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);         // 实际上仍然是阻塞接收，因为接下来的操作需要这些数据
            }
        }
#pragma omp for schedule(guided)
        for (i = std::max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                m_mpi_async_omp_sse[i][j] = m_mpi_async_omp_sse[i][j] - m_mpi_async_omp_sse[i][k] * m_mpi_async_omp_sse[k][j];
            }
            m_mpi_async_omp_sse[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); // 各进程同步
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("MPI 块划分 非阻塞 openMP SSE：%.4lf ms\n", 1000 * (end_time - start_time));
        /*for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << m_mpi_async_omp_sse[i][j] << " ";
            }
            cout << endl;
        };*/
    }
    MPI_Finalize();

    return;
}
int main(int argc, char* argv[])
{
    
    m_reset();

    /*auto start = chrono::high_resolution_clock::now();
    gaussian_elimination();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration = end - start;
    float execution_guass_time = duration.count();
    cout << "Execution GAUSS time: " << execution_guass_time << " ms" << endl;*/

    /*gaussian_elimination_mpi(argc, argv);*/
    /*gaussian_elimination_mpi_async(argc, argv);*/
    /*gaussian_elimination_mpi_circle(argc, argv);*/
    /*gaussian_elimination_mpi_circle_b(argc, argv);*/
    /*gaussian_elimination_mpi_async_omp(argc, argv);*/
    /*gaussian_elimination_mpi_async_omp_avx(argc, argv);*/
    /*gaussian_elimination_mpi_async_avx(argc, argv);*/
    /*gaussian_elimination_mpi_async_sse(argc, argv);*/
    /*gaussian_elimination_mpi_async_omp_sse(argc, argv);*/
    return 0;
}