#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <mpi.h>
#include <arm_neon.h>
#include <omp.h>

using namespace std;

int N = 512;

vector<vector<float>> m(N, vector<float>(N, 0.0));

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

void luDecomposition(vector<vector<float>>& mat, vector<vector<float>>& lower, vector<vector<float>>& upper, int rank, int size) {
    int n = mat.size();
    lower = vector<vector<float>>(n, vector<float>(n, 0.0));
    upper = vector<vector<float>>(n, vector<float>(n, 0.0));

    for (int i = 0; i < n; i++) {
        if (rank == i % size) {
            lower[i][i] = 1.0;

            #pragma omp parallel for
            for (int j = i; j < n; j++) {
                float32x4_t sum_vec = vdupq_n_f32(0.0);
                for (int k = 0; k < i; k += 4) {
                    float32x4_t lower_vec = vld1q_f32(&lower[i][k]);
                    float32x4_t upper_vec = vld1q_f32(&upper[k][j]);
                    sum_vec = vmlaq_f32(sum_vec, lower_vec, upper_vec);
                }
                float sum = vaddvq_f32(sum_vec);
                upper[i][j] = mat[i][j] - sum;
            }
        }

        MPI_Bcast(&upper[i][0], n, MPI_FLOAT, i % size, MPI_COMM_WORLD);

        #pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            if (rank == j % size) {
                float32x4_t sum_vec = vdupq_n_f32(0.0);
                for (int k = 0; k < i; k += 4) {
                    float32x4_t lower_vec = vld1q_f32(&lower[j][k]);
                    float32x4_t upper_vec = vld1q_f32(&upper[k][i]);
                    sum_vec = vmlaq_f32(sum_vec, lower_vec, upper_vec);
                }
                float sum = vaddvq_f32(sum_vec);
                lower[j][i] = (mat[j][i] - sum) / upper[i][i];
            }
        }

        if (rank == i % size) {
            // Sending row i to the next process in the pipeline
            if (rank < size - 1) {
                MPI_Send(&lower[i][0], n, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
            }
        } else if (rank == (i + 1) % size) {
            // Receiving row i from the previous process in the pipeline
            MPI_Recv(&lower[i][0], n, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set the number of OpenMP threads to 8
    omp_set_num_threads(8);

    m_reset();
    int iterations = 1;
    vector<vector<float>> L, U;

    using namespace std::chrono;
    float count = 0;
    for (int i = 0; i < iterations; ++i)
    {
        m_reset();
        auto start = steady_clock::now();
        luDecomposition(m, L, U, rank, size);
        auto end = steady_clock::now();
        duration<float, milli> duration = end - start;
        count += duration.count();
    }

    if (rank == 0) {
        cout << "Parallel (MPI with pipeline, NEON, and OpenMP optimization, 8 threads): " << count << "ms" << endl;
    }

    MPI_Finalize();
    return 0;
}