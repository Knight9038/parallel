#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <pthread.h>
using namespace std;

int N = 2048;
const int NUM_THREADS = 8;

vector<vector<double>> m(N, vector<double>(N, 0.0)); // Global matrix m
vector<vector<double>> L(N, vector<double>(N, 0.0)); // Lower matrix
vector<vector<double>> U(N, vector<double>(N, 0.0)); // Upper matrix

void printMatrix(const vector<vector<double>>& mat) {
    for (const auto& row : mat) {
        for (double val : row) {
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
            m[i][j] = rand() % RAND_MAX + 2;
            m[j][i] = rand() % RAND_MAX + 2;
        }
    }
}

// Struct to hold thread arguments
struct ThreadArgs {
    int startRow;
    int endRow;
};

// LU decomposition function for each thread
void* threadCroutLuDecomposition(void* args) {
    ThreadArgs* threadArgs = (ThreadArgs*)args;
    int startRow = threadArgs->startRow;
    int endRow = threadArgs->endRow;

    for (int i = startRow; i < endRow; i++) {
        L[i][i] = 1.0;

        for (int j = i; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += L[i][k] * U[k][j];
            }
            U[i][j] = m[i][j] - sum;
        }

        for (int j = i + 1; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += L[j][k] * U[k][i];
            }
            L[j][i] = (m[j][i] - sum) / U[i][i];
        }
    }

    pthread_exit(NULL);
}

int main() {
    m_reset();
    int iterations = 1;

    using namespace std::chrono;
    double count = 0;

    pthread_t threads[NUM_THREADS];
    ThreadArgs threadArgs[NUM_THREADS];

    for (int i = 0; i < iterations; ++i)
    {
        m_reset();
        auto start = steady_clock::now();

        // Create threads
        for (int t = 0; t < NUM_THREADS; ++t) {
            int chunkSize = N / NUM_THREADS;
            threadArgs[t].startRow = t * chunkSize;
            threadArgs[t].endRow = (t + 1) * chunkSize;
            pthread_create(&threads[t], NULL, threadCroutLuDecomposition, (void*)&threadArgs[t]);
        }

        // Join threads
        for (int t = 0; t < NUM_THREADS; ++t) {
            pthread_join(threads[t], NULL);
        }

        auto end = steady_clock::now();
        duration<double, milli> duration = end - start;
        count += duration.count();
    }
    cout << "Crout-LU with " << NUM_THREADS << " threads: " << count << "ms" << endl;

    /*cout << "Lower triangular matrix L:" << endl;
    printMatrix(L);

    cout << "Upper triangular matrix U:" << endl;
    printMatrix(U);*/

    return 0;
}