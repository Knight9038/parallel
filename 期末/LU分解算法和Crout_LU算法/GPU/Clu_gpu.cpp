#include <iostream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <CL/sycl.hpp>

using namespace std;
using namespace sycl;

const int N = 4096;

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

void luDecomposition(queue& q, vector<vector<float>>& mat, vector<vector<float>>& lower, vector<vector<float>>& upper) {
    buffer<float, 2> mat_buf(mat.data(), range<2>(N, N));
    buffer<float, 2> lower_buf(lower.data(), range<2>(N, N));
    buffer<float, 2> upper_buf(upper.data(), range<2>(N, N));

    q.submit([&](handler& h) {
        auto mat_acc = mat_buf.get_access<access::mode::read>(h);
        auto lower_acc = lower_buf.get_access<access::mode::write>(h);
        auto upper_acc = upper_buf.get_access<access::mode::write>(h);

        h.parallel_for(range<2>(N, N), [=](id<2> idx) {
            int i = idx[0];
            int j = idx[1];
            if (i == j) lower_acc[i][i] = 1.0;
        });
    });

    for (int i = 0; i < N; i++) {
        q.submit([&](handler& h) {
            auto mat_acc = mat_buf.get_access<access::mode::read>(h);
            auto lower_acc = lower_buf.get_access<access::mode::read_write>(h);
            auto upper_acc = upper_buf.get_access<access::mode::read_write>(h);

            h.parallel_for(range<1>(N-i), [=](id<1> idx) {
                int j = idx[0] + i;
                float sum = 0.0;
                for (int k = 0; k < i; k++) {
                    sum += lower_acc[i][k] * upper_acc[k][j];
                }
                upper_acc[i][j] = mat_acc[i][j] - sum;
            });
        });

        q.submit([&](handler& h) {
            auto mat_acc = mat_buf.get_access<access::mode::read>(h);
            auto lower_acc = lower_buf.get_access<access::mode::read_write>(h);
            auto upper_acc = upper_buf.get_access<access::mode::read>(h);

            h.parallel_for(range<1>(N-i-1), [=](id<1> idx) {
                int j = idx[0] + i + 1;
                float sum = 0.0;
                for (int k = 0; k < i; k++) {
                    sum += lower_acc[j][k] * upper_acc[k][i];
                }
                lower_acc[j][i] = (mat_acc[j][i] - sum) / upper_acc[i][i];
            });
        });
    }
}

int main() {
    m_reset();
    int iterations = 1;
    vector<vector<float>> L(N, vector<float>(N, 0.0));
    vector<vector<float>> U(N, vector<float>(N, 0.0));

    queue q;
    float count = 0;
    for (int i = 0; i < iterations; ++i)
    {
        m_reset();
        auto start = chrono::steady_clock::now();
        luDecomposition(q, m, L, U);
        q.wait(); // Ensure all tasks are completed
        auto end = chrono::steady_clock::now();
        chrono::duration<float, milli> duration = end - start;
        count += duration.count();
    }
    cout << "Normal: " << count << "ms" << endl;

    return 0;
}