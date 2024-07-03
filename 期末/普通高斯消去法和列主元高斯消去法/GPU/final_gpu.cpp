//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <windows.h>

using namespace sycl;

static const int N = 4096;
typedef float ele_t;
ele_t mat[N][N];

void guass(ele_t mat[N][N], int n)
{
	LARGE_INTEGER nFreq;
	LARGE_INTEGER nBeginTime;
	LARGE_INTEGER nEndTime;
	double time;
	QueryPerformanceFrequency(&nFreq);
	ele_t(*new_mat)[N] = (ele_t(*)[N])malloc(N * N * sizeof(ele_t));
	memcpy(new_mat, mat, sizeof(ele_t) * N * N);

	QueryPerformanceCounter(&nBeginTime);
	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++)
		{
			if (new_mat[i][i] == 0)
				continue;
			ele_t div = new_mat[j][i] / new_mat[i][i];
			for (int k = i; k < n; k++)
				new_mat[j][k] -= new_mat[i][k] * div;
		}
	QueryPerformanceCounter(&nEndTime);

	time = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;

	std::cout << time << std::endl;

	if (n > 16) return;

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			std::cout << new_mat[i][j] << ' ';
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void guass_gpu(ele_t mat[N][N], int n) {
	LARGE_INTEGER nFreq;
	LARGE_INTEGER nBeginTime;
	LARGE_INTEGER nEndTime;
	double time;
	QueryPerformanceFrequency(&nFreq);
	
	queue q{ cpu_selector{} };

	std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;

	//# 需要使用共享内存
	ele_t(*new_mat)[N] = (ele_t(*)[N])malloc_shared<ele_t>(N * N, q);

	memcpy(new_mat, mat, sizeof(ele_t) * N * N);
	QueryPerformanceCounter(&nBeginTime);

	for (int i = 0; i < n; i++)
		q.parallel_for(range{ n - (i + 1)}, [=](id<1> idx) {
		int j = idx[0] + i + 1;
		ele_t div = new_mat[j][i] / new_mat[i][i];
		for (int k = i; k < n; k++)
			new_mat[j][k] -= new_mat[i][k] * div;
			}).wait();
	QueryPerformanceCounter(&nEndTime);

	time = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;

	std::cout << time << ' ' << std::endl;

	if (n > 16) return;

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			std::cout << new_mat[i][j] << ' ';
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main()
{
	std::ifstream data("gauss.dat", std::ios::in | std::ios::binary);
	data.read((char*)mat, N * N * sizeof(ele_t));
	data.close();

	guass(mat, N);
	guass_gpu(mat, N);

	return 0;
}
