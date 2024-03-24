#include<iostream>
#include <random> 
#include <cstdlib>
#include <ctime>
#define N 100

float b[N][N];
float a[N];
float sum[N];
float _sum;

void ini() {
	unsigned seed = static_cast<unsigned>(std::time(nullptr));
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<double> distribution(-100.0, 100.0);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			b[i][j] = distribution(generator);
		}
	}
	for (int i = 0; i < N; i++) {
		a[i] = distribution(generator);
	}
}
//矩阵每一列与给定向量内积
	//平凡算法
	// 逐列访问矩阵元素：一步外层循环（内存循环一次完整执行）计算出一个内积结果

void trivial_mul() {
	for (int i = 0; i < N ; i++) {
		 sum[i] = 0.0;
		for (int j = 0; j < N ; j++)
			sum[i] += b[j][i] * a[j];
	}
}

//优化算法
	// 改为逐行访问矩阵元素：一步外层循环计算不出任何一个内积，只是向每个内积累加一个乘法结果
	//后者的访存模式与行主存储匹配，具有很好空间局部性，令 cache 作用得以发挥。
void nontrivial_mul() {
	for (int i = 0; i < N ; i++)sum[i] = 0.0;
	for (int j = 0; j < N ; j++)
		for (int i = 0; i < N ; i++)
			sum[i] += b[j][i] * a[j];
}

//n个数求和
	//平凡算法
	// 链式：将给定元素依次累加到结果变量即可
void trivial_add() {
	for (int i = 0; i < N; i++)
		_sum += a[i];
}

//优化算法
// 多链路式
void nontrivial_add1() {
	float sum1 = 0;
	float sum2 = 0;
	for (int i = 0; i < N; i += 2) {
		sum1 += a[i];
		sum2 += a[i + 1];

	}
	_sum = sum1 + sum2;
}

// 实现方式2：二重循环
void nontrivial_add2() {
	for (int m = N; m > 1; m /= 2) // log(n)个步骤
		for (int i = 0; i < m / 2; i++)
			a[i] = a[i * 2] + a[i * 2 + 1];// 相邻元素相加连续存储到数组最前面
	// a[0]为最终结果
}

int main() {
	ini();
	clock_t start1, finish1, start2, finish2, start3, finish3, start4, finish4, start5, finish5;
	int count = 50000;
	int count2 = 100000;
	start1 = clock();
	for (int m = 0; m < count; m++) {
		trivial_mul();
	}
	finish1 = clock();
	float seconds1 = (finish1 - start1) / float(CLOCKS_PER_SEC);
	float per1 = seconds1 / count;
	std::cout << "平凡乘法" << " " << "规模： " << N << " " << "重复次数： " << count << " " << "时间： " << seconds1 << " " << "平均时间：" << per1 << std::endl;

	start2 = clock();
	for (int m = 0; m < count; m++) {
		nontrivial_mul();
	}
	finish2 = clock();
	float seconds2 = (finish2 - start2) / float(CLOCKS_PER_SEC);
	float per2 = seconds2 / count;
	std::cout << "优化乘法" << " " << "规模： " << N << " " << "重复次数： " << count << " " << "时间： " << seconds2 << " " << "平均时间：" << per2 << std::endl;

	start3 = clock();
	for (int m = 0; m < count2; m++) {
		trivial_add();
	}
	finish3 = clock();
	float seconds3 = (finish3 - start3) / float(CLOCKS_PER_SEC);
	float per3 = seconds3 / count;
	std::cout << "平凡加法" << " " << "规模： " << N << " " << "重复次数： " << count2 << " " << "时间： " << seconds3 << " " << "平均时间：" << per3 << std::endl;


	start4 = clock();
	for (int m = 0; m < count2; m++) {
		nontrivial_add1();
	}
	finish4 = clock();
	float seconds4 = (finish4 - start4) / float(CLOCKS_PER_SEC);
	float per4 = seconds4 / count;
	std::cout << "优化加法1" << " " << "规模： " << N << " " << "重复次数： " << count2 << " " << "时间： " << seconds4 << " " << "平均时间：" << per4 << std::endl;

	start5 = clock();
	for (int m = 0; m < count2; m++) {
		nontrivial_add2();
	}
	finish5 = clock();
	float seconds5 = (finish5 - start5) / float(CLOCKS_PER_SEC);
	float per5 = seconds5 / count;
	std::cout << "优化加法2" << " " << "规模： " << N << " " << "重复次数： " << count2 << " " << "时间： " << seconds5 << " " << "平均时间：" << per5 << std::endl;
	
	return 0;
}
