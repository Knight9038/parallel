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
//����ÿһ������������ڻ�
	//ƽ���㷨
	// ���з��ʾ���Ԫ�أ�һ�����ѭ�����ڴ�ѭ��һ������ִ�У������һ���ڻ����

void trivial_mul() {
	for (int i = 0; i < N ; i++) {
		 sum[i] = 0.0;
		for (int j = 0; j < N ; j++)
			sum[i] += b[j][i] * a[j];
	}
}

//�Ż��㷨
	// ��Ϊ���з��ʾ���Ԫ�أ�һ�����ѭ�����㲻���κ�һ���ڻ���ֻ����ÿ���ڻ��ۼ�һ���˷����
	//���ߵķô�ģʽ�������洢ƥ�䣬���кܺÿռ�ֲ��ԣ��� cache ���õ��Է��ӡ�
void nontrivial_mul() {
	for (int i = 0; i < N ; i++)sum[i] = 0.0;
	for (int j = 0; j < N ; j++)
		for (int i = 0; i < N ; i++)
			sum[i] += b[j][i] * a[j];
}

//n�������
	//ƽ���㷨
	// ��ʽ��������Ԫ�������ۼӵ������������
void trivial_add() {
	for (int i = 0; i < N; i++)
		_sum += a[i];
}

//�Ż��㷨
// ����·ʽ
void nontrivial_add1() {
	float sum1 = 0;
	float sum2 = 0;
	for (int i = 0; i < N; i += 2) {
		sum1 += a[i];
		sum2 += a[i + 1];

	}
	_sum = sum1 + sum2;
}

// ʵ�ַ�ʽ2������ѭ��
void nontrivial_add2() {
	for (int m = N; m > 1; m /= 2) // log(n)������
		for (int i = 0; i < m / 2; i++)
			a[i] = a[i * 2] + a[i * 2 + 1];// ����Ԫ����������洢��������ǰ��
	// a[0]Ϊ���ս��
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
	std::cout << "ƽ���˷�" << " " << "��ģ�� " << N << " " << "�ظ������� " << count << " " << "ʱ�䣺 " << seconds1 << " " << "ƽ��ʱ�䣺" << per1 << std::endl;

	start2 = clock();
	for (int m = 0; m < count; m++) {
		nontrivial_mul();
	}
	finish2 = clock();
	float seconds2 = (finish2 - start2) / float(CLOCKS_PER_SEC);
	float per2 = seconds2 / count;
	std::cout << "�Ż��˷�" << " " << "��ģ�� " << N << " " << "�ظ������� " << count << " " << "ʱ�䣺 " << seconds2 << " " << "ƽ��ʱ�䣺" << per2 << std::endl;

	start3 = clock();
	for (int m = 0; m < count2; m++) {
		trivial_add();
	}
	finish3 = clock();
	float seconds3 = (finish3 - start3) / float(CLOCKS_PER_SEC);
	float per3 = seconds3 / count;
	std::cout << "ƽ���ӷ�" << " " << "��ģ�� " << N << " " << "�ظ������� " << count2 << " " << "ʱ�䣺 " << seconds3 << " " << "ƽ��ʱ�䣺" << per3 << std::endl;


	start4 = clock();
	for (int m = 0; m < count2; m++) {
		nontrivial_add1();
	}
	finish4 = clock();
	float seconds4 = (finish4 - start4) / float(CLOCKS_PER_SEC);
	float per4 = seconds4 / count;
	std::cout << "�Ż��ӷ�1" << " " << "��ģ�� " << N << " " << "�ظ������� " << count2 << " " << "ʱ�䣺 " << seconds4 << " " << "ƽ��ʱ�䣺" << per4 << std::endl;

	start5 = clock();
	for (int m = 0; m < count2; m++) {
		nontrivial_add2();
	}
	finish5 = clock();
	float seconds5 = (finish5 - start5) / float(CLOCKS_PER_SEC);
	float per5 = seconds5 / count;
	std::cout << "�Ż��ӷ�2" << " " << "��ģ�� " << N << " " << "�ظ������� " << count2 << " " << "ʱ�䣺 " << seconds5 << " " << "ƽ��ʱ�䣺" << per5 << std::endl;
	
	return 0;
}
