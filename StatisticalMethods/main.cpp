#include <iostream>
#include <fstream>
#include <functional>
#include <vector>
#include <chrono>

#include "vec2.h"
#include "Statistics.h"

using namespace std;

vector<double> C = { 5, 5, 3, 6, 2, 3 };
vector<double> a = { -9, -10, 8, -6, -9, 1 };
vector<double> b = { -3, -1, 8, -5, -3, 5 };


function<double(vec2)> test = [](vec2 x) { return x.x * x.x + x.y * x.y; };

function<double(vec2)> f = [](vec2 x) 
{
	double sum = 0;
	for (int i = 0; i < 6; i++)
		sum += C[i] / (1 + (x.x - a[i]) * (x.x - a[i]) + (x.y - b[i]) * (x.y - b[i]));

	return -sum;
};

int main()
{
	ofstream out("simple.csv");
	vec2 x;

	out << "eps;P;N;(x, y);f(x, y);time" << endl;
	vector<double> eps = { 1, 0.5, 0.1, 0.05, 0.01, 0.005 };
	vector<double> P = { 0.1, 0.3, 0.5, 0.7, 0.9 };

	for (double e : eps)
	{
		for (double p : P)
		{
			x = vec2(0, 0);
			auto start = std::chrono::high_resolution_clock::now();
			uint64_t N = SimpleRandomSearch(f, x, e, p);
			auto end = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double> span = end - start;
			out << e << ";" << p << ";" << N << ";" << x << ";" << f(x) << ";" << span.count() << endl;
		}
	}

	cout << endl << endl;

	vec2 x1, x2, x3;
	vector<int> M = { 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240 };

	ofstream out1("algorithm1.csv");
	ofstream out2("algorithm2.csv");
	ofstream out3("algorithm3.csv");

	out1 << "Количество случайных проб;Количество вычислений функций;Найденная точка;Значение функции в точке" << endl;
	out2 << "Количество случайных проб;Количество вычислений функций;Найденная точка;Значение функции в точке" << endl;
	out3 << "Количество случайных проб;Количество вычислений функций;Найденная точка;Значение функции в точке" << endl;

	for (auto m : M)
	{
		int k1 = Algorithm1(f, x1, m);
		int k2 = Algorithm2(f, x2, m);
		int k3 = Algorithm3(f, x3, m);

		out1 << m << ";" << k1 << ";" << x1 << ";" << f(x1) << endl;
		out2 << m << ";" << k2 << ";" << x2 << ";" << f(x2) << endl;
		out3 << m << ";" << k3 << ";" << x3 << ";" << f(x3) << endl;
	}

	out.close();
	out1.close();
	out2.close();
	out3.close();

	return 0;
}