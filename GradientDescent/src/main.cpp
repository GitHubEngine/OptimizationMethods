#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include "Matrix.h"
#include "Newton.h"
#include "Broyden.h"

func f = [](vec2 x) { return 100.0 * (x.y - x.x) * (x.y - x.x) + (1.0 - x.x) * (1.0 - x.x); };

grad g =
{
	[](vec2 x) { return -200.0 * (x.y - x.x) - 2.0 * (1.0 - x.x); },
	[](vec2 x) { return 200.0 * (x.y - x.x); }
};

hesse h =
{
	[](vec2 x) { return 202.0; },
	[](vec2 x) { return -200.0; },
	[](vec2 x) { return -200.0; },
	[](vec2 x) { return 200.0; }
};

func f1 = [](vec2 x) { return 100.0 * (x.y - x.x * x.x) * (x.y - x.x * x.x) + (1.0 - x.x) * (1.0 - x.x); };

grad g1 =
{
	[](vec2 x) { return -400.0 * x.x * (x.y - x.x * x.x) + 2.0 * (x.x - 1.0); },
	[](vec2 x) { return 200.0 * (x.y - x.x * x.x); }
};

hesse h1 =
{
	[](vec2 x) { return -400.0 * (x.y - 3.0 * x.x * x.x) + 2.0; },
	[](vec2 x) { return -400.0 * x.x; },
	[](vec2 x) { return -400.0 * x.x; },
	[](vec2 x) { return 200.0; }
};


double exp1(double x, double y)
{
	return exp(-(x - 1.0) * (x - 1.0) / 4.0 - (y - 1.0) * (y - 1.0));
}

double exp2(double x, double y)
{
	return exp(-(x - 2.0) * (x - 2.0) / 9.0 - (y - 3.0) * (y - 3.0) / 4.0);
}

func f2 = [](vec2 x) { return -(2.0 * exp1(x.x, x.y) + 3.0 * exp2(x.x, x.y)); };

grad g2 =
{
	[](vec2 x) { return -((1.0 - x.x) * exp1(x.x, x.y) + 2.0 / 3.0 * (2.0 - x.x) * exp2(x.x, x.y)); },
	[](vec2 x) { return -(4.0 * (1.0 - x.y) * exp1(x.x, x.y) + 3.0 / 2.0 * (3.0 - x.y) * exp2(x.x, x.y)); }
};

hesse h2 =
{
	[](vec2 x) { return -((0.5 * (1.0 - x.x) * (1.0 - x.x) - 1.0) * exp1(x.x, x.y) + 2.0 / 3.0 * (2.0 / 9.0 * (2.0 - x.x) * (2.0 - x.x) - 1.0) * exp2(x.x, x.y)); },
	[](vec2 x) { return -(2.0 * (1.0 - x.x) * (1.0 - x.y) * exp1(x.x, x.y) + 1.0 / 3.0 * (2.0 - x.x) * (3.0 - x.y) * exp2(x.x, x.y)); },
	[](vec2 x) { return -(2.0 * (1.0 - x.x) * (1.0 - x.y) * exp1(x.x, x.y) + 1.0 / 3.0 * (2.0 - x.x) * (3.0 - x.y) * exp2(x.x, x.y)); },
	[](vec2 x) { return -(4.0 * (2.0 * (1.0 - x.y) * (1.0 - x.y) - 1.0) * exp1(x.x, x.y) + 1.5 * (0.5 * (3.0 - x.y) * (3.0 - x.y) - 1.0) * exp2(x.x, x.y)); }
};

int main()
{
	newton_info info;
	double x1 = 1;					// Начальное значение x
	double y1 = 2;					// Начальное значение y
	info.x0 = vec2(x1, y1);
	info.f = f2;					// Функция
	info.g = g2;					// Градиент
	info.h = h2;					// Матрица вторых производных
	info.maxiter = 100000;			// Максимальное количество итераций
	info.minimize_eps = 1.0e-12;	// Эпсилон для одномерной минимизаций
	info.delta = 1.0e-12;			// Дельта для ||x(k+1) - x(k)||

	std::vector<vec2> points1, points2, points3;	// Вектор точек
	result_info res1, res2, res3;					// Количество итераций + количество вычислений функций

	std::ofstream out;
	out.open("result.csv");
	out << "Точность;Начальное значение x;Начальное значение y;Количество итераций;Количество вычислений;Найденное значение x;Найденное значение y;Значение функции" << std::endl;
	for (double eps = 1.0e-3; eps > 1.0e-7; eps *= 1.0e-1)
	{
		info.eps = eps;				// Эпсилон для |f(k+1) - f(k)|
		res1 = lambda_newton(info, points1);
		vec2 min1 = points1.back();

		// Вывод
		out << eps << ";" << x1 << ";" << y1 << ";" << res1.iter_count << ";" << res1.calc_count << ";" << min1.x << ";" << min1.y << ";" << info.f(min1) << std::endl;

		points1.clear();
	}
	out.close();

	broyden_info b_info;
	b_info.f = f2;
	b_info.g = g2;
	b_info.minimize_eps = 1.0e-12;
	b_info.maxiter = 100000;
	b_info.x0 = vec2(x1, y1);

	//std::ofstream out;
	//out.open("result.csv");
	//out << "Точность;Начальное значение x;Начальное значение y;Количество итераций;Количество вычислений;Найденное значение x;Найденное значение y;Значение функции" << std::endl;
	//for (double eps = 1.0e-3; eps > 1.0e-7; eps *= 1.0e-1)
	//{
	//	b_info.eps = eps;				// Эпсилон для |f(k+1) - f(k)|
	//	res3 = broyden(b_info, points3);
	//	vec2 min3;
	//	if (points3.size() >= 2)
	//		min3 = points3[points3.size() - 2];
	//	else
	//		min3 = points3.back();

	//	// Вывод
	//	out << eps << ";" << x1 << ";" << y1 << ";" << res3.iter_count << ";" << res3.calc_count << ";" << min3.x << ";" << min3.y << ";" << b_info.f(min3) << std::endl;

	//	points3.clear();
	//}
	//out.close();

	system("result.csv");
	return 0;
}