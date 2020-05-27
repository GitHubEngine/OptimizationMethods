#include <iostream>
#include <fstream>

#include "vec2.h"
#include "Gauss.h"

using namespace std;

std::function<double(vec2)> f = [](vec2 x) { return 5 * (x.x - x.y) * (x.x - x.y) + (x.x - 2.0) * (x.x - 2.0);  };

std::function<double(vec2)> G1 = [](vec2 x) { return  0.5 * (x.x + x.y - 1.0 + abs(x.x + x.y - 1.0)); };
std::function<double(vec2)> G2 = [](vec2 x) { return  0.25 * pow(x.x + x.y - 1.0 + abs(x.x + x.y - 1.0), 2); };
std::function<double(vec2)> G3 = [](vec2 x) { return  0.0625 * pow(x.x + x.y - 1.0 + abs(x.x + x.y - 1.0), 4); };

std::function<double(vec2)> H1 = [](vec2 x) { return abs(x.x + x.y); };
std::function<double(vec2)> H2 = [](vec2 x) { return pow(x.x + x.y, 2); };
std::function<double(vec2)> H3 = [](vec2 x) { return pow(x.x + x.y, 4); };


int main()
{
	/*
	Таблицы, где меняется:
	1. Выбор штрафных функций
	2. Начальная величина коэффициента штрафа
	3. Стратегия изменения коэффициентов штрафа
	4. Начальные точки
	5. Задаваемая точность eps
	*/

	double eps = 1.0e-5;
	vec2 x0, x1;
	double r;
	GaussResult res;

	std::ofstream out1("result1.csv");
	std::ofstream out2("result2.csv");
	std::ofstream out3("result3.csv");
	std::ofstream out4("result4.csv");
	std::ofstream out5("result5.csv");

#pragma region Выбор функции штрафа
	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	out1 << "G1; r(k) = 1.3 * r(k - 1); r0 = 1" << endl;
	out1 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out1 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	out1 << endl << endl;

	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	out1 << "G2;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	out1 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out1 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	out1 << endl << endl;

	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	out1 << "G3;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	out1 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G3(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out1 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}
#pragma endregion

#pragma region Начальная величина штрафа
	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 0.1;
	res = { 0 };

	out2 << "G1; r(k) = 1.3 * r(k - 1); r0 = 0.1" << endl;
	out2 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out2 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	out2 << endl << endl;

	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 3;
	res = { 0 };

	out2 << "G1;r(k) = 1.3 * r(k - 1);r0 = 3" << endl;
	out2 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out2 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	out2 << endl << endl;

	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 20;
	res = { 0 };

	out2 << "G1;r(k) = 1.3 * r(k - 1);r0 = 20" << endl;
	out2 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out2 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}
#pragma endregion

#pragma region Стратегия изменения коэффициентов штрафа
	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	out3 << "G1; r(k) = 1.3 * r(k - 1); r0 = 1" << endl;
	out3 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out3 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	out3 << endl << endl;

	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	out3 << "G1;r(k) = r(k - 1) + 10;r0 = 1" << endl;
	out3 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out3 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r += 10;
	}

	out3 << endl << endl;

	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	out3 << "G1;r(k) = r(k - 1) * r(k - 1) + 0.03;r0 = 1" << endl;
	out3 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out3 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r = r * r + 0.03;
	}
#pragma endregion

	out1.close();
	out2.close();
	out3.close();
	out4.close();
	out5.close();

	return 0;
}