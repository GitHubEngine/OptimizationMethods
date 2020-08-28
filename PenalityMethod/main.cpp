#include <iostream>
#include <fstream>

#include "vec2.h"
#include "Gauss.h"

using namespace std;

std::function<double(vec2)> f = [](vec2 x) { return 5 * (x.x - x.y) * (x.x - x.y) + (x.x - 2.0) * (x.x - 2.0);  };
std::function<double(vec2)> g = [](vec2 x) { return x.x + x.y - 1.0;  };
std::function<double(vec2)> h = [](vec2 x) { return x.x + x.y; };

std::function<double(vec2)> G1 = [](vec2 x) { return  0.5 * (g(x) + abs(g(x))); };
std::function<double(vec2)> G2 = [](vec2 x) { return  0.25 * pow(g(x) + abs(g(x)), 2); };
std::function<double(vec2)> G3 = [](vec2 x) { return  0.0625 * pow(g(x) + abs(g(x)), 4); };

std::function<double(vec2)> H1 = [](vec2 x) { return abs(h(x)); };
std::function<double(vec2)> H2 = [](vec2 x) { return pow(h(x), 2); };
std::function<double(vec2)> H3 = [](vec2 x) { return pow(h(x), 4); };


std::function<double(vec2)> J1 = [](vec2 x) { return -1.0 / g(x); };
std::function<double(vec2)> J2 = [](vec2 x) { return -log(-g(x)); };


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

	std::ofstream out1("Gresult1.csv");
	std::ofstream out2("Gresult2.csv");
	std::ofstream out3("Gresult3.csv");
	std::ofstream out4("Gresult4.csv");
	std::ofstream out5("Gresult5.csv");

	std::ofstream Hout1("Hresult1.csv");
	std::ofstream Hout2("Hresult2.csv");
	std::ofstream Hout3("Hresult3.csv");
	std::ofstream Hout4("Hresult4.csv");
	std::ofstream Hout5("Hresult5.csv");

	std::ofstream Jout1("Jresult1.csv");
	std::ofstream Jout2("Jresult2.csv");
	std::ofstream Jout3("Jresult3.csv");
	std::ofstream Jout4("Jresult4.csv");
	std::ofstream Jout5("Jresult5.csv");

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

#pragma region Начальная точка
	x0 = vec2(-10, 15);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	out4 << "G1; r(k) = 1.3 * r(k - 1); r0 = 1" << endl;
	out4 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out4 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	out4 << endl << endl;

	x0 = vec2(2, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	out4 << "G1;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	out4 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out4 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	out4 << endl << endl;

	x0 = vec2(0.05, 0.2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	out4 << "G1;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	out4 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out4 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}
#pragma endregion

#pragma region Задаваемая точность
	eps = 1.0e-3;
	x0 = vec2(0.05, 0.2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	out5 << "G1; r(k) = 1.3 * r(k - 1); r0 = 1" << endl;
	out5 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out5 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	out5 << endl << endl;

	eps = 1.0e-5;
	x0 = vec2(0.05, 0.2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	out5 << "G1;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	out5 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out5 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	out5 << endl << endl;

	eps = 1.0e-6;
	x0 = vec2(0.05, 0.2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	out5 << "G1;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	out5 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * G1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		out5 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}
#pragma endregion

	eps = 1.0e-5;
#pragma region Выбор функции штрафа
	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Hout1 << "H1;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	Hout1 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout1 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	Hout1 << endl << endl;

	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Hout1 << "H2;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	Hout1 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout1 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	Hout1 << endl << endl;

	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Hout1 << "H3;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	Hout1 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H3(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout1 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}
#pragma endregion

#pragma region Начальная величина штрафа
	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 0.1;
	res = { 0 };

	Hout2 << "H1;r(k) = 1.3 * r(k - 1);r0 = 0.1" << endl;
	Hout2 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout2 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	Hout2 << endl << endl;

	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 3;
	res = { 0 };

	Hout2 << "H1;r(k) = 1.3 * r(k - 1);r0 = 3" << endl;
	Hout2 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout2 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	Hout2 << endl << endl;

	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 20;
	res = { 0 };

	Hout2 << "H1;r(k) = 1.3 * r(k - 1);r0 = 20" << endl;
	Hout2 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout2 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}
#pragma endregion

#pragma region Стратегия изменения коэффициентов штрафа
	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Hout3 << "H1;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	Hout3 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout3 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	Hout3 << endl << endl;

	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Hout3 << "H1;r(k) = r(k - 1) + 10;r0 = 1" << endl;
	Hout3 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout3 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r += 10;
	}

	Hout3 << endl << endl;

	x0 = vec2(1, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Hout3 << "H1;r(k) = r(k - 1) * r(k - 1) + 0.03;r0 = 1" << endl;
	Hout3 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout3 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r = r * r + 0.03;
	}
#pragma endregion

#pragma region Начальная точка
	x0 = vec2(-10, 15);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Hout4 << "H1;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	Hout4 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout4 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	Hout4 << endl << endl;

	x0 = vec2(2, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Hout4 << "H1;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	Hout4 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout4 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	Hout4 << endl << endl;

	x0 = vec2(0.05, 0.2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Hout4 << "H1;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	Hout4 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout4 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}
#pragma endregion

#pragma region Задаваемая точность
	eps = 1.0e-3;
	x0 = vec2(0.05, 0.2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Hout5 << "H1;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	Hout5 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout5 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	Hout5 << endl << endl;

	eps = 1.0e-5;
	x0 = vec2(0.05, 0.2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Hout5 << "H1;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	Hout5 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout5 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}

	Hout5 << endl << endl;

	eps = 1.0e-6;
	x0 = vec2(0.05, 0.2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Hout5 << "H1;r(k) = 1.3 * r(k - 1);r0 = 1" << endl;
	Hout5 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * H1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Hout5 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 1.3;
	}
#pragma endregion

	eps = 1.0e-5;
#pragma region Выбор функции штрафа
	x0 = vec2(-1, -2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Jout1 << "J1;r(k) = 0.8 * r(k - 1);r0 = 1" << endl;
	Jout1 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J1(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout1 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 0.8;
	}

	Jout1 << endl << endl;

	x0 = vec2(-1, -2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Jout1 << "J2;r(k) = 0.8 * r(k - 1);r0 = 1" << endl;
	Jout1 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout1 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 0.8;
	}
#pragma endregion

#pragma region Начальная величина штрафа
	x0 = vec2(-1, -2);
	x1 = vec2(0.0, 0.0);
	r = 0.1;
	res = { 0 };

	Jout2 << "J2;r(k) = 0.8 * r(k - 1);r0 = 0.1" << endl;
	Jout2 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout2 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 0.8;
	}

	Jout2 << endl << endl;

	x0 = vec2(-1, -2);
	x1 = vec2(0.0, 0.0);
	r = 3;
	res = { 0 };

	Jout2 << "J2;r(k) = 0.8 * r(k - 1);r0 = 3" << endl;
	Jout2 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout2 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 0.8;
	}

	Jout2 << endl << endl;

	x0 = vec2(-1, -2);
	x1 = vec2(0.0, 0.0);
	r = 20;
	res = { 0 };

	Jout2 << "J2;r(k) = 0.8 * r(k - 1);r0 = 20" << endl;
	Jout2 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout2 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 0.8;
	}
#pragma endregion

#pragma region Стратегия изменения коэффициентов штрафа
	x0 = vec2(-1, -2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Jout3 << "J2;r(k) = 0.8 * r(k - 1);r0 = 1" << endl;
	Jout3 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout3 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 0.8;
	}

	Jout3 << endl << endl;

	x0 = vec2(-1, -2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Jout3 << "J2;r(k) =  0.9 * r(k - 1) * r(k - 1);r0 = 1" << endl;
	Jout3 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout3 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r = 0.9 * r * r;
	}

	Jout3 << endl << endl;

	x0 = vec2(-1, -2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Jout3 << "J2;r(k) = sin(r(k - 1));r0 = 1" << endl;
	Jout3 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout3 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r = sin(r);
	}
#pragma endregion

#pragma region Начальная точка
	x0 = vec2(-10, 15);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Jout4 << "J2;r(k) = 0.8 * r(k - 1);r0 = 1" << endl;
	Jout4 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout4 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 0.8;
	}

	Jout4 << endl << endl;

	x0 = vec2(2, 2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Jout4 << "J2;r(k) = 0.8 * r(k - 1);r0 = 1" << endl;
	Jout4 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout4 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 0.8;
	}

	Jout4 << endl << endl;

	x0 = vec2(0.05, 0.2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Jout4 << "J2;r(k) = 0.8 * r(k - 1);r0 = 1" << endl;
	Jout4 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout4 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 0.8;
	}
#pragma endregion

#pragma region Задаваемая точность
	eps = 1.0e-3;
	x0 = vec2(-1, -2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Jout5 << "J2;r(k) = 0.8 * r(k - 1);r0 = 1" << endl;
	Jout5 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout5 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 0.8;
	}

	Jout5 << endl << endl;

	eps = 1.0e-5;
	x0 = vec2(-1, -2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Jout5 << "J2;r(k) = 0.8 * r(k - 1);r0 = 1" << endl;
	Jout5 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout5 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 0.8;
	}

	Jout5 << endl << endl;

	eps = 1.0e-6;
	x0 = vec2(-1, -2);
	x1 = vec2(0.0, 0.0);
	r = 1;
	res = { 0 };

	Jout5 << "J2;r(k) = 0.8 * r(k - 1);r0 = 1" << endl;
	Jout5 << "Нач. приближение;Коэффициент штрафа;Точность;Количество итераций;Число вычислений функции;Найденая точка;Значение функции в точке" << endl;

	for (int i = 0; i < 15; i++)
	{
		std::function<double(vec2)> func = [r](vec2 x) { return f(x) + r * J2(x); };
		res = Gauss(func, x0, x1, eps, eps);
		Jout5 << x0 << ";" << r << ";" << eps << ";" << res.iterCount << ";" << res.calcCount << ";" << x1 << ";" << f(x1) << endl;
		r *= 0.8;
	}
#pragma endregion

	out1.close();
	out2.close();
	out3.close();
	out4.close();
	out5.close();

	Hout1.close();
	Hout2.close();
	Hout3.close();
	Hout4.close();
	Hout5.close();

	Jout1.close();
	Jout2.close();
	Jout3.close();
	Jout4.close();
	Jout5.close();

	return 0;
}