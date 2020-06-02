#pragma once

#include <iostream>
#include <functional>
#include <random>
#include <cmath>
#include <cstdint>

#include "vec2.h"
#include "Gauss.h"

using namespace std;

double RandomDouble(double a, double b)
{
	random_device device;
	uniform_real_distribution<double> distribution(a, b);
	return distribution(device);
}

vec2 DirectionSearch(function<double(vec2)> f, vec2& x0, vec2& dir)
{
	double w = 1.0;
	double f0 = f(x0);
	vec2 x1;

	for (int i = 0; i < 4; i++)
	{
		int k = 1;
		x1 = x0 + k * w * dir;

		while (x1.x < 10 && x1.x > -10 && x1.y < 10 && x1.y > -10 && f(x1) >= f0)
		{
			x1 = x0 + k * w * dir;
			k++;
		}

		if (x1.x < 10 && x1.x > -10 && x1.y < 10 && x1.y > -10 && f(x1) < f0)
			return x1;
		else
			w *= 0.1;
	}

	return vec2(-1.0e+50, -1.0e+50);
}

void SimpleRandomSearch(function<double(vec2)> f, vec2& x, double eps = 1.0e-2, double P = 0.8)
{
	double Peps = eps * eps / (20 * 20);
	uint64_t N = static_cast<uint64_t>(log(1.0 - P) / log(1.0 - Peps));

	vec2 x0 = vec2(RandomDouble(-10, 10), RandomDouble(-10, 10));
	double min = f(x0);

	for (int i = 0; i < N; i++)
	{
		x0 = vec2(RandomDouble(-10, 10), RandomDouble(-10, 10));
		double f0 = f(x0);
		if (f0 < min)
		{
			min = f0;
			x = x0;
			cout << x0 << "\t" << f0 << "\t" << i << endl;
		}
	}
}

void Algorithm1(function<double(vec2)> f, vec2& x, int m = 100)
{
	vec2 x0 = vec2(RandomDouble(-10, 10), RandomDouble(-10, 10));
	vec2 x1;

	Gauss(f, x0, x1, 1.0e-7, 1.0e-7);
	x = x1;
	double min = f(x1);

	int badPoints = 0;
	while (badPoints < m)
	{
		vec2 x0 = vec2(RandomDouble(-10, 10), RandomDouble(-10, 10));
		Gauss(f, x0, x1, 1.0e-7, 1.0e-7);

		double f0 = f(x1);

		if (f0 < min)
		{
			min = f0;
			x = x1;
			badPoints = 0;
			cout << x0 << "\t" << f0 << endl;
		}
		else
		{
			badPoints++;
		}
	}
}

void Algorithm2(function<double(vec2)> f, vec2& x, int m = 100)
{
	vec2 x0(-10, -10);
	vec2 x1;

	Gauss(f, x0, x1, 1.0e-7, 1.0e-7);
	x = x1;
	double min = f(x1);
	cout << x1 << "\t" << min << endl;

	int badPoints = 0;
	while (badPoints < m)
	{
		vec2 x0 = vec2(RandomDouble(-10, 10), RandomDouble(-10, 10));
		double f0 = f(x0);

		if (f0 < min)
		{
			Gauss(f, x0, x1, 1.0e-7, 1.0e-7);
			x = x1;
			min = f(x1);
			badPoints = 0;
			cout << x0 << "\t" << f0 << endl;
		}
		else
		{
			badPoints++;
		}
	}
}

void Algorithm3(function<double(vec2)> f, vec2& x, int m = 100)
{
	vec2 x0(-10, -10);
	vec2 x1;

	Gauss(f, x0, x1, 1.0e-7, 1.0e-7);
	x = x1;
	double min = f(x1);
	cout << x1 << "\t" << min << endl;

	int badPoints = 0;
	while (badPoints < m)
	{
		vec2 dir = vec2(RandomDouble(-10, 10), RandomDouble(-10, 10)) - x1;
		vec2 x2 = DirectionSearch(f, x1, dir);

		if (x2.x < 10 && x2.x > -10 && x2.y < 10 && x2.y > -10)
		{
			vec2 x3;
			Gauss(f, x2, x3, 1.0e-7, 1.0e-7);

			double f0 = f(x3);
			if (f0 < min)
			{
				x1 = x3;
				x = x3;
				min = f0;
				badPoints = 0;

				cout << x << "\t" << min << endl;
			}
			else
			{
				badPoints++;
			}
		}
		else
		{
			badPoints++;
		}
	}
}