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

int DirectionSearch(function<double(vec2)> f, vec2& x0, vec2& dir, vec2& x1)
{
	int k = 0;

	double w = 1.0;
	double f0 = f(x0);
	k++;
	vec2 x;

	for (int i = 0; i < 4; i++)
	{
		int s = 1;
		x = x0 + s * w * dir;

		while (x.x < 10 && x.x > -10 && x.y < 10 && x.y > -10 && f(x) >= f0)
		{
			x = x0 + s * w * dir;
			s++;
			k++;
		}

		if (x.x < 10 && x.x > -10 && x.y < 10 && x.y > -10 && f(x) < f0)
		{
			k++;
			x1 = x;
			return k;
		}
		else
			w *= 0.1;
	}

	x1 = vec2(-1.0e+50, -1.0e+50);
	return k;
}

uint64_t SimpleRandomSearch(function<double(vec2)> f, vec2& x, double eps = 1.0e-2, double P = 0.8)
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
		}
	}

	return N;
}

int Algorithm1(function<double(vec2)> f, vec2& x, int m = 100)
{
	int k = 0;
	GaussResult res;

	vec2 x0 = vec2(RandomDouble(-10, 10), RandomDouble(-10, 10));
	vec2 x1;

	res = Gauss(f, x0, x1, 1.0e-7, 1.0e-7);
	k += res.calcCount;

	x = x1;
	double min = f(x1);
	k++;

	int badPoints = 0;
	while (badPoints < m)
	{
		vec2 x0 = vec2(RandomDouble(-10, 10), RandomDouble(-10, 10));
		res = Gauss(f, x0, x1, 1.0e-7, 1.0e-7);
		k += res.calcCount;

		double f0 = f(x1);
		k++;

		if (f0 < min)
		{
			min = f0;
			x = x1;
			badPoints = 0;
		}
		else
			badPoints++;
	}

	return k;
}

int Algorithm2(function<double(vec2)> f, vec2& x, int m = 100)
{
	int k = 0;
	GaussResult res;

	vec2 x0(-10, -10);
	vec2 x1;

	res = Gauss(f, x0, x1, 1.0e-7, 1.0e-7);
	k += res.calcCount;

	x = x1;
	double min = f(x1);
	k++;

	int badPoints = 0;
	while (badPoints < m)
	{
		vec2 x0 = vec2(RandomDouble(-10, 10), RandomDouble(-10, 10));
		double f0 = f(x0);
		k++;

		if (f0 < min)
		{
			res = Gauss(f, x0, x1, 1.0e-7, 1.0e-7);
			k += res.calcCount;
			x = x1;
			min = f(x1);
			k++;
			badPoints = 0;
		}
		else
			badPoints++;
	}

	return k;
}

int Algorithm3(function<double(vec2)> f, vec2& x, int m = 100)
{
	int k = 0;
	GaussResult res;

	vec2 x0(-10, -10);
	vec2 x1;

	res = Gauss(f, x0, x1, 1.0e-7, 1.0e-7);
	k += res.calcCount;
	x = x1;
	double min = f(x1);
	k++;

	int badPoints = 0;
	while (badPoints < m)
	{
		vec2 dir = vec2(RandomDouble(-10, 10), RandomDouble(-10, 10)) - x1;
		vec2 x2;
		k += DirectionSearch(f, x1, dir, x2);

		if (x2.x < 10 && x2.x > -10 && x2.y < 10 && x2.y > -10)
		{
			vec2 x3;
			res = Gauss(f, x2, x3, 1.0e-7, 1.0e-7);
			k += res.calcCount;

			double f0 = f(x3);
			k++;
			if (f0 < min)
			{
				x1 = x3;
				x = x3;
				min = f0;
				badPoints = 0;
			}
			else
				badPoints++;
		}
		else
			badPoints++;
	}
	
	return k;
}