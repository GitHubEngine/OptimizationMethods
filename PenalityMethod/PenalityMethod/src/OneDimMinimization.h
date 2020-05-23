#pragma once

#include <functional>
#include <cmath>
#include "vec2.h"

int golden_ratio(std::function<double(vec2)> f, vec2& a, vec2& b, double eps, vec2& x)
{
	int k = 0;
	double diff = (b - a).norm();
	vec2 dir_normalized = (b - a).normalize();

	int n = static_cast<int>(log((diff) / eps) / log((sqrt(5) + 1) / 2));
	double prevDiff, ratio;

	vec2 x1 = a + dir_normalized * 0.381966011 * diff;
	vec2 x2 = b - dir_normalized * 0.381966011 * diff;

	double f1 = f(x1);
	double f2 = f(x2);
	k += 2;

	for (int i = 0; i <= n; i++)
	{
		if (f1 < f2)
		{
			b = x2;
			x2 = x1;
			x1 = a + (b - a) * 0.381966011;
			f2 = f1;
			f1 = f(x1);
			k++;
		}
		else
		{
			a = x1;
			x1 = x2;
			x2 = b - (b - a) * 0.381966011;
			f1 = f2;
			f2 = f(x2);
			k++;
		}

		prevDiff = diff;
		diff = (b - a).norm();

		ratio = prevDiff / diff;
	}

	x = (x1 + x2) / 2;

	return k;
}

int interval(std::function<double(vec2)> f, vec2& x0, const vec2& dir, vec2& b)
{
	int k = 0;
	int h = 1;
	vec2 x1 = x0 + dir;
	vec2 x2 = x0 - dir;
	double f0 = f(x0);
	double f1 = f(x1);
	double f2 = f(x2);
	k += 3;

	if (f0 > f1)
	{
		while (f0 > f1)
		{
			h *= 2;
			x1 = x0 + dir * h;
			f1 = f(x1);
			k++;
		}

		b = x1;
	}
	else if (f0 > f2)
	{
		while (f0 > f2)
		{
			h *= 2;
			x2 = x0 - dir * h;
			f2 = f(x2);
			k++;
		}

		b = x2;
	}


	return k;
}

int minimize(std::function<double(vec2)> f, vec2 a, const vec2& dir, vec2& b)
{
	int k1 = interval(f, a, dir, b);

	int k2 = 0;
	if ((b - a).norm() > 1.0e-10)
	{
		k2 = golden_ratio(f, a, b, 10e-7, b);
	}

	return k1 + k2;
}