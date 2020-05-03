#pragma once

#include <cmath>
#include "Matrix.h"

double golden_ratio(const func& f, vec2& a, vec2& b, const vec2& dir, double eps)
{
	double diff = (b - a).norm();
	vec2 dir_normalized = dir.normalize();

	int n = static_cast<int>(log((diff) / eps) / log((sqrt(5) + 1) / 2));
	double prevDiff, ratio;

	vec2 saved = a;

	vec2 x1 = a + dir_normalized * 0.381966011 * diff;
	vec2 x2 = b - dir_normalized * 0.381966011 * diff;

	double f1 = f(x1);
	double f2 = f(x2);

	for (int i = 0; i <= n; i++)
	{
		if (f1 < f2)
		{
			b = x2;
			x2 = x1;
			x1 = a + (b - a) * 0.381966011;
			f2 = f1;
			f1 = f(x1);
		}
		else
		{
			a = x1;
			x1 = x2;
			x2 = b - (b - a) * 0.381966011;
			f1 = f2;
			f2 = f(x2);
		}

		prevDiff = diff;
		diff = (b - a).norm();

		ratio = prevDiff / diff;
	}

	vec2 result = (x1 + x2) / 2;

	vec2 dist = result - saved;
	return dist.x / dir.x;
}

vec2 interval(const func& f, vec2& x0, const vec2& dir, double h)
{
	vec2 x1 = x0 + dir * h;
	double f0 = f(x0);
	double f1 = f(x1);

	while (f0 > f1)
	{
		h *= 2;
		x1 = x0 + dir * h;

		f0 = f(x0);
		f1 = f(x1);
	}

	return x1;
}

double minimize(const func& f, vec2 a, const vec2& dir, double eps)
{
	vec2 b = interval(f, a, dir, 1.0);
	return golden_ratio(f, a, b, dir, eps);
}
