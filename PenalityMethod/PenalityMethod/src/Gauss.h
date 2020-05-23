#pragma once

#include "OneDimMinimization.h"
#include "vec2.h"

struct GaussResult
{
	int iterCount = 0;
	int calcCount = 0;
};

GaussResult Gauss(std::function<double(vec2)>& f, vec2 x0, vec2& x1, double delta, double eps)
{
	GaussResult res;
	double f1 = f(x0); res.calcCount++;
	double f2 = 0.0;

	double diffX = 0.0, diffY = 0.0;

	do
	{
		res.calcCount += minimize(f, x0, vec2(0.01, 0), x1);
		res.calcCount += minimize(f, x1, vec2(0, 0.01), x1);

		f2 = f(x1); res.calcCount++;

		diffX = (x1 - x0).norm();
		x0 = x1;

		diffY = abs(f2 - f1);
		f1 = f(x0); res.calcCount++;

		res.iterCount++;
	}
	while (diffX >= delta && diffY >= eps);

	return res;
}