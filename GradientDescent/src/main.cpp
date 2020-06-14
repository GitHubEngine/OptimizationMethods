#include <iostream>
#include <vector>
#include <cmath>

#include "Matrix.h"
#include "Newton.h"
#include "Broyden.h"

func f = [](vec2 x) { return 100 * (x.y - x.x) * (x.y - x.x) + (1 - x.x) * (1 - x.x); };

grad g =
{
	[](vec2 x) { return -200 * (x.y - x.x) - 2 * (1 - x.x); },
	[](vec2 x) { return 200 * (x.y - x.x); }
};

hesse h =
{
	[](vec2 x) { return 202; },
	[](vec2 x) { return -200; },
	[](vec2 x) { return -200; },
	[](vec2 x) { return 200; }
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
	return exp(-(x - 1) * (x - 1) / 4 - (y - 1) * (y - 1));
}

double exp2(double x, double y)
{
	return exp(-(x - 2) * (x - 2) / 9 - (y - 3) * (y - 3) / 4);
}

func f2 = [](vec2 x) { return -(2 * exp1(x.x, x.y) + 3 * exp2(x.x, x.y)); };

grad g2 =
{
	[](vec2 x) { return -((1 - x.x) * exp1(x.x, x.y) + 2.0 / 3 * (2 - x.x) * exp2(x.x, x.y)); },
	[](vec2 x) { return -(4 * (1 - x.y) * exp1(x.x, x.y) + 3.0 / 2 * (3 - x.y) * exp2(x.x, x.y)); }
};

hesse h2 =
{
	[](vec2 x) { return -((0.5 * (1 - x.x) * (1 - x.x) - 1) * exp1(x.x, x.y) + 2.0 / 3 * (2.0 / 9 * (2 - x.x) * (2 - x.x) - 1) * exp2(x.x, x.y)); },
	[](vec2 x) { return -(2 * (1 - x.x) * (1 - x.y) * exp1(x.x, x.y) + 1.0 / 3 * (2 - x.x) * (3 - x.y) * exp2(x.x, x.y)); },
	[](vec2 x) { return -(2 * (1 - x.x) * (1 - x.y) * exp1(x.x, x.y) + 1.0 / 3 * (2 - x.x) * (3 - x.y) * exp2(x.x, x.y)); },
	[](vec2 x) { return -(4 * (2 * (1 - x.y) * (1 - x.y) - 1) * exp1(x.x, x.y) + 1.5 * (0.5 * (3 - x.y) * (3 - x.y) - 1) * exp2(x.x, x.y)); }
};

int main()
{
	newton_info info;
	info.x0 = vec2(2, 2);
	info.f = f;
	info.g = g;
	info.h = h;
	info.maxiter = 10000;
	info.minimize_eps = 1.0e-1;
	info.eps = 1.0e-7;

	std::vector<vec2> points1, points2, points3;
	result_info res1, res2, res3;

	res1 = lambda_newton(info, points1);
	vec2 min1 = points1.back();

	return 0;
}