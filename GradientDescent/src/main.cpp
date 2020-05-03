#include <iostream>
#include <vector>
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

func f1 = [](vec2 x) { return x.x * x.x + x.y * x.y; };

grad g1 =
{
	[](vec2 x) { return 2 * x.x; },
	[](vec2 x) { return 2 * x.y; }
};

hesse h1 =
{
	[](vec2 x) { return 2; },
	[](vec2 x) { return 0; },
	[](vec2 x) { return 0; },
	[](vec2 x) { return 2; }
};

func f2 = [](vec2 x) { return pow(x.x, 4) + pow(x.y, 4); };

grad g2 =
{
	[](vec2 x) { return 4 * pow(x.x, 3); },
	[](vec2 x) { return 4 * pow(x.y, 3); }
};

hesse h2 =
{
	[](vec2 x) { return 4 * 3 * pow(x.x, 2); },
	[](vec2 x) { return 0; },
	[](vec2 x) { return 0; },
	[](vec2 x) { return 4 * 3 * pow(x.y, 2); }
};

int main()
{
	newton_info info;
	info.x0 = vec2(10.0, 156.3);
	info.f = f2;
	info.g = g2;
	info.h = h2;
	info.maxiter = 100000;
	info.minimize_eps = 1.0e-12;
	info.delta = 1.0e-12;
	info.eps = 1.0e-12;

	vec2 min1, min2;
	int k = lambda_newton(info, min1);
	k = modified_newton(info, min2);

	broyden_info b_info;
	b_info.f = f;
	b_info.g = g;
	b_info.eps = 1.0e-12;
	b_info.minimize_eps = 1.0e-12;
	b_info.maxiter = 100000;
	b_info.x0 = vec2(10.0, 40.0);

	vec2 min3;
	k = broyden(b_info, min3);
	return 0;
}