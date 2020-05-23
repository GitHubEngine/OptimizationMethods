#include <iostream>

#include "vec2.h"
#include "Gauss.h"


int main()
{
	vec2 x0(0, 0.5), x1;

	double r1 = 0, r2 = 10;

	std::function<double(vec2)> f = [r1, r2](vec2 x) { return
		5 * (x.x - x.y) * (x.x - x.y) + (x.x - 2.0) * (x.x - 2.0) +
		r1 * (0.25 * (x.x + x.y - 1.0 + abs(x.x + x.y - 1.0)) * (x.x + x.y - 1.0 + abs(x.x + x.y - 1.0))) +
		r2 * abs(x.x + x.y); };


	GaussResult res = Gauss(f, x0, x1, 10e-5, 10e-5);

	return 0;
}