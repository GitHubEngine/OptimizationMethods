#pragma once

#include <vector>
#include "Matrix.h"
#include "OneDimMinimizer.h"
#include "ResultInfo.h"

struct broyden_info 
{
	func f;
	grad g;
	vec2 x0;
	double eps;
	double minimize_eps;
	int maxiter;
};

result_info broyden(broyden_info& info, std::vector<vec2>& points)
{
	result_info result;
	result.calc_count = 0;
	result.iter_count = 0;

	vec2 x0 = info.x0;
	func f = info.f;
	grad g = info.g;
	double eps = info.eps;
	double minimize_eps = info.minimize_eps;
	int maxiter = info.maxiter;

	points.push_back(x0);

	vec2 grad0(g, x0);
	double diff = grad0.norm();
	// ¬ыбрали положительно определенную матрицу и x0
	mat2 eta(1.0);

	int count = 0;
	while (diff >= eps && count < maxiter)
	{
		// Calculate gradient at x[k] point
		vec2 grad0(g, x0);

		// Calculate direction
		vec2 dx = -eta * grad0;

		// Calculate step size
		double lambda;
		result.calc_count += minimize(f, x0, dx, minimize_eps, lambda);

		// Calculate new approximation point
		vec2 x1 = x0 + lambda * dx;
		points.push_back(x1);

		// Calculate difference 
		vec2 grad1 = vec2(g, x1);
		diff = grad1.norm();

		// Calculate delta eta
		vec2 delta_x = x1 - x0;
		vec2 delta_g = grad1 - grad0;
		vec2 temp = delta_x - eta * delta_g;

		mat2 numerator(temp.x * temp.x, temp.x * temp.y, temp.y * temp.x, temp.y * temp.y);
		double denominator = temp * delta_g;
		mat2 d_eta = numerator / denominator;

		eta = eta + d_eta;
		
		x0 = x1;
		result.iter_count++;
	}

	return result;
}