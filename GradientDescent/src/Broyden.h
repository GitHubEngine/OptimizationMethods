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

	CSV csv(100, 100);

	csv(0, 0, '(');
	csv(0, 0, x0.x);
	csv(0, 0, ',');
	csv(0, 0, x0.y);
	csv(0, 0, ')');

	csv(0, 1, f(x0));

	int line = 2;
	while (diff >= eps && result.iter_count < maxiter)
	{
		// Calculate gradient at x[k] point
		vec2 grad0(g, x0);

		// Calculate direction
		vec2 dx = -eta * grad0;
		dx.normalize();

		// Calculate step size
		double lambda = 0.0;
		result.calc_count += minimize([&](double l) { return f(x0 + l * dx); }, lambda, minimize_eps);

		// Calculate new approximation point
		vec2 x1 = x0 + lambda * dx;
		points.push_back(x1);

		csv(line, 0, '(');
		csv(line, 0, x1.x);
		csv(line, 0, ',');
		csv(line, 0, x1.y);
		csv(line, 0, ')');

		csv(line, 1, f(x1));

		csv(line - 2, 2, '(');
		csv(line - 2, 2, dx.x);
		csv(line - 2, 2, ',');
		csv(line - 2, 2, dx.y);
		csv(line - 2, 2, ')');

		csv(line - 2, 3, lambda);

		csv(line, 4, abs(x1.x - x0.x));
		csv(line, 4, ", ");
		csv(line, 4, abs(x1.y - x0.y));
		csv(line, 4, ", ");
		csv(line, 4, abs(f(x1) - f(x0)));

		csv(line - 2, 5, '(');
		csv(line - 2, 5, grad0.x);
		csv(line - 2, 5, ',');
		csv(line - 2, 5, grad0.y);
		csv(line - 2, 5, ')');

		csv(line - 2, 6, eta.a11);
		csv(line - 2, 7, eta.a12);

		csv(line - 1, 6, eta.a21);
		csv(line - 1, 7, eta.a22);
		line += 2;

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

	vec2 grad = vec2(g, x0);

	csv(line - 2, 5, '(');
	csv(line - 2, 5, grad.x);
	csv(line - 2, 5, ',');
	csv(line - 2, 5, grad.y);
	csv(line - 2, 5, ')');

	csv(line - 2, 6, eta.a11);
	csv(line - 2, 7, eta.a12);

	csv(line - 1, 6, eta.a21);
	csv(line - 1, 7, eta.a22);
	
	csv.Write("secondRes.csv");
		
	return result;
}