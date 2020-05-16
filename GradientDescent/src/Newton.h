#pragma once

#include <vector>
#include <functional>

#include "Matrix.h"
#include "OneDimMinimizer.h"
#include "ResultInfo.h"
#include "CSV.h"

struct newton_info {
	func f;
	grad g;
	hesse h;
	vec2 x0;
	double delta, eps, minimize_eps;
	int maxiter;
};


result_info lambda_newton(newton_info& info, std::vector<vec2>& points)
{
	result_info result;
	result.calc_count = 0;
	result.iter_count = 0;

	vec2 x0 = info.x0;
	func f = info.f;
	grad g = info.g;
	hesse h = info.h;
	double delta = info.delta;
	double eps = info.eps;
	double minimize_eps = info.minimize_eps;
	int maxiter = info.maxiter;

	double diff_x = 1.0;
	double diff_f = 1.0;

	points.push_back(x0);

	CSV csv(100, 100);

	csv(0, 0, '(');
	csv(0, 0, x0.x);
	csv(0, 0, ',');
	csv(0, 0, x0.y);
	csv(0, 0, ')');

	csv(0, 1, f(x0));

	int line = 2;
	while (diff_x >= delta && diff_f >= eps && result.iter_count != maxiter)
	{

		mat2 H = mat2(h, x0);
		// Calculate second partial derivative matrix
		mat2 H_inv = H.inverse();

		// Calculate gradient
		vec2 grad(g, x0);

		// Calculate direction
		vec2 mult = H_inv * grad;
		vec2 dx = -mult.normalize();

		// Calculate step size
		double lambda;
		result.calc_count += minimize(f, x0, dx, minimize_eps, lambda);

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
		csv(line - 2, 5, grad.x);
		csv(line - 2, 5, ',');
		csv(line - 2, 5, grad.y);
		csv(line - 2, 5, ')');

		csv(line - 2, 6, H.a11);
		csv(line - 2, 7, H.a12);

		csv(line - 1, 6, H.a21);
		csv(line - 1, 7, H.a22);

		line += 2;
		// Calculate ||x[k + 1] - x[k]||
		diff_x = (x1 - x0).norm();

		// Calculate |f[k + 1] - f[k]|
		diff_f = abs(f(x1) - f(x0));
		x0 = x1;
		result.iter_count++;
	}

	vec2 grad = vec2(g, x0);
	mat2 H = mat2(h, x0);

	csv(line - 2, 5, '(');
	csv(line - 2, 5, grad.x);
	csv(line - 2, 5, ',');
	csv(line - 2, 5, grad.y);
	csv(line - 2, 5, ')');

	csv(line - 2, 6, H.a11);
	csv(line - 2, 7, H.a12);

	csv(line - 1, 6, H.a21);
	csv(line - 1, 7, H.a22);
	
	//csv.Write("secondRes.csv");
	return result;
}

result_info modified_newton(newton_info& info, std::vector<vec2>& points)
{
	result_info result;
	result.calc_count = 0;
	result.iter_count = 0;

	vec2 x0 = info.x0;
	func f = info.f;
	grad g = info.g;
	hesse h = info.h;
	double delta = info.delta;
	double eps = info.eps;
	double minimize_eps = info.minimize_eps;
	int maxiter = info.maxiter;

	double diff_x = 1.0;
	double diff_f = 1.0;
	int count = 0;

	points.push_back(x0);

	// Calculate second partial derivative matrix at the point x0
	mat2 H_inv = mat2(h, x0).inverse();

	while (diff_x >= delta && diff_f >= eps && count != maxiter)
	{
		// Calculate gradient
		vec2 grad(g, x0);

		// Calculate direction
		vec2 mult = H_inv * grad;
		vec2 dx = -mult.normalize();

		// Calculate step size
		double lambda;
		result.calc_count += minimize(f, x0, dx, minimize_eps, lambda);

		// Calculate new approximation point
		vec2 x1 = x0 + dx * lambda;
		points.push_back(x1);

		// Calculate ||x[k + 1] - x[k]||
		diff_x = (x1 - x0).norm();

		// Calculate |f[k + 1] - f[k]|
		diff_f = abs(f(x1) - f(x0));
		x0 = x1;
		result.iter_count++;
	}

	return result;
}