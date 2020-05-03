#pragma once

#include <functional>
#include "Matrix.h"
#include "OneDimMinimizer.h"

struct newton_info {
	func f;
	grad g;
	hesse h;
	vec2 x0;
	double delta, eps, minimize_eps;
	int maxiter;
};


int lambda_newton(newton_info& info, vec2& min)
{
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

	while (diff_x >= delta && diff_f >= eps && count != maxiter)
	{
		// Calculate second partial derivative matrix
		mat2 H_inv = mat2(h, x0).inverse();

		// Calculate gradient
		vec2 grad(g, x0);

		// Calculate direction
		vec2 mult = H_inv * grad;
		vec2 dx = -mult.normalize();

		// Calculate step size
		double lambda = minimize(f, x0, dx, minimize_eps);

		// Calculate new approximation point
		vec2 x1 = x0 + lambda * dx;

		// Calculate ||x[k + 1] - x[k]||
		diff_x = (x1 - x0).norm();

		// Calculate |f[k + 1] - f[k]|
		diff_f = abs(f(x1) - f(x0));
		x0 = x1;
		count++;
	}

	min = x0;
	return count;
}

int modified_newton(newton_info& info, vec2& min)
{
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
		double lambda = minimize(f, x0, dx, minimize_eps);

		// Calculate new approximation point
		vec2 x1 = x0 + dx * lambda;

		// Calculate ||x[k + 1] - x[k]||
		diff_x = (x1 - x0).norm();

		// Calculate |f[k + 1] - f[k]|
		diff_f = abs(f(x1) - f(x0));
		x0 = x1;
		count++;
	}

	min = x0;
	return count;
}