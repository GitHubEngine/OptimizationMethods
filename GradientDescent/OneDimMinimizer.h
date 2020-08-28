#pragma once

#include <functional>
#include <vector>
#include <cstdint>
#include <cmath>

int golden_ratio(std::function<double(double)> f, double a, double b, double& x, double eps = 1.0e-7)
{
	int k = 0;
	double diff = (b - a);

	int n = static_cast<int>(log((b - a) / eps) / log((sqrt(5) + 1) / 2));
	double prevDiff, ratio;

	double x1 = a + 0.381966011 * diff;
	double x2 = b - 0.381966011 * diff;

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
		diff = (b - a);

		ratio = prevDiff / diff;
	}

	x = (x1 + x2) / 2;

	return k;
}

int parabola(std::function<double(double)> f, double a, double b, double& x, double eps = 1.0e-7)
{
	int k = 0;
	double x1 = a, x2 = x1, x3 = b;
	double prevX = 0.0;
	bool found = false;

	do
	{
		prevX = x;
		double h = (x3 - x1) / 1000;

		double f1 = f(x1);
		double f3 = f(x3);
	k += 2;
		double f2 = 0.0;
		for (double xi = x1 + h; xi < x3 && !found; xi += h)
		{
			f2 = f(xi);
			k++;
			if (f2 <= f1 && f2 <= f3)
			{
				found = true;
				x2 = xi;
			}
		}


		if (found)
		{
			found = false;
			x = 0.5 * (x1 + x2 - ((f2 - f1) * (x3 - x2) / (x2 - x1)) / ((f3 - f1) / (x3 - x1) - (f2 - f1) / (x2 - x1)));

			double fx = f(x);
			k++;

			if (f2 < fx)
			{
				if (x > x2)
					x3 = x;
				else
					x1 = x;
			}
			else
			{
				if (x > x2)
					x1 = x2;
				else
					x3 = x2;
			}
		}
		else
		{
			if (f1 > f3)
				x = x3;
			else
				x = x1;

		}

	} while (abs(x - prevX) >= eps);

	return k;
}

int fibonacci(std::function<double(double)> f, double a, double b, double& x, double eps = 1.0e-7)
{
	int k = 0;
	std::vector<int64_t> F;

	F.push_back(1);
	F.push_back(1);
	int64_t Fn = 1;

	int n = 2;
	while (Fn < (b - a) / eps)
	{
		Fn = F[n - 1] + F[n - 2];
		F.push_back(Fn);
		n++;
	}

	n = F.size() - 2;

	double x1 = a + (b - a) * F[n - 1] / F[n + 1];
	double x2 = a + (b - a) * F[n] / F[n + 1];

	double f1 = f(x1);
	double f2 = f(x2);
	k += 2;

	for (int k = 1; k <= n; k++)
	{
		if (f1 < f2)
		{
			b = x2;
			x2 = x1;
			x1 = a + (b - a) * F[n - k] / F[n - k + 2];
			f2 = f1;
			f1 = f(x1);
			k++;
		}
		else
		{
			a = x1;
			x1 = x2;
			x2 = a + (b - a) * F[n - k + 1] / F[n - k + 2];
			f1 = f2;
			f2 = f(x2);
			k++;
		}
	}

	x = (x1 + x2) / 2;
	return k;
}

int interval(std::function<double(double)> f, double a, double& b)
{
	int k = 0;
	double h = 1.0;

	double x0 = a;
	double x1 = a + h;

	double f0 = f(x0);
	double f1 = f(x1);
	k += 2;

	while (f0 > f1)
	{
		h *= 2;
		x1 = x0 + h;

		f0 = f(x0);
		f1 = f(x1);
		k += 2;
	}

	b = x1;
	return k;
}

int minimize(std::function<double(double)> f, double a, double& min, double eps = 1.0e-12)
{
	double b = 0.0;
	int k1 = interval(f, a, b);
	int k2 = fibonacci(f, a, b, min, eps);

	return k1 + k2;
}