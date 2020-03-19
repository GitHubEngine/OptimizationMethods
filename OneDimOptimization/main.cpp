#include <fstream>
#include <vector> 
#include <math.h>

using namespace std;

const double pi = 3.14159265358979;

double f(double x)
{
	return sin(x);
}

void dichotomy(double a, double b, double eps, double delta, ofstream& out)
{
	int n = 0;
	double diff = b - a, prevDiff, ratio;

	out << "n;a;b;diff;ratio;x1;x2;f1;f2" << endl;

	while (diff > eps)
	{
		double x1 = (a + b - delta) / 2.0;
		double x2 = (a + b + delta) / 2.0;

		double f1 = f(x1);
		double f2 = f(x2);

		if (f1 < f2)
			b = x2;
		else
			a = x1;

		prevDiff = diff;
		diff = b - a;

		ratio = prevDiff / diff;
		n++;

		out << n << ";" << a << ";" << b << ";" << diff << ";" << ratio;
		out << ";" << x1 << ";" << x2 << ";" << f1 << ";" << f2 << endl;
	}
	out << endl;
}

void golden_ratio(double a, double b, double eps, ofstream& out)
{
	int n = 0;
	double diff = b - a;
	double prevDiff, ratio;

	out << "n;a;b;diff;ratio;x1;x2;f1;f2" << endl;

	double x1 = a + 0.381966011 * diff;
	double x2 = b - 0.381966011 * diff;

	double f1 = f(x1);
	double f2 = f(x2);

	while (fabs(diff) > eps)
	{
		if (f1 < f2)
		{
			b = x2;
			x2 = x1;
			x1 = a + 0.381966011 * (b - a);
			f2 = f1;
			f1 = f(x1);
		}
		else
		{
			a = x1;
			x1 = x2;
			x2 = b - 0.381966011 * (b - a);
			f1 = f2;
			f2 = f(x2);
		}

		prevDiff = diff;
		diff = b - a;

		ratio = prevDiff / diff;
		n++;

		out << n << ";" << a << ";" << b << ";" << diff << ";" << ratio;
		out << ";" << x1 << ";" << x2 << ";" << f1 << ";" << f2 << endl;
	}
	out << endl;
}

void fibonacci(double a, double b, double eps, ofstream& out)
{
	double x0 = 0, diff0 = b - a;
	double diff = b - a, prevDiff, ratio;
	vector<int64_t> F;

	out << "n;a;b;diff;ratio;x1;x2;f1;f2" << endl;

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

	for (int k = 1; k <= n; k++)
	{
		if (f1 < f2)
		{
			b = x2;
			x2 = x1;
			x1 = a + (b - a) * F[n - k] / F[n - k + 2];
			f2 = f1;
			f1 = f(x1);
		}
		else
		{
			a = x1;
			x1 = x2;
			x2 = a + (b - a) * F[n - k + 1] / F[n - k + 2];
			f1 = f2;
			f2 = f(x2);
		}

		prevDiff = diff;
		diff = b - a;

		ratio = prevDiff / diff;

		out << k << ";" << a << ";" << b << ";" << diff << ";" << ratio;
		out << ";" << x1 << ";" << x2 << ";" << f1 << ";" << f2 << endl;
	}

	out << fabs(x1 - x0) << endl;

}

void interval(double x0, double h, ofstream& out)
{
	int k = 1;
	double x1, xHelp;

	out << "n;x;f;x-h;x+h" << endl;

	double f0 = f(x0);
	double f1 = f(x0 + h);

	if (f0 > f1)
	{
		x1 = x0 + h;
		out << k << ";" << x1 << ";" << f1 << ";" << x1 - h << ";" << x1 + h << endl;
	}
	else
	{
		x1 = x0 - h;
		h = -h;
		out << k << ";" << x1 << ";" << f1 << ";" << x1 + h << ";" << x1 - h << endl;
	}

	h *= 2;
	x1 = x0 + h;

	f0 = f(x0);
	f1 = f(x1);

	x0 = x1;

	while (f0 > f1)
	{
		h *= 2;
		x1 = x0 + h;

		f0 = f(x0);
		f1 = f(x1);

		x0 = x1;

		k++;
		out << k << ";" << x1 << ";" << f1 << ";" << x1 - h << ";" << x1 + h << endl;
	}
	out << endl;
}

void main()
{
	ofstream out;
	double a = -pi / 2, b = pi / 2;
	out.open("dihotomy.csv");
	for (double eps = 1.E-1; eps > 1.E-7; eps *= 1.E-1)
	{
		out << eps << endl;
		double delta = eps / 3.;
		dichotomy(a, b, eps, delta, out);
	}
	out.close();

	out.open("golden_ratio.csv");
	for (double eps = 1.E-1; eps > 1.E-7; eps *= 1.E-1)
	{
		out << eps << endl;
		golden_ratio(a, b, eps, out);
	}
	out.close();

	out.open("fibonacci.csv");
	for (double eps = 1.E-1; eps > 1.E-7; eps *= 1.E-1)
	{
		out << eps << endl;
		fibonacci(a, b, eps, out);
	}
	out.close();

	out.open("interval.csv");
	interval(0.7, 0.01, out);
	out.close();
}