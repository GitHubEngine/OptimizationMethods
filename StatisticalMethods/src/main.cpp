#include <iostream>
#include <functional>
#include <vector>

#include "vec2.h"
#include "Statistics.h"

using namespace std;

vector<double> C = { 5, 5, 3, 6, 2, 3 };
vector<double> a = { -9, -10, 8, -6, -9, 1 };
vector<double> b = { -3, -1, 8, -5, -3, 5 };


function<double(vec2)> test = [](vec2 x) { return x.x * x.x + x.y * x.y; };

function<double(vec2)> f = [](vec2 x) 
{
	double sum = 0;
	for (int i = 0; i < 6; i++)
		sum += C[i] / (1 + (x.x - a[i]) * (x.x - a[i]) + (x.y - b[i]) * (x.y - b[i]));

	return -sum;
};

int main()
{
	vec2 x, x1, x2, x3;
	cout << "Simple" << endl;
	SimpleRandomSearch(f, x, 0.01, 0.6);
	cout << endl << endl;

	cout << "1" << endl;
	Algorithm1(f, x1, 1000);
	cout << endl << endl;

	cout << "2" << endl;
	Algorithm2(f, x2, 1000);
	cout << endl << endl;

	cout << "3" << endl;
	Algorithm3(f, x3, 1000);
	cout << endl << endl;

	return 0;
}